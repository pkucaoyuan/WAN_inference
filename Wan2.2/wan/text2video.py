# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from functools import partial
import concurrent.futures

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        
        # 并行加载两个专家模型以减少加载时间
        def load_expert_model(subfolder, model_name):
            print(f"📥 并行加载 {model_name}...")
            model = WanModel.from_pretrained(checkpoint_dir, subfolder=subfolder)
            return self._configure_model(
                model=model,
                use_sp=use_sp,
                dit_fsdp=dit_fsdp,
                shard_fn=shard_fn,
                convert_model_dtype=convert_model_dtype)
        
        # 多GPU环境下需要同步加载以避免竞争
        if dit_fsdp or use_sp:
            # 分布式环境：只有rank 0加载，然后广播
            if self.rank == 0:
                print(f"📥 主进程加载专家模型（分布式模式）...")
                self.low_noise_model = load_expert_model(config.low_noise_checkpoint, "低噪声专家")
                self.high_noise_model = load_expert_model(config.high_noise_checkpoint, "高噪声专家")
                print(f"✅ 主进程专家模型加载完成")
            else:
                # 其他进程等待并加载相同模型
                self.low_noise_model = load_expert_model(config.low_noise_checkpoint, "低噪声专家")
                self.high_noise_model = load_expert_model(config.high_noise_checkpoint, "高噪声专家")
            
            # 同步所有进程
            if dist.is_initialized():
                dist.barrier()
                if self.rank == 0:
                    print(f"✅ 所有进程专家模型同步完成")
        else:
            # 单GPU环境：使用并行加载
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                low_noise_future = executor.submit(
                    load_expert_model, config.low_noise_checkpoint, "低噪声专家")
                high_noise_future = executor.submit(
                    load_expert_model, config.high_noise_checkpoint, "高噪声专家")
                
                # 等待加载完成
                self.low_noise_model = low_noise_future.result()
                self.high_noise_model = high_noise_future.result()
                print(f"✅ 专家模型并行加载完成")
        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt
        self.total_switch_time = 0.0  # 记录总的专家切换时间
        self.step_timings = []  # 记录每步推理时间

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            # 检查是否需要切换
            current_device = next(getattr(self, required_model_name).parameters()).device.type
            need_switch = current_device == 'cpu'
            
            if need_switch and self.rank == 0:
                switch_start = time.time()
                print(f"🔄 专家切换: {required_model_name} (t={t.item():.0f})")
            
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if current_device == 'cpu':
                getattr(self, required_model_name).to(self.device)
                
            if need_switch and self.rank == 0:
                switch_time = time.time() - switch_start
                self.total_switch_time += switch_time
                print(f"⏱️ 专家切换耗时: {switch_time:.3f}秒")
                
        return getattr(self, required_model_name)

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 cfg_truncate_steps=5,
                 cfg_truncate_high_noise_steps=3,
                 output_dir=None,
                 enable_half_frame_generation=False,
):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            cfg_truncate_steps (`int`, *optional*, defaults to 5):
                Number of final steps to skip conditional forward pass (CFG truncate).
                In the last N steps, only unconditional prediction is used to speed up inference.
            cfg_truncate_high_noise_steps (`int`, *optional*, defaults to 3):
                Number of final steps in high-noise phase to skip conditional forward pass.
                Applied before switching to low-noise expert.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            
            # 帧数减半优化：第一个专家只生成一半帧数
            original_seq_len = seq_len
            if enable_half_frame_generation:
                # 计算减半后的序列长度
                half_seq_len = seq_len // 2
                if self.rank == 0:
                    print(f"🎬 帧数减半优化: 第一个专家生成{half_seq_len}帧，最终补齐到{seq_len}帧")
            else:
                half_seq_len = seq_len

            arg_c = {'context': context, 'seq_len': half_seq_len}
            arg_null = {'context': context_null, 'seq_len': half_seq_len}


            import time
            for step_idx, t in enumerate(tqdm(timesteps)):
                step_start_time = time.time()  # 记录每步开始时间
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                # 双重CFG Truncate策略
                is_final_steps = step_idx >= (len(timesteps) - cfg_truncate_steps)
                
                # 检查是否在高噪声专家的最后几步
                is_high_noise_phase = t.item() >= boundary
                high_noise_steps = [i for i, ts in enumerate(timesteps) if ts.item() >= boundary]
                is_high_noise_final = (is_high_noise_phase and 
                                     step_idx >= (max(high_noise_steps) - cfg_truncate_high_noise_steps + 1))
                
                # 准备模型调用参数
                model_kwargs_c = {**arg_c}
                model_kwargs_null = {**arg_null}
                
                if is_final_steps or is_high_noise_final:
                    # CFG截断：跳过条件前向传播
                    if self.rank == 0:
                        if is_final_steps:
                            print(f"低噪声专家CFG截断: Step {step_idx+1}/{len(timesteps)}, t={t.item()}")
                        else:
                            print(f"高噪声专家CFG截断: Step {step_idx+1}/{len(timesteps)}, t={t.item()}")
                    
                    # 只计算无条件预测
                    noise_pred_uncond = model(
                        latent_model_input, timestep, **model_kwargs_null)
                    noise_pred = noise_pred_uncond
                else:
                    # 正常CFG计算
                    noise_pred_cond = model(
                        latent_model_input, timestep, **model_kwargs_c)
                    noise_pred_uncond = model(
                        latent_model_input, timestep, **model_kwargs_null)
                    
                    # CFG引导
                    noise_pred = noise_pred_uncond + sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond)
                # 使用scheduler进行去噪步骤
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
                
                # 记录每步推理时间
                step_end_time = time.time()
                step_duration = step_end_time - step_start_time
                step_timing = {
                    'step': step_idx + 1,
                    'timestep': t.item(),
                    'duration': step_duration,
                    'is_high_noise': is_high_noise_phase,
                    'expert': 'high_noise' if is_high_noise_phase else 'low_noise'
                }
                self.step_timings.append(step_timing)
        
        # 帧数减半优化：在第一个专家完成后补齐帧数
        if enable_half_frame_generation and latents[0].shape[1] < original_seq_len:
            if self.rank == 0:
                print(f"🔄 帧数补齐: 从{latents[0].shape[1]}帧补齐到{original_seq_len}帧")
            
            # 每一帧复制自己插入到自己后面，最后一帧不需要复制
            current_frames = latents[0].shape[1]  # 当前帧数（减半后）
            target_frames = original_seq_len      # 目标帧数（原始）
            
            # 创建新的latents tensor
            new_latents = torch.zeros_like(latents[0][:, :target_frames, :, :])
            
            # 每一帧复制自己插入到自己后面
            for i in range(current_frames):
                # 原始帧
                new_latents[:, i*2, :, :] = latents[0][:, i, :, :]
                # 复制帧（除了最后一帧）
                if i*2+1 < target_frames:
                    new_latents[:, i*2+1, :, :] = latents[0][:, i, :, :]
            
            # 更新latents
            latents[0] = new_latents
            
            if self.rank == 0:
                print(f"✅ 帧数补齐完成: {latents[0].shape[1]}帧 (每帧复制插入)")
        
        # 生成推理报告
        if self.rank == 0:
            print(f"✅ 推理完成: {len(self.step_timings)}步")
        # 解码latents为视频
        x0 = latents
        if offload_model:
            self.low_noise_model.cpu()
            self.high_noise_model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        # 返回结果和时间信息
        result_videos = videos[0] if self.rank == 0 else None
        timing_info = {
            'total_switch_time': getattr(self, 'total_switch_time', 0.0),
            'step_timings': getattr(self, 'step_timings', [])
        }
        return result_videos, timing_info
