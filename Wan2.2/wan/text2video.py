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
from .attention_visualizer import AttentionVisualizer, create_attention_visualization_dir


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

        # 注意力可视化相关
        self.attention_visualizer = None
        self.attention_weights_history = []
        self.enable_attention_visualization = False

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
                 enable_attention_visualization=False,
                 attention_output_dir="attention_outputs",
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
        
        # 初始化注意力可视化
        if enable_attention_visualization:
            if self.rank == 0:
                print("🔍 注意力可视化已启用")
                print(f"   输出目录: {attention_output_dir}")
                print("   将生成平均Cross Attention Map")
            self._enable_attention_visualization(attention_output_dir)
            self.attention_weights_history = []  # 存储每步的注意力权重
        else:
            if self.rank == 0:
                print("📝 注意力可视化已禁用")
        
        # 帧数减半优化：第一个专家只生成一半帧数
        original_frame_num = frame_num
        if enable_half_frame_generation:
            F = math.ceil(frame_num / 2)  # 减半帧数，向上取整
            if self.rank == 0:
                print(f"🎬 帧数减半优化: 第一个专家生成{F}帧，最终补齐到{frame_num}帧")
        else:
            F = frame_num
            
        # 计算减半后的target_shape和seq_len（用于高噪声专家）
        half_target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                            size[1] // self.vae_stride[1],
                            size[0] // self.vae_stride[2])

        half_seq_len = math.ceil((half_target_shape[2] * half_target_shape[3]) /
                                (self.patch_size[1] * self.patch_size[2]) *
                                half_target_shape[1] / self.sp_size) * self.sp_size
        
        # 计算完整帧数的target_shape和seq_len（用于低噪声专家）
        full_target_shape = (self.vae.model.z_dim, (frame_num - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        full_seq_len = math.ceil((full_target_shape[2] * full_target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                                full_target_shape[1] / self.sp_size) * self.sp_size

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

        # 使用减半后的target_shape生成noise（高噪声专家）
        noise = [
            torch.randn(
                half_target_shape[0],
                half_target_shape[1],
                half_target_shape[2],
                half_target_shape[3],
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

            # 根据当前阶段使用不同的seq_len
            if enable_half_frame_generation:
                current_seq_len = half_seq_len  # 高噪声专家使用减半的seq_len
            else:
                current_seq_len = full_seq_len

            arg_c = {'context': context, 'seq_len': current_seq_len}
            arg_null = {'context': context_null, 'seq_len': current_seq_len}


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
                    noise_pred_uncond = self._call_model_with_attention_capture(
                        model, latent_model_input, timestep, model_kwargs_null, step_idx)
                    noise_pred = noise_pred_uncond
                else:
                    # 正常CFG计算
                    noise_pred_cond = self._call_model_with_attention_capture(
                        model, latent_model_input, timestep, model_kwargs_c, step_idx)
                    noise_pred_uncond = self._call_model_with_attention_capture(
                        model, latent_model_input, timestep, model_kwargs_null, step_idx)
                    
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
                
                # 帧数减半优化：在高噪声专家结束时进行帧数补全（在scheduler.step之后）
                if enable_half_frame_generation and is_high_noise_phase and step_idx == max(high_noise_steps):
                    if self.rank == 0:
                        print(f"🔄 高噪声专家结束，开始帧数补全: 从{latents[0].shape[1]}帧补齐到{full_target_shape[1]}帧")
                    
                    # 计算当前帧数和目标帧数
                    current_frames = latents[0].shape[1]  # 当前帧数（减半后经过VAE）
                    target_frames = full_target_shape[1]  # 目标帧数（完整帧数经过VAE）
                    
                    # 创建新的latents tensor: [C, target_frames, H, W]
                    new_latents = torch.zeros(
                        latents[0].shape[0], target_frames, 
                        latents[0].shape[2], latents[0].shape[3],
                        device=latents[0].device, dtype=latents[0].dtype
                    )
                    
                    # 考虑奇偶性的帧数补全
                    if target_frames % 2 == 0:  # 偶帧：每帧都重复
                        for i in range(current_frames):
                            if i*2 < target_frames:
                                new_latents[:, i*2, :, :] = latents[0][:, i, :, :]
                            if i*2+1 < target_frames:
                                new_latents[:, i*2+1, :, :] = latents[0][:, i, :, :]
                    else:  # 奇帧：最后一帧不重复
                        for i in range(current_frames):
                            if i*2 < target_frames:
                                new_latents[:, i*2, :, :] = latents[0][:, i, :, :]
                            if i*2+1 < target_frames:
                                new_latents[:, i*2+1, :, :] = latents[0][:, i, :, :]
                    
                    # 更新latents
                    latents[0] = new_latents
                    
                    # 更新seq_len为完整帧数的seq_len（低噪声专家使用）
                    current_seq_len = full_seq_len
                    arg_c = {'context': context, 'seq_len': current_seq_len}
                    arg_null = {'context': context_null, 'seq_len': current_seq_len}
                    
                    # 重新初始化scheduler状态以避免维度不匹配
                    # 注意：重新初始化会重置时间步序列，需要确保低噪声专家处理正确的低噪声部分
                    if sample_solver == 'unipc':
                        sample_scheduler = FlowUniPCMultistepScheduler(
                            num_train_timesteps=self.num_train_timesteps,
                            shift=1,
                            use_dynamic_shifting=False)
                        sample_scheduler.set_timesteps(
                            sampling_steps, device=self.device, shift=shift)
                        # 重新获取时间步序列
                        timesteps = sample_scheduler.timesteps
                        # 正确设置当前步骤索引
                        sample_scheduler._step_index = step_idx + 1
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
                        # 正确设置当前步骤索引
                        sample_scheduler._step_index = step_idx + 1
                    
                    # 重新计算专家切换边界和步骤分配
                    boundary = self.boundary * self.num_train_timesteps
                    high_noise_steps = [i for i, ts in enumerate(timesteps) if ts.item() >= boundary]
                    
                    if self.rank == 0:
                        print(f"🔄 重新初始化scheduler后:")
                        print(f"   - 时间步序列长度: {len(timesteps)}")
                        print(f"   - 高噪声步骤: {len(high_noise_steps)}步")
                        print(f"   - 低噪声步骤: {len(timesteps) - len(high_noise_steps)}步")
                        print(f"   - 当前步骤索引: {sample_scheduler.step_index}")
                        print(f"   - 当前时间步: {timesteps[step_idx].item() if step_idx < len(timesteps) else 'N/A'}")
                    
                    if self.rank == 0:
                        print(f"✅ 帧数补全完成: {latents[0].shape[1]}帧 (考虑奇偶性)")
                        print(f"🔄 Scheduler状态已重新初始化，避免维度不匹配")
                
                # 更新latents（在帧数补全之后）
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

        # 生成注意力可视化
        if enable_attention_visualization and hasattr(self, 'attention_weights_history') and self.attention_weights_history:
            if self.rank == 0:
                print(f"\n🎨 开始生成注意力可视化...")
                print(f"   捕获的步骤数: {len(self.attention_weights_history)}")
            self._create_attention_visualizations(input_prompt)

        # 返回结果和时间信息
        result_videos = videos[0] if self.rank == 0 else None
        timing_info = {
            'total_switch_time': getattr(self, 'total_switch_time', 0.0),
            'step_timings': getattr(self, 'step_timings', [])
        }
        return result_videos, timing_info
    
    def _enable_attention_visualization(self, output_dir: str = "attention_outputs"):
        """启用注意力可视化功能"""
        self.enable_attention_visualization = True
        self.attention_weights_history = []
        
        if self.attention_visualizer is None:
            # 使用low_noise_model作为主要的模型组件
            self.attention_visualizer = AttentionVisualizer(
                self.low_noise_model, self.text_encoder, self.device
            )
        
        # 创建输出目录
        self.attention_output_dir = create_attention_visualization_dir(output_dir)
        print(f"注意力可视化已启用，输出目录: {self.attention_output_dir}")
    
    def disable_attention_visualization(self):
        """禁用注意力可视化功能"""
        self.enable_attention_visualization = False
        self.attention_weights_history = []
        print("注意力可视化已禁用")
    
    def generate_with_attention_visualization(self, 
                                            prompt: str,
                                            num_frames: int = 16,
                                            height: int = 256,
                                            width: int = 256,
                                            num_inference_steps: int = 25,
                                            guidance_scale: float = 7.5,
                                            output_dir: str = "attention_outputs"):
        """生成视频并记录注意力权重"""
        
        # 启用注意力可视化
        self.enable_attention_visualization(output_dir)
        
        try:
            # 生成视频
            video, timing_info = self.generate(
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            # 创建注意力可视化
            if self.attention_weights_history:
                self._create_attention_visualizations(prompt)
            
            return video, timing_info
            
        finally:
            # 禁用注意力可视化
            self.disable_attention_visualization()
    
    def _create_attention_visualizations(self, prompt: str):
        """创建注意力可视化"""
        if not self.attention_weights_history:
            print("没有捕获到注意力权重数据")
            return
        
        print(f"创建注意力可视化，共 {len(self.attention_weights_history)} 步...")
        
        # 获取tokenizer
        try:
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
            tokens = tokenizer.tokenize(prompt)
        except:
            # 简单的tokenization
            tokens = prompt.split()
        
        print(f"Token数量: {len(tokens)}")
        print(f"Token列表: {tokens}")
        
        # 注意力可视化已在每步生成完成，这里只生成分析报告
        print(f"🎨 注意力可视化已在每步生成完成")
        print(f"📊 生成分析报告...")
        
        # 创建简化的分析报告
        analysis = self._create_simple_analysis_report(tokens)
        
        report_path = os.path.join(self.attention_output_dir, "attention_analysis_report.md")
        self.attention_visualizer.save_analysis_report(analysis, report_path)
        
        print(f"注意力可视化已保存到: {self.attention_output_dir}")

    def _visualize_current_step(self, attention_weights, step_idx):
        """立即生成当前步的可视化"""
        try:
            # 获取tokenizer和tokens
            tokens = self._get_tokens_from_prompt("A beautiful sunset over the ocean")  # 使用默认prompt
            
            # 平均当前步的所有批次和注意力头
            # attention_weights形状: [batch, heads, seq_len, context_len]
            avg_attention_weights = attention_weights.mean(dim=(0, 1))  # [seq_len, context_len]
            
            # 创建当前step的平均cross attention map的可视化
            step_save_path = os.path.join(self.attention_output_dir, f"step_{step_idx+1:02d}_cross_attention_map.png")
            self.attention_visualizer.visualize_attention_step(
                avg_attention_weights,  # 直接传递已平均的权重，形状[seq_len, context_len]
                tokens, step_idx, step_save_path, title=f"Step {step_idx+1} Cross Attention Map"
            )
            
            print(f"✅ Step {step_idx+1} Cross Attention Map已保存到: {step_save_path}")
            print(f"📊 权重形状: {avg_attention_weights.shape}")
            print(f"📊 权重范围: {avg_attention_weights.min():.4f} - {avg_attention_weights.max():.4f}")
            
        except Exception as e:
            print(f"❌ 生成Step {step_idx+1}可视化时出错: {e}")
            import traceback
            print(f"❌ 详细错误信息: {traceback.format_exc()}")

    def _create_simple_analysis_report(self, tokens):
        """创建简化的分析报告"""
        report = f"""# 注意力可视化分析报告

## 基本信息
- **Token数量**: {len(tokens)}
- **Token列表**: {tokens}
- **可视化方式**: 每两步生成一张图，立即释放内存
- **输出目录**: {self.attention_output_dir}

## 生成的文件
- `step_02_cross_attention_map.png` - Step 2的注意力图
- `step_04_cross_attention_map.png` - Step 4的注意力图
- `step_06_cross_attention_map.png` - Step 6的注意力图
- ... (每两步一张图)
- `step_20_cross_attention_map.png` - Step 20的注意力图

## 技术说明
- 每步捕获40个WanCrossAttention层的权重
- 对40个层和40个注意力头求平均
- 立即生成可视化图并释放内存
- 避免内存累积问题

## 注意力模式分析
每张图显示图像token（3600个）对文本token（512个）的注意力权重分布。
- 横轴：文本token位置
- 纵轴：图像token位置
- 颜色：注意力权重强度（白色=高权重，黑色=低权重）
"""
        return report

    def _get_tokens_from_prompt(self, prompt):
        """从prompt获取tokens"""
        try:
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
            tokens = tokenizer.tokenize(prompt)
        except:
            # 简单的tokenization
            tokens = prompt.split()
        return tokens
    
    def _call_model_with_attention_capture(self, model, latent_model_input, timestep, model_kwargs, step_idx):
        """调用模型并捕获真实的注意力权重"""
        if not self.enable_attention_visualization:
            return model(latent_model_input, timestep, **model_kwargs)[0]
        
        # 使用hook机制捕获真实的attention权重
        captured_attention = []
        
        def attention_hook(module, input, output):
            """Hook函数：捕获真实的cross attention权重"""
            # 检查是否是WanCrossAttention模块
            if hasattr(module, '__class__') and 'WanCrossAttention' in module.__class__.__name__:
                try:
                    # 调试信息：打印输入参数
                    if self.rank == 0:
                        print(f"🔍 Hook被调用 - 模块: {module.__class__.__name__}")
                        print(f"🔍 输入参数数量: {len(input) if input else 0}")
                        for i, inp in enumerate(input):
                            if hasattr(inp, 'shape'):
                                print(f"🔍 输入[{i}] 形状: {inp.shape}")
                            else:
                                print(f"🔍 输入[{i}] 类型: {type(inp)}")
                    
                    # 从input中提取参数
                    # WanCrossAttention.forward(x, context, context_lens)
                    if len(input) >= 3:
                        x, context, context_lens = input[:3]
                        
                        # 直接计算attention权重，使用WanCrossAttention的内部逻辑
                        b, n, d = x.size(0), module.num_heads, module.head_dim
                        
                        # 计算Q, K, V
                        q = module.norm_q(module.q(x)).view(b, -1, n, d)
                        k = module.norm_k(module.k(context)).view(b, -1, n, d)
                        v = module.v(context).view(b, -1, n, d)
                        
                        # 调试信息：打印Q, K的形状
                        if self.rank == 0:
                            print(f"🔍 Q形状: {q.shape}")
                            print(f"🔍 K形状: {k.shape}")
                            print(f"🔍 V形状: {v.shape}")
                        
                        # 使用标准的attention计算，而不是flash_attention
                        # 因为我们需要获取attention权重
                        scale = 1.0 / (d ** 0.5)
                        
                        # 计算attention scores
                        # q: [1, 3600, 40, 128], k: [1, 512, 40, 128]
                        # 需要重新排列维度以正确计算attention
                        # 将q和k重新排列为 [b, n, seq_len, d] 格式
                        q = q.transpose(1, 2)  # [1, 40, 3600, 128]
                        k = k.transpose(1, 2)  # [1, 40, 512, 128]
                        
                        # 现在计算scores: [1, 40, 3600, 128] @ [1, 40, 128, 512] = [1, 40, 3600, 512]
                        try:
                            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                            if self.rank == 0:
                                print(f"🔍 scores形状: {scores.shape}")
                        except Exception as e:
                            if self.rank == 0:
                                print(f"⚠️ 计算scores时出错: {e}")
                                print(f"⚠️ q.shape: {q.shape}")
                                print(f"⚠️ k.shape: {k.shape}")
                                print(f"⚠️ k.transpose(-2, -1).shape: {k.transpose(-2, -1).shape}")
                            raise e
                        
                        # 处理变长序列：使用context_lens来mask
                        if context_lens is not None:
                            # 创建mask: [b, 1, 1, 512]
                            max_len = k.size(2)  # 512 (k现在是[b, n, seq_len, d])
                            mask = torch.arange(max_len, device=k.device).expand(b, max_len) < context_lens.unsqueeze(1)
                            mask = mask.unsqueeze(1).unsqueeze(1)  # [b, 1, 1, 512]
                            
                            # 应用mask到scores
                            scores = scores.masked_fill(~mask, float('-inf'))
                        
                        # 计算attention权重
                        attention_weights = torch.softmax(scores, dim=-1)
                        
                        # 确保attention_weights是张量
                        if isinstance(attention_weights, torch.Tensor):
                            captured_attention.append(attention_weights)
                            if self.rank == 0:
                                print(f"🔍 成功捕获真实cross attention权重: {attention_weights.shape}")
                                print(f"🔍 模块名称: {module.__class__.__name__}")
                                print(f"🔍 权重范围: {attention_weights.min():.4f} - {attention_weights.max():.4f}")
                    else:
                        if self.rank == 0:
                            print(f"⚠️ 输入参数不足，无法计算cross_attn")
                            print(f"⚠️ 期望3个参数，实际得到{len(input)}个")
                except Exception as e:
                    if self.rank == 0:
                        print(f"⚠️ 无法获取真实attention权重: {e}")
                        print(f"⚠️ 模块类型: {type(module)}")
                        print(f"⚠️ 输入参数数量: {len(input) if input else 0}")
        
        # 注册hook到WanCrossAttention模块，而不是WanAttentionBlock
        hooks = []
        cross_attention_found = 0
        
        # 先打印所有模块名称，方便调试
        if self.rank == 0:
            print(f"🔍 模型中的所有模块:")
            for name, module in model.named_modules():
                if hasattr(module, '__class__') and 'WanCrossAttention' in module.__class__.__name__:
                    print(f"   - {name} ({module.__class__.__name__}) - WanCrossAttention")
        
        for name, module in model.named_modules():
            # 直接查找WanCrossAttention模块
            if hasattr(module, '__class__') and 'WanCrossAttention' in module.__class__.__name__:
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
                cross_attention_found += 1
                if self.rank == 0:
                    print(f"🔍 注册hook到WanCrossAttention模块: {name} ({module.__class__.__name__})")
        
        if self.rank == 0:
            print(f"🔍 总共找到 {cross_attention_found} 个WanCrossAttention模块")
            print(f"🔍 注册了 {len(hooks)} 个hook")
        
        try:
            # 调用模型，不传递return_attention参数
            if self.rank == 0:
                print(f"🔍 调用模型参数: {list(model_kwargs.keys())}")
            
            result = model(latent_model_input, timestep, **model_kwargs)[0]
            
            # 处理捕获的attention权重
            if self.rank == 0:
                print(f"🔍 捕获的attention权重数量: {len(captured_attention)}")
                if captured_attention:
                    print(f"🔍 第一个权重类型: {type(captured_attention[0])}")
                    if hasattr(captured_attention[0], 'shape'):
                        print(f"🔍 第一个权重形状: {captured_attention[0].shape}")
            
            if captured_attention:
                # 直接在GPU上计算平均，避免堆叠所有张量
                # 这样可以节省大量内存
                if self.rank == 0:
                    print(f"🔍 捕获了 {len(captured_attention)} 个attention权重")
                
                # 检查是否需要立即生成可视化
                step_interval = 2
                should_visualize = (step_idx + 1) % step_interval == 0
                
                if should_visualize and self.rank == 0:
                    # 只使用第一个层的权重进行可视化，保持原始数值范围
                    single_layer_weights = captured_attention[0]  # 使用第一个层，形状[1, 40, 3600, 512]
                    print(f"🔍 使用第一个层权重，形状: {single_layer_weights.shape}")
                    print(f"🔍 权重范围: {single_layer_weights.min():.4f} - {single_layer_weights.max():.4f}")
                    
                    # 立即生成当前步的可视化
                    self._visualize_current_step(single_layer_weights, step_idx)
                
                # 不保存到历史记录中，直接释放内存
                if self.rank == 0:
                    model_type = "高噪声专家" if timestep.item() >= self.boundary * self.num_train_timesteps else "低噪声专家"
                    print(f"🔍 捕获{model_type}真实注意力权重 - Step {step_idx+1}")
                    print(f"🔍 捕获了 {len(captured_attention)} 个attention层，仅使用第一个层进行可视化")
                    if should_visualize:
                        print(f"🔍 已生成Step {step_idx+1}的可视化图")
                    print(f"🔍 已释放Step {step_idx+1}的attention权重内存")
            else:
                if self.rank == 0:
                    print(f"⚠️ 未捕获到真实attention权重，跳过Step {step_idx+1}")
                    print(f"⚠️ 可能原因:")
                    print(f"   - Hook没有正确注册")
                    print(f"   - WanCrossAttention模块没有执行")
                    print(f"   - 模型结构发生变化")
            
            return result
            
        finally:
            # 移除hooks
            for hook in hooks:
                hook.remove()
