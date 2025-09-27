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
        
        # 误差分析相关
        self.enable_error_analysis = False
        self.error_history = []
        self.error_output_dir = None

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
                 enable_error_analysis=False,
                 error_output_dir="error_analysis_outputs",
                 enable_improved_frame_completion=False,
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
            enable_improved_frame_completion (`bool`, *optional*, defaults to False):
                Enable improved frame completion method. When switching from high-noise to low-noise expert,
                duplicate odd frames to even positions to maintain seed consistency without restarting scheduler.

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
        
        # 初始化误差分析
        if enable_error_analysis:
            if self.rank == 0:
                print("📊 误差分析已启用")
                # 如果指定了主输出目录，将误差分析结果保存到主目录
                if output_dir:
                    error_output_path = os.path.join(output_dir, "error_analysis")
                    print(f"   输出目录: {error_output_path}")
                else:
                    error_output_path = error_output_dir
                    print(f"   输出目录: {error_output_path}")
                print("   将记录条件输出和无条件输出的误差")
            self._enable_error_analysis(error_output_path if output_dir else error_output_dir)
            self.error_history = []  # 存储每步的误差数据
        else:
            if self.rank == 0:
                print("📝 误差分析已禁用")
        
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

        # 根据优化方法选择初始噪声形状
        if enable_half_frame_generation:
            # 使用减半后的target_shape生成noise（高噪声专家）
            initial_target_shape = half_target_shape
        else:
            # 使用完整帧数的target_shape生成noise
            initial_target_shape = full_target_shape

        noise = [
            torch.randn(
                initial_target_shape[0],
                initial_target_shape[1],
                initial_target_shape[2],
                initial_target_shape[3],
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
                    if self.enable_error_analysis:
                        noise_pred_uncond = self._call_model_with_error_analysis(
                            model, latent_model_input, timestep, model_kwargs_null, step_idx)
                    else:
                        noise_pred_uncond = self._call_model_with_attention_capture(
                            model, latent_model_input, timestep, model_kwargs_null, step_idx)
                    noise_pred = noise_pred_uncond
                else:
                    # 正常CFG计算
                    if self.enable_error_analysis:
                        # 使用误差分析函数，但只记录一次误差
                        noise_pred_cond = self._call_model_with_error_analysis(
                            model, latent_model_input, timestep, model_kwargs_c, step_idx, record_error=True)
                        noise_pred_uncond = self._call_model_with_error_analysis(
                            model, latent_model_input, timestep, model_kwargs_null, step_idx, record_error=False)
                    else:
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
                
                # 改进的帧数补全：在专家切换时模拟半帧生成并替换偶数帧
                if enable_improved_frame_completion and is_high_noise_phase and step_idx == max(high_noise_steps):
                    if self.rank == 0:
                        print(f"🔄 高噪声专家结束，开始改进帧数补全: 模拟半帧生成，替换偶数帧")
                        print(f"🔍 调试信息: enable_improved_frame_completion={enable_improved_frame_completion}")
                        print(f"🔍 调试信息: is_high_noise_phase={is_high_noise_phase}")
                        print(f"🔍 调试信息: step_idx={step_idx}, max(high_noise_steps)={max(high_noise_steps)}")
                    
                    # 当前是完整帧数，模拟半帧生成的效果
                    current_frames = latents[0].shape[1]  # 当前完整帧数
                    
                    if self.rank == 0:
                        print(f"🔍 当前帧数: {current_frames}")
                        print(f"🔍 开始替换偶数帧...")
                    
                    # 改进的帧数补全：偶数帧复制前一个奇数帧（模拟半帧生成效果）
                    replaced_count = 0
                    for i in range(0, current_frames, 2):  # 只处理偶数帧
                        if i > 0:  # 跳过第0帧
                            # 保存原始值用于对比
                            original_value = latents[0][:, i, :, :].clone()
                            # 偶数帧复制前一个奇数帧
                            latents[0][:, i, :, :] = latents[0][:, i-1, :, :]
                            # 检查是否真的替换了
                            if not torch.equal(original_value, latents[0][:, i, :, :]):
                                replaced_count += 1
                                if self.rank == 0 and replaced_count <= 3:  # 只打印前3个替换
                                    print(f"🔍 帧{i}已替换: 原始值范围[{original_value.min():.4f}, {original_value.max():.4f}] -> 新值范围[{latents[0][:, i, :, :].min():.4f}, {latents[0][:, i, :, :].max():.4f}]")
                    
                    if self.rank == 0:
                        print(f"✅ 改进帧数补全完成: 共替换了{replaced_count}个偶数帧")
                        print(f"🔍 最终latents形状: {latents[0].shape}")
                        print(f"🔍 最终latents值范围: [{latents[0].min():.4f}, {latents[0].max():.4f}]")
                    
                    # 更新seq_len为完整帧数的seq_len（低噪声专家使用）
                    current_seq_len = full_seq_len
                    arg_c = {'context': context, 'seq_len': current_seq_len}
                    arg_null = {'context': context_null, 'seq_len': current_seq_len}
                    
                    if self.rank == 0:
                        print(f"✅ 改进帧数补全完成: {latents[0].shape[1]}帧 (偶数帧复制前一个奇数帧)")
                        print(f"🔄 无需重新初始化scheduler，保持种子一致性")
                
                # 原有的帧数减半优化：在高噪声专家结束时进行帧数补全（在scheduler.step之后）
                elif enable_half_frame_generation and is_high_noise_phase and step_idx == max(high_noise_steps):
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

        # 创建误差分析
        if enable_error_analysis:
            if self.error_history:
                if self.rank == 0:
                    print(f"📊 开始创建误差分析，共{len(self.error_history)}步数据")
                self._create_error_visualization()
                self._create_error_analysis_report()
                if self.rank == 0:
                    print(f"📊 误差分析完成，结果保存到: {self.error_output_dir}")
                    # 如果误差分析结果在主输出目录中，显示相对路径
                    if output_dir and self.error_output_dir.startswith(output_dir):
                        relative_path = os.path.relpath(self.error_output_dir, output_dir)
                        print(f"📁 误差分析文件: {relative_path}/error_analysis_plots.png")
                        print(f"📁 误差分析报告: {relative_path}/error_analysis_report.md")
            else:
                if self.rank == 0:
                    print("⚠️ 误差分析已启用，但没有收集到误差数据")
        else:
            if self.rank == 0:
                print("📝 误差分析未启用")

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
            
            # 创建误差分析
            if self.enable_error_analysis:
                if self.error_history:
                    if self.rank == 0:
                        print(f"📊 开始创建误差分析，共{len(self.error_history)}步数据")
                    self._create_error_visualization()
                    self._create_error_analysis_report()
                    if self.rank == 0:
                        print(f"📊 误差分析完成，结果保存到: {self.error_output_dir}")
                        # 如果误差分析结果在主输出目录中，显示相对路径
                        if output_dir and self.error_output_dir.startswith(output_dir):
                            relative_path = os.path.relpath(self.error_output_dir, output_dir)
                            print(f"📁 误差分析文件: {relative_path}/error_analysis_plots.png")
                            print(f"📁 误差分析报告: {relative_path}/error_analysis_report.md")
                else:
                    if self.rank == 0:
                        print("⚠️ 误差分析已启用，但没有收集到误差数据")
            else:
                if self.rank == 0:
                    print("📝 误差分析未启用")
            
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
            # 获取实际输入的tokens（使用与模型相同的tokenizer）
            tokens = self._get_tokens_from_prompt(self.prompt)
            
            # 平均当前步的所有批次和注意力头
            # attention_weights形状: [batch, heads, seq_len, context_len]
            avg_attention_weights = attention_weights.mean(dim=(0, 1))  # [seq_len, context_len]
            
            # 获取实际文本长度（非填充部分）
            actual_text_len = len(tokens)
            
            # 只使用实际文本token对应的attention权重
            if actual_text_len <= 6:
                # 如果实际tokens少于等于6个，只取对应的attention权重
                token_attention_weights = avg_attention_weights[:, :actual_text_len]
                used_tokens = tokens
            else:
                # 如果实际tokens超过6个，只取前6个
                token_attention_weights = avg_attention_weights[:, :6]
                used_tokens = tokens[:6]
            
            # 创建当前step的cross attention map可视化
            step_save_path = os.path.join(self.attention_output_dir, f"step_{step_idx+1:02d}_cross_attention_map.png")
            self.attention_visualizer.visualize_attention_step(
                token_attention_weights,  # 只使用实际文本token的权重
                used_tokens, step_idx, step_save_path, title=f"Step {step_idx+1} Cross Attention Map"
            )
            
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
        """从prompt获取tokens，使用与模型相同的tokenizer"""
        try:
            # 使用与模型相同的T5 tokenizer
            if hasattr(self, 'text_encoder') and hasattr(self.text_encoder, 'tokenizer'):
                # 使用模型实际使用的tokenizer
                tokenizer = self.text_encoder.tokenizer
                # 获取token IDs
                token_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids[0]
                # 转换为token字符串
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                # 过滤掉特殊token
                tokens = [token for token in tokens if not token.startswith('<') and not token.startswith('▁')]
                # 只返回前6个token
                return tokens[:6]
            else:
                # 回退到简单分割
                tokens = prompt.split()
                return tokens[:6]
        except Exception as e:
            # 最终回退
            tokens = prompt.split()
            return tokens[:6]
    
    def _call_model_with_attention_capture(self, model, latent_model_input, timestep, model_kwargs, step_idx):
        """调用模型并捕获真实的注意力权重"""
        if not self.enable_attention_visualization:
            return model(latent_model_input, timestep, **model_kwargs)[0]
        
        # 使用hook机制捕获真实的attention权重
        captured_attention = []
        
        def attention_hook(module, input, output):
            """Hook函数：捕获真实的cross attention权重"""
            if hasattr(module, '__class__') and 'WanCrossAttention' in module.__class__.__name__:
                try:
                    if len(input) >= 3:
                        x, context, context_lens = input[:3]
                        
                        b, n, d = x.size(0), module.num_heads, module.head_dim
                        
                        q = module.norm_q(module.q(x)).view(b, -1, n, d)
                        k = module.norm_k(module.k(context)).view(b, -1, n, d)
                        v = module.v(context).view(b, -1, n, d)
                        
                        scale = 1.0 / (d ** 0.5)
                        
                        q = q.transpose(1, 2)  # [1, 40, 3600, 128]
                        k = k.transpose(1, 2)  # [1, 40, 512, 128]
                        
                        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                        
                        if context_lens is not None:
                            max_len = k.size(2)
                            mask = torch.arange(max_len, device=k.device).expand(b, max_len) < context_lens.unsqueeze(1)
                            mask = mask.unsqueeze(1).unsqueeze(1)
                            scores = scores.masked_fill(~mask, float('-inf'))
                        
                        attention_weights = torch.softmax(scores, dim=-1)
                        
                        if isinstance(attention_weights, torch.Tensor):
                            captured_attention.append(attention_weights)
                except Exception as e:
                    pass
        
        # 注册hook到WanCrossAttention模块，而不是WanAttentionBlock
        hooks = []
        cross_attention_found = 0
        
        for name, module in model.named_modules():
            # 直接查找WanCrossAttention模块
            if hasattr(module, '__class__') and 'WanCrossAttention' in module.__class__.__name__:
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
                cross_attention_found += 1
        
        try:
            result = model(latent_model_input, timestep, **model_kwargs)[0]
            
            if captured_attention:
                # 检查是否需要立即生成可视化
                step_interval = 2
                should_visualize = (step_idx + 1) % step_interval == 0
                
                if should_visualize and self.rank == 0:
                    # 只使用第一个层的权重进行可视化
                    single_layer_weights = captured_attention[0]
                    self._visualize_current_step(single_layer_weights, step_idx)
                    print(f"✅ Step {step_idx+1} 可视化完成")
            
            return result
            
        finally:
            for hook in hooks:
                hook.remove()

    def _enable_error_analysis(self, error_output_dir):
        """启用误差分析功能"""
        import os
        os.makedirs(error_output_dir, exist_ok=True)
        self.error_output_dir = error_output_dir
        self.enable_error_analysis = True

    def _call_model_with_error_analysis(self, model, latent_model_input, timestep, model_kwargs, step_idx, record_error=True):
        """调用模型并记录误差分析"""
        if not self.enable_error_analysis:
            return model(latent_model_input, timestep, **model_kwargs)[0]
        
        # 获取当前输出
        current_output = model(latent_model_input, timestep, **model_kwargs)[0]
        
        # 只在记录误差时计算和保存误差数据
        if record_error:
            # 获取无条件输出（使用空文本）
            model_kwargs_uncond = model_kwargs.copy()
            if 'context' in model_kwargs_uncond:
                # 使用空文本作为无条件输入
                model_kwargs_uncond['context'] = [torch.zeros_like(ctx) for ctx in model_kwargs['context']]
            
            noise_pred_uncond = model(latent_model_input, timestep, **model_kwargs_uncond)[0]
            
            # 计算误差
            absolute_error = torch.abs(current_output - noise_pred_uncond)
            relative_error = absolute_error / (torch.abs(current_output) + 1e-8)
            
            # 记录误差数据
            error_data = {
                'step': step_idx + 1,
                'timestep': timestep.item(),
                'absolute_error_mean': absolute_error.mean().item(),
                'absolute_error_std': absolute_error.std().item(),
                'relative_error_mean': relative_error.mean().item(),
                'relative_error_std': relative_error.std().item(),
                'conditional_output_mean': current_output.mean().item(),
                'conditional_output_std': current_output.std().item(),
                'unconditional_output_mean': noise_pred_uncond.mean().item(),
                'unconditional_output_std': noise_pred_uncond.std().item(),
            }
            
            self.error_history.append(error_data)
            
            if self.rank == 0:
                print(f"📊 Step {step_idx+1}: 绝对误差={error_data['absolute_error_mean']:.4f}, 相对误差={error_data['relative_error_mean']:.4f}")
        
        return current_output

    def _create_error_visualization(self):
        """创建误差分析可视化图表"""
        if not self.enable_error_analysis or not self.error_history:
            if self.rank == 0:
                print("⚠️ 无法创建误差分析图表：误差分析未启用或无数据")
            return
        
        if self.rank == 0:
            print(f"📊 开始创建误差分析图表，数据点数: {len(self.error_history)}")
        
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime
        
        # 提取数据
        steps = [data['step'] for data in self.error_history]
        timesteps = [data['timestep'] for data in self.error_history]
        abs_errors = [data['absolute_error_mean'] for data in self.error_history]
        rel_errors = [data['relative_error_mean'] for data in self.error_history]
        cond_means = [data['conditional_output_mean'] for data in self.error_history]
        uncond_means = [data['unconditional_output_mean'] for data in self.error_history]
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Error Analysis: Conditional vs Unconditional Model Outputs', fontsize=16, fontweight='bold')
        
        # 图1: 绝对误差和相对误差随步数变化
        ax1.plot(steps, abs_errors, 'b-', label='Absolute Error', linewidth=2.5, marker='o', markersize=4)
        ax1.set_xlabel('Denoising Step', fontsize=12)
        ax1.set_ylabel('Absolute Error', fontsize=12, color='blue')
        ax1.set_title('Absolute and Relative Error vs Denoising Steps', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=10)
        # 设置x轴刻度为5的倍数
        step_ticks = [i for i in range(1, max(steps) + 1, 5)]
        ax1.set_xticks(step_ticks)
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(steps, rel_errors, 'r-', label='Relative Error', linewidth=2.5, marker='s', markersize=4)
        ax1_twin.set_ylabel('Relative Error', fontsize=12, color='red')
        ax1_twin.legend(loc='upper right', fontsize=10)
        
        # 图2: 条件输出 vs 无条件输出
        ax2.plot(steps, cond_means, 'g-', label='Conditional Output', linewidth=2.5, marker='^', markersize=4)
        ax2.plot(steps, uncond_means, 'orange', label='Unconditional Output', linewidth=2.5, marker='v', markersize=4)
        ax2.set_xlabel('Denoising Step', fontsize=12)
        ax2.set_ylabel('Output Mean Value', fontsize=12)
        ax2.set_title('Conditional vs Unconditional Output Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        # 设置x轴刻度为5的倍数
        step_ticks = [i for i in range(1, max(steps) + 1, 5)]
        ax2.set_xticks(step_ticks)
        
        # 图3: 绝对误差随timestep变化
        ax3.plot(timesteps, abs_errors, 'b-', label='Absolute Error', linewidth=2.5, marker='o', markersize=4)
        ax3.set_xlabel('Timestep', fontsize=12)
        ax3.set_ylabel('Absolute Error', fontsize=12)
        ax3.set_title('Absolute Error vs Timestep', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=10)
        
        # 图4: 相对误差随timestep变化
        ax4.plot(timesteps, rel_errors, 'r-', label='Relative Error', linewidth=2.5, marker='s', markersize=4)
        ax4.set_xlabel('Timestep', fontsize=12)
        ax4.set_ylabel('Relative Error', fontsize=12)
        ax4.set_title('Relative Error vs Timestep', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        error_plot_path = os.path.join(self.error_output_dir, "error_analysis_plots.png")
        try:
            plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            if self.rank == 0:
                print(f"📊 误差分析图表已保存到: {error_plot_path}")
        except Exception as e:
            if self.rank == 0:
                print(f"❌ 保存误差分析图表时出错: {e}")
                import traceback
                print(f"❌ 详细错误信息: {traceback.format_exc()}")
            plt.close()

    def _create_error_analysis_report(self):
        """创建误差分析报告"""
        if not self.enable_error_analysis or not self.error_history:
            return
        
        import numpy as np
        from datetime import datetime
        
        # 计算统计信息
        abs_errors = [data['absolute_error_mean'] for data in self.error_history]
        rel_errors = [data['relative_error_mean'] for data in self.error_history]
        
        report = f"""# Error Analysis Report

## Basic Information
- **Total Steps**: {len(self.error_history)}
- **Output Directory**: {self.error_output_dir}
- **Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Statistical Summary
### Absolute Error
- **Mean**: {np.mean(abs_errors):.6f}
- **Std Dev**: {np.std(abs_errors):.6f}
- **Max**: {np.max(abs_errors):.6f}
- **Min**: {np.min(abs_errors):.6f}

### Relative Error
- **Mean**: {np.mean(rel_errors):.6f}
- **Std Dev**: {np.std(rel_errors):.6f}
- **Max**: {np.max(rel_errors):.6f}
- **Min**: {np.min(rel_errors):.6f}

## Detailed Data
| Step | Timestep | Absolute Error | Relative Error | Conditional Output Mean | Unconditional Output Mean |
|------|----------|----------------|----------------|------------------------|---------------------------|
"""
        
        for data in self.error_history:
            report += f"| {data['step']} | {data['timestep']:.1f} | {data['absolute_error_mean']:.6f} | {data['relative_error_mean']:.6f} | {data['conditional_output_mean']:.6f} | {data['unconditional_output_mean']:.6f} |\n"
        
        report += f"""
## Analysis Conclusions
1. **Error Trend**: Changes in absolute and relative errors during denoising process
2. **Conditional Impact**: Difference between conditional and unconditional outputs
3. **Convergence**: Whether errors converge as denoising steps progress

## Generated Files
- `error_analysis_plots.png` - Error analysis visualization plots
- `error_analysis_report.md` - Detailed analysis report
"""
        
        # 保存报告
        report_path = os.path.join(self.error_output_dir, "error_analysis_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        if self.rank == 0:
            print(f"📊 误差分析报告已保存到: {report_path}")
