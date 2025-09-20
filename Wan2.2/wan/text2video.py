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
                 enable_token_pruning=False,
                 pruning_threshold=20,
                 pruning_baseline_steps=5,
                 pruning_start_layer=6,
                 pruning_end_layer=35):
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

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            # 初始化token裁剪器（如果启用）
            token_pruner = None
            if enable_token_pruning and output_dir is not None:
                from .modules.adaptive_token_pruning import AdaptiveTokenPruning
                
                # 计算高噪声专家的实际结束步数
                high_noise_steps = [i for i, ts in enumerate(timesteps) if ts.item() >= boundary]
                actual_high_noise_end = max(high_noise_steps) if high_noise_steps else len(timesteps) - 1
                
                # 如果用户指定的end_layer超出高噪声专家范围，自动调整
                effective_end_layer = min(pruning_end_layer, actual_high_noise_end)
                
                token_pruner = AdaptiveTokenPruning(
                    baseline_steps=pruning_baseline_steps,
                    percentile_threshold=pruning_threshold,
                    start_layer=pruning_start_layer,
                    end_layer=effective_end_layer,
                    expert_name="high_noise"
                )
                if self.rank == 0:
                    print(f"🧠 Token裁剪器已启用")
                    print(f"   📊 百分位阈值: {pruning_threshold}% (越高越激进)")
                    print(f"   🔢 基准步数: {pruning_baseline_steps}")
                    print(f"   🎯 高噪声专家结束步数: {actual_high_noise_end + 1}")
                    print(f"   📍 裁剪范围: Layer {pruning_start_layer}-{effective_end_layer}")
                    print(f"   ✅ 真正的Token裁剪：在Transformer层中减少计算量")
                    print(f"   ✅ CFG截断：跳过条件前向传播节省50%计算")
                    if effective_end_layer != pruning_end_layer:
                        print(f"   ⚠️ 结束层已自动调整: {pruning_end_layer} → {effective_end_layer} (高噪声专家边界)")

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
                
                # 计算当前步骤的active_mask（真正的token裁剪）
                current_active_mask = None
                
                # 专家切换检测：从高噪声切换到低噪声时清除token裁剪状态
                if token_pruner is not None:
                    # 检查是否从高噪声专家切换到低噪声专家
                    prev_is_high_noise = getattr(self, '_prev_is_high_noise_phase', True)
                    if prev_is_high_noise and not is_high_noise_phase:
                        # 专家切换：清除所有token裁剪预测状态
                        if hasattr(self, '_next_step_frozen_indices'):
                            delattr(self, '_next_step_frozen_indices')
                        if hasattr(self, '_next_step_active_indices'):
                            delattr(self, '_next_step_active_indices')
                        if hasattr(self, '_prev_latents'):
                            delattr(self, '_prev_latents')
                        
                        if self.rank == 0:
                            print(f"🔄 专家切换: 高噪声→低噪声，清除Token裁剪状态")
                            print(f"   🔓 低噪声专家: 100%token激活，完整推理")
                    
                    # 记录当前专家状态
                    self._prev_is_high_noise_phase = is_high_noise_phase
                
                if token_pruner is not None and is_high_noise_phase:
                    expert_name = "high_noise"
                    
                    # 首先检查是否有上一步的预测结果
                    if hasattr(self, '_next_step_frozen_indices') and hasattr(self, '_next_step_active_indices'):
                        # 使用上一步预测的结果进行当前步的token裁剪
                        frozen_indices = self._next_step_frozen_indices
                        active_indices = self._next_step_active_indices
                        
                        # 创建active_mask
                        model_seq_len = seq_len
                        current_active_mask = torch.ones(model_seq_len, dtype=torch.bool, device=latents[0].device)
                        
                        if len(frozen_indices) > 0:
                            # 设置冻结token为False
                            image_token_end = min(len(frozen_indices) + len(active_indices), model_seq_len)
                            if len(frozen_indices) > 0:
                                valid_frozen_indices = frozen_indices[frozen_indices < image_token_end]
                                if len(valid_frozen_indices) > 0:
                                    current_active_mask[valid_frozen_indices] = False
                            
                            if self.rank == 0:
                                active_count = len(active_indices)
                                frozen_count = len(frozen_indices)
                                total_image_tokens = active_count + frozen_count
                                
                                print(f"🔥 Step {step_idx+1} 使用预测的Token裁剪:")
                                print(f"   📊 激活Token: {active_count}/{total_image_tokens} ({100*active_count/total_image_tokens:.1f}%)")
                                print(f"   🧊 冻结Token: {frozen_count} 个 (基于上一步预测)")
                                print(f"   💾 实际节省计算: {100*frozen_count/total_image_tokens:.1f}%")
                                print(f"   🎯 使用上一步的变化分数预测")
                                
                                # 计算实际的节省
                                ffn_savings = 1 - (active_count / total_image_tokens)
                                update_savings = 1 - (active_count / total_image_tokens)
                                
                                print(f"   ⚡ FFN计算节省: {100*ffn_savings:.1f}%")
                                print(f"   ⚡ Hidden State更新节省: {100*update_savings:.1f}%") 
                                print(f"   🔄 QKV缓存: 冻结token复用上一步QKV投影")
                                print(f"   📝 Self-Attention: 混合计算（新QKV + 缓存QKV）")
                                print(f"   📝 Cross-Attention: 完整计算（所有token参与）")
                                print(f"   🧊 冻结Token: 跳过FFN+QKV投影，保持hidden state不变")
                        
                        # 更新token_pruner的累积冻结状态
                        for idx in frozen_indices.cpu().tolist():
                            token_pruner.frozen_tokens.add(idx)
                        
                        # 清除预测结果，避免重复使用
                        delattr(self, '_next_step_frozen_indices')
                        delattr(self, '_next_step_active_indices')
                    
                    # 前1-4步：只保存latents，不收集统计
                    elif step_idx < token_pruner.baseline_steps - 1:
                        # 保存当前latents用于后续比较
                        self._prev_latents = latents[0].clone()
                        if self.rank == 0:
                            print(f"📝 Step {step_idx+1} 基准期：保存latents状态")
                    
                    # 第5步：收集所有token信息
                    elif step_idx == token_pruner.baseline_steps - 1:
                        # 第5步：收集所有token的变化信息用于确定动态阈值
                        if step_idx > 0:  # 需要前一步的latents来计算变化
                            prev_latents = getattr(self, '_prev_latents', None)
                            if prev_latents is not None:
                                # 计算token变化（修复维度问题）
                                # latents[0]形状: [C, F, H, W] = [16, 1, 90, 160]
                                # 沿着通道维度计算变化
                                change_magnitude = torch.norm(latents[0] - prev_latents, dim=0)  # 结果: [F, H, W]
                                prev_magnitude = torch.norm(prev_latents, dim=0)  # 结果: [F, H, W]
                                # 使用更大的epsilon和clamp确保数值稳定性
                                relative_change = change_magnitude / torch.clamp(prev_magnitude, min=1e-6)
                                
                                if self.rank == 0:
                                    print(f"   🔍 变化计算调试:")
                                    print(f"      📐 change_magnitude形状: {change_magnitude.shape}")
                                    print(f"      📐 relative_change形状: {relative_change.shape}")
                                
                                # 更新变化分数统计（基于真实token数量）
                                C, F, H, W = latents[0].shape
                                patch_size = (1, 2, 2)
                                actual_token_count = F * (H // patch_size[1]) * (W // patch_size[2])
                                
                                if self.rank == 0:
                                    print(f"🔍 第5步Latent调试:")
                                    print(f"   📐 Latent形状: {latents[0].shape} -> C={C}, F={F}, H={H}, W={W}")
                                    print(f"   📐 VAE stride: (4, 8, 8), patch_size: {patch_size}")
                                    print(f"   📐 原始输入尺寸推测: Frame={F*patch_size[0]}, H={H*8//4}, W={W*8//4}")
                                    print(f"   🧮 Token数量计算: {F} * ({H}//{patch_size[1]}) * ({W}//{patch_size[2]}) = {actual_token_count}")
                                    print(f"   📊 相对变化形状: {relative_change.shape}")
                                    print(f"   📏 变化计算维度: {len(relative_change.shape)}D")
                                
                                # 第5步高效收集所有token的变化信息（向量化操作）
                                # relative_change形状: [F, H, W] = [1, 90, 160]
                                # 使用unfold进行高效的patch提取
                                patches = relative_change.unfold(1, patch_size[1], patch_size[1])  # [F, H//2, W, patch_h]
                                patches = patches.unfold(2, patch_size[2], patch_size[2])          # [F, H//2, W//2, patch_h, patch_w]
                                
                                # 计算每个patch的平均值：[F, H//2, W//2]
                                token_changes_tensor = patches.mean(dim=(-2, -1))  # 对patch_h和patch_w求平均
                                
                                # 展平为1D tensor：[F * H//2 * W//2] = [3600]
                                token_changes_tensor = token_changes_tensor.view(-1)
                                
                                # 转换为Python列表用于统计（只转换一次）
                                all_token_changes = token_changes_tensor.cpu().tolist()
                                
                                # 批量更新统计信息
                                valid_changes = [v for v in all_token_changes if not (math.isnan(v) or math.isinf(v))]
                                for change_val in valid_changes:
                                    token_pruner.update_change_score_statistics(change_val)
                                
                                if self.rank == 0:
                                    print(f"📊 Step {step_idx+1} 收集所有token信息: {len(all_token_changes)} 个token变化值")
                                    print(f"✅ Token数量验证: 预期={actual_token_count}, 实际收集={len(all_token_changes)}")
                                    if len(all_token_changes) != actual_token_count:
                                        print(f"⚠️ Token数量不匹配！需要检查收集逻辑")
                                
                                # 基于第5步的所有token变化计算动态阈值
                                if len(all_token_changes) > 0:
                                    import numpy as np
                                    # 过滤掉无效值并统计
                                    valid_changes = [v for v in all_token_changes if not (np.isnan(v) or np.isinf(v))]
                                    nan_count = sum(1 for v in all_token_changes if np.isnan(v))
                                    inf_count = sum(1 for v in all_token_changes if np.isinf(v))
                                    
                                    if self.rank == 0:
                                        print(f"🔍 Token变化值统计:")
                                        print(f"   📊 总数: {len(all_token_changes)}")
                                        print(f"   ✅ 有效值: {len(valid_changes)}")
                                        print(f"   ❌ NaN值: {nan_count}")
                                        print(f"   ❌ Inf值: {inf_count}")
                                    
                                    if len(valid_changes) > 0:
                                        token_pruner.baseline_scores = valid_changes
                                        token_pruner.dynamic_threshold = token_pruner.calculate_dynamic_threshold()
                                        
                                        if self.rank == 0:
                                            print(f"🎯 动态阈值已确定: {token_pruner.dynamic_threshold:.4f} (第{token_pruner.percentile_threshold}百分位数)")
                                            print(f"   📊 基于{len(valid_changes)}个有效token变化值计算")
                                            print(f"   📈 变化范围: {min(valid_changes):.4f} - {max(valid_changes):.4f}")
                                        
                                        # ✅ 第5步立即预测第6步的冻结token
                                        threshold_tensor = torch.tensor(token_pruner.dynamic_threshold, 
                                                                      device=token_changes_tensor.device, 
                                                                      dtype=token_changes_tensor.dtype)
                                        
                                        # 基于第5步的变化分数预测第6步的冻结token
                                        frozen_mask = token_changes_tensor < threshold_tensor
                                        active_mask = ~frozen_mask
                                        
                                        next_step_frozen_indices = torch.where(frozen_mask)[0]
                                        next_step_active_indices = torch.where(active_mask)[0]
                                        
                                        # 确保第6步至少有一些token保持激活
                                        if len(next_step_active_indices) == 0:
                                            _, sorted_indices = torch.sort(token_changes_tensor, descending=True)
                                            min_active = max(len(token_changes_tensor) // 10, 1)
                                            next_step_active_indices = sorted_indices[:min_active]
                                            next_step_frozen_indices = sorted_indices[min_active:]
                                        
                                        # 存储预测结果供第6步使用
                                        self._next_step_frozen_indices = next_step_frozen_indices
                                        self._next_step_active_indices = next_step_active_indices
                                        
                                        if self.rank == 0:
                                            next_frozen_count = len(next_step_frozen_indices)
                                            next_active_count = len(next_step_active_indices)
                                            total_tokens = len(token_changes_tensor)
                                            
                                            print(f"🔮 第5步预测第6步Token裁剪:")
                                            print(f"   📊 第6步激活Token: {next_active_count}/{total_tokens} ({100*next_active_count/total_tokens:.1f}%)")
                                            print(f"   🧊 第6步冻结Token: {next_frozen_count} 个")
                                            print(f"   💾 预期节省计算: {100*next_frozen_count/total_tokens:.1f}%")
                                            print(f"   🎯 基于第5步变化分数预测")
                                            
                                    else:
                                        if self.rank == 0:
                                            print(f"⚠️ 第5步未收集到有效的变化值，使用默认阈值")
                                        token_pruner.dynamic_threshold = 0.01  # 默认阈值
                        
                        # 保存当前latents用于下一步比较
                        self._prev_latents = latents[0].clone()
                        
                    
                    # 应用token裁剪（基于真实latent变化）- 从第6步开始
                    elif token_pruner.should_apply_pruning(step_idx, expert_name):
                        prev_latents = getattr(self, '_prev_latents', None)
                        if prev_latents is not None and token_pruner.dynamic_threshold is not None:
                            # 计算真实的token变化幅度（修复维度问题）
                            # latents[0]形状: [C, F, H, W] = [16, 1, 90, 160]
                            # 沿着通道维度计算变化
                            change_magnitude = torch.norm(latents[0] - prev_latents, dim=0)  # 结果: [F, H, W]
                            prev_magnitude = torch.norm(prev_latents, dim=0)  # 结果: [F, H, W]
                            # 使用更大的epsilon和clamp确保数值稳定性
                            relative_change = change_magnitude / torch.clamp(prev_magnitude, min=1e-6)
                            
                            # 获取实际的token序列长度
                            # latents[0]形状: [C, F, H, W] 
                            # patch_size = (1, 2, 2) -> token数量 = F * (H//2) * (W//2)
                            C, F, H, W = latents[0].shape
                            patch_size = (1, 2, 2)  # 从模型配置获取
                            actual_token_count = F * (H // patch_size[1]) * (W // patch_size[2])
                            
                            if self.rank == 0:
                                print(f"🔍 第{step_idx+1}步Latent调试:")
                                print(f"   📐 Latent形状: {latents[0].shape} -> C={C}, F={F}, H={H}, W={W}")
                                print(f"   🧮 Token数量计算: {F} * ({H}//{patch_size[1]}) * ({W}//{patch_size[2]}) = {actual_token_count}")
                                print(f"   📊 相对变化形状: {relative_change.shape}, 维度: {len(relative_change.shape)}D")
                            
                            # 高效计算每个token位置的变化（向量化操作）
                            # relative_change形状: [F, H, W] = [1, 90, 160]
                            # 使用unfold进行高效的patch提取，避免嵌套循环
                            
                            # 对H和W维度进行patch分组
                            # unfold(dimension, size, step) 
                            patches = relative_change.unfold(1, patch_size[1], patch_size[1])  # [F, H//2, W, patch_h]
                            patches = patches.unfold(2, patch_size[2], patch_size[2])          # [F, H//2, W//2, patch_h, patch_w]
                            
                            # 计算每个patch的平均值：[F, H//2, W//2]
                            token_changes = patches.mean(dim=(-2, -1))  # 对patch_h和patch_w求平均
                            
                            # 展平为1D tensor：[F * H//2 * W//2] = [3600]
                            token_changes = token_changes.view(-1)
                            
                            if self.rank == 0:
                                print(f"⚡ 高效Token变化计算: {token_changes.shape} (向量化操作，避免3600次循环)")
                            
                            if self.rank == 0:
                                print(f"✅ Step {step_idx+1} Token数量验证: 预期={actual_token_count}, 实际处理={len(token_changes)}")
                            
                            # 高效的token选择（GPU tensor操作，避免Python循环）
                            threshold_tensor = torch.tensor(token_pruner.dynamic_threshold, 
                                                          device=token_changes.device, dtype=token_changes.dtype)
                            
                            # 累积式冻结逻辑：已冻结的token保持冻结，新的低变化token加入冻结
                            # 获取当前已冻结的token集合
                            current_frozen_set = set()
                            if hasattr(self, '_next_step_frozen_indices'):
                                current_frozen_set = set(self._next_step_frozen_indices.cpu().tolist())
                            
                            # 基于变化分数找出新的候选冻结token
                            new_frozen_mask = token_changes < threshold_tensor  # [3600] boolean tensor
                            new_frozen_candidates = torch.where(new_frozen_mask)[0]
                            
                            # 合并：已冻结 + 新冻结候选
                            all_frozen_indices = list(current_frozen_set)
                            for idx in new_frozen_candidates.cpu().tolist():
                                if idx not in current_frozen_set:
                                    all_frozen_indices.append(idx)
                            
                            # 生成最终的冻结和激活索引
                            next_step_frozen_indices = torch.tensor(all_frozen_indices, device=token_changes.device)
                            all_indices = torch.arange(len(token_changes), device=token_changes.device)
                            active_mask = torch.ones(len(token_changes), dtype=torch.bool, device=token_changes.device)
                            if len(all_frozen_indices) > 0:
                                active_mask[all_frozen_indices] = False
                            next_step_active_indices = torch.where(active_mask)[0]
                            
                            # 确保下一步至少有一些token保持激活
                            if len(next_step_active_indices) == 0:
                                # 如果所有token都低于阈值，保留变化最大的前10%
                                _, sorted_indices = torch.sort(token_changes, descending=True)  # GPU排序
                                min_active = max(len(token_changes) // 10, 1)
                                next_step_active_indices = sorted_indices[:min_active]
                                next_step_frozen_indices = sorted_indices[min_active:]
                            
                            # 存储预测结果供下一步使用（已经是GPU tensor）
                            self._next_step_frozen_indices = next_step_frozen_indices
                            self._next_step_active_indices = next_step_active_indices
                            
                            # 更新token_pruner的累积冻结状态
                            for idx in all_frozen_indices:
                                token_pruner.frozen_tokens.add(idx)
                            
                            if self.rank == 0:
                                next_frozen_count = len(next_step_frozen_indices)
                                next_active_count = len(next_step_active_indices)
                                total_image_tokens = len(token_changes)
                                
                                print(f"🔮 Step {step_idx+1} 预测下一步Token裁剪:")
                                print(f"   📊 下一步激活Token: {next_active_count}/{total_image_tokens} ({100*next_active_count/total_image_tokens:.1f}%)")
                                print(f"   🧊 下一步冻结Token: {next_frozen_count} 个 (变化 < {token_pruner.dynamic_threshold:.4f})")
                                print(f"   💾 预期节省计算: {100*next_frozen_count/total_image_tokens:.1f}%")
                                print(f"   🎯 基于当前步变化分数预测")
                                print(f"   📈 下一步将缓存冻结token的hidden state")
                                print(f"   ⚡ GPU tensor操作: 避免3600次.item()调用")
                        
                        # 保存当前latents
                        self._prev_latents = latents[0].clone()

                # 准备模型调用参数（包含active_mask）
                model_kwargs_c = {**arg_c, 'active_mask': current_active_mask}
                model_kwargs_null = {**arg_null, 'active_mask': current_active_mask}
                
                # 验证active_mask确实被使用（调试信息）
                if current_active_mask is not None and self.rank == 0:
                    active_ratio = current_active_mask.sum().item() / current_active_mask.size(0)
                    print(f"   🔍 Active_mask验证: {current_active_mask.sum().item()}/{current_active_mask.size(0)} "
                          f"({100*active_ratio:.1f}%) 将传递给模型")

                if is_final_steps or is_high_noise_final:
                    # CFG截断：只进行无条件预测（真正节省50%计算）
                    noise_pred = model(
                        latent_model_input, t=timestep, **model_kwargs_null)[0]
                    if self.rank == 0:
                        if is_high_noise_final:
                            print(f"高噪声专家CFG截断: Step {step_idx+1}/{len(timesteps)}, t={t.item():.0f}")
                        else:
                            print(f"低噪声专家CFG截断: Step {step_idx+1}/{len(timesteps)}, t={t.item():.0f}")
                else:
                    # 标准CFG流程（可能包含token裁剪优化）
                    noise_pred_cond = model(
                        latent_model_input, t=timestep, **model_kwargs_c)[0]
                    noise_pred_uncond = model(
                        latent_model_input, t=timestep, **model_kwargs_null)[0]

                    noise_pred = noise_pred_uncond + sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

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
                
                # 如果启用token裁剪，也记录到token_pruner中
                if token_pruner is not None:
                    token_pruner.step_timings.append(step_timing)

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

        # 生成token裁剪日志（如果启用）
        if token_pruner is not None and self.rank == 0 and output_dir is not None:
            try:
                # 生成详细的汇总报告
                report_path = token_pruner.generate_pruning_summary_report(output_dir)
                print(f"📄 Token裁剪报告已保存: {report_path}")
                
                # 保存最终的裁剪统计
                final_log_path = token_pruner.save_pruning_log(output_dir)
                print(f"📊 Token裁剪日志已保存: {final_log_path}")
            except Exception as e:
                print(f"⚠️ Token裁剪日志保存失败: {e}")

        # 返回结果和时间信息
        result_videos = videos[0] if self.rank == 0 else None
        timing_info = {
            'total_switch_time': getattr(self, 'total_switch_time', 0.0),
            'step_timings': getattr(self, 'step_timings', [])
        }
        return result_videos, timing_info
