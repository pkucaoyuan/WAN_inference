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

            for step_idx, t in enumerate(tqdm(timesteps)):
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
                if token_pruner is not None and is_high_noise_phase:
                    expert_name = "high_noise"
                    
                    # 在基准步骤收集统计信息
                    if step_idx + 1 <= token_pruner.baseline_steps:
                        # 基准期：完全推理，收集token变化统计
                        if step_idx > 0:  # 需要前一步的latents来计算变化
                            prev_latents = getattr(self, '_prev_latents', None)
                            if prev_latents is not None:
                                # 计算token变化
                                change_magnitude = torch.norm(latents[0] - prev_latents, dim=-1)
                                relative_change = change_magnitude / (torch.norm(prev_latents, dim=-1) + 1e-8)
                                
                                # 更新变化分数统计（基于真实token数量）
                                C, F, H, W = latents[0].shape
                                patch_size = (1, 2, 2)
                                actual_token_count = F * (H // patch_size[1]) * (W // patch_size[2])
                                
                                # 计算真实的token级别变化
                                for f in range(F):
                                    for h in range(0, H, patch_size[1]):
                                        for w in range(0, W, patch_size[2]):
                                            h_end = min(h + patch_size[1], H)
                                            w_end = min(w + patch_size[2], W)
                                            # 计算这个patch的真实变化
                                            patch_change = relative_change[:, h:h_end, w:w_end].mean()
                                            if not torch.isnan(patch_change) and not torch.isinf(patch_change):
                                                token_pruner.update_change_score_statistics(patch_change.item())
                                
                                if self.rank == 0:
                                    print(f"📊 Step {step_idx+1} 收集变化统计: {token_pruner.change_score_stats['count']} 个token变化值")
                        
                        # 保存当前latents用于下一步比较
                        self._prev_latents = latents[0].clone()
                        
                        # 基准期结束时计算动态阈值
                        if step_idx + 1 == token_pruner.baseline_steps:
                            # 基于真实变化统计计算动态阈值
                            stats = token_pruner.change_score_stats
                            if stats['count'] > 0 and len(stats['values']) > 0:
                                # 过滤掉无效值
                                import numpy as np
                                valid_values = [v for v in stats['values'] if not (np.isnan(v) or np.isinf(v))]
                                
                                if len(valid_values) > 0:
                                    token_pruner.baseline_scores = valid_values
                                    token_pruner.dynamic_threshold = token_pruner.calculate_dynamic_threshold()
                                    
                                    if self.rank == 0:
                                        print(f"🎯 动态阈值已确定: {token_pruner.dynamic_threshold:.4f} (第{token_pruner.percentile_threshold}百分位数)")
                                        print(f"   📊 基于{len(valid_values)}个有效token变化值计算")
                                        print(f"   📈 变化范围: {min(valid_values):.4f} - {max(valid_values):.4f}")
                                else:
                                    if self.rank == 0:
                                        print(f"⚠️ 基准期未收集到有效的变化值，使用默认阈值")
                                    token_pruner.dynamic_threshold = 0.01  # 默认阈值
                    
                    # 应用token裁剪（基于真实latent变化）
                    elif token_pruner.should_apply_pruning(step_idx + 1, expert_name):
                        prev_latents = getattr(self, '_prev_latents', None)
                        if prev_latents is not None and token_pruner.dynamic_threshold is not None:
                            # 计算真实的token变化幅度
                            change_magnitude = torch.norm(latents[0] - prev_latents, dim=-1)
                            relative_change = change_magnitude / (torch.norm(prev_latents, dim=-1) + 1e-8)
                            
                            # 获取实际的token序列长度
                            # latents[0]形状: [C, F, H, W] 
                            # patch_size = (1, 2, 2) -> token数量 = F * (H//2) * (W//2)
                            C, F, H, W = latents[0].shape
                            patch_size = (1, 2, 2)  # 从模型配置获取
                            actual_token_count = F * (H // patch_size[1]) * (W // patch_size[2])
                            
                            # 计算每个token位置的变化（基于空间位置）
                            # 将latent变化映射到token级别
                            if len(relative_change.shape) == 3:  # [C, H, W]
                                # 按patch_size分组计算平均变化
                                token_changes = []
                                for f in range(F):
                                    for h in range(0, H, patch_size[1]):
                                        for w in range(0, W, patch_size[2]):
                                            h_end = min(h + patch_size[1], H)
                                            w_end = min(w + patch_size[2], W)
                                            # 计算这个patch的平均变化
                                            patch_change = relative_change[:, h:h_end, w:w_end].mean()
                                            token_changes.append(patch_change)
                                token_changes = torch.stack(token_changes)
                            else:
                                # 如果维度不匹配，使用flatten后的前N个值
                                token_changes = relative_change.flatten()[:actual_token_count]
                            
                            # 基于真实变化幅度创建active_mask
                            active_token_indices = []
                            for i, change_val in enumerate(token_changes):
                                # 使用真实的变化值与动态阈值比较
                                if change_val.item() >= token_pruner.dynamic_threshold:
                                    active_token_indices.append(i)
                            
                            # 确保至少保留30%的token
                            min_active_tokens = max(len(token_changes) // 3, 1)
                            if len(active_token_indices) < min_active_tokens:
                                # 按变化幅度排序，保留top-k
                                sorted_indices = sorted(range(len(token_changes)), 
                                                       key=lambda i: token_changes[i].item(), reverse=True)
                                active_token_indices = sorted_indices[:min_active_tokens]
                            
                            # 创建完整的active_mask（用于模型计算）
                            # 使用模型的实际seq_len参数
                            model_seq_len = seq_len  # 模型forward中的seq_len参数
                            current_active_mask = torch.ones(model_seq_len, dtype=torch.bool, device=latents[0].device)
                            
                            # 设置非激活token（只针对实际的图像token范围）
                            image_token_end = min(len(token_changes), model_seq_len)
                            inactive_indices = [i for i in range(image_token_end) if i not in active_token_indices]
                            
                            if inactive_indices:
                                current_active_mask[inactive_indices] = False
                                
                                # 更新token_pruner的冻结列表（用于日志）
                                for idx in inactive_indices:
                                    token_pruner.frozen_tokens.add(idx)
                                
                                if self.rank == 0:
                                    active_count = len(active_token_indices)
                                    total_image_tokens = image_token_end
                                    frozen_count = len(inactive_indices)
                                    
                                    print(f"🔥 Step {step_idx+1} 真实Token裁剪:")
                                    print(f"   📊 激活Token: {active_count}/{total_image_tokens} ({100*active_count/total_image_tokens:.1f}%)")
                                    print(f"   🧊 冻结Token: {frozen_count} 个")
                                    print(f"   💾 实际节省计算: {100*frozen_count/total_image_tokens:.1f}%")
                                    print(f"   🎯 动态阈值: {token_pruner.dynamic_threshold:.4f}")
                                    
                                    # 计算实际的节省（CAT算法 + QKV缓存优化）
                                    ffn_savings = 1 - (active_count / total_image_tokens)             # FFN: O(N) -> O(k)
                                    update_savings = 1 - (active_count / total_image_tokens)          # Hidden state更新节省
                                    qkv_computation_savings = frozen_count / total_image_tokens       # QKV计算节省
                                    
                                    print(f"   ⚡ FFN计算节省: {100*ffn_savings:.1f}%")
                                    print(f"   ⚡ Hidden State更新节省: {100*update_savings:.1f}%") 
                                    print(f"   🔄 QKV计算节省: {100*qkv_computation_savings:.1f}%的token复用上一步QKV")
                                    print(f"   📝 Attention矩阵: 混合计算（新Q,K,V + 缓存Q,K,V）")
                                    print(f"   🧊 冻结Token: 复用hidden state + 复用QKV，跳过投影计算")
                        
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

        return videos[0] if self.rank == 0 else None, getattr(self, 'total_switch_time', 0.0)
