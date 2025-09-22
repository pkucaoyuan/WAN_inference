"""
WAN2.2 T2V Model for text-to-video generation.
"""
import logging
import os
import time
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from diffusers import DPMSolverMultistepScheduler
from transformers import T5Tokenizer, T5EncoderModel
from .modules.dit import WanModel
from .modules.vae import WanVAE
from .utils.data import preprocess_text_prompt
from .utils.flow_solver import UniPCScheduler


class WanT2V(nn.Module):
    def __init__(self,
                 config,
                 t5_model: T5EncoderModel = None,
                 vae: WanVAE = None,
                 dit: WanModel = None,
                 t5_tokenizer: T5Tokenizer = None,
                 device="cuda",
                 rank=0,
                 dtype=torch.bfloat16):
        super().__init__()
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.config = config
        self.sample_steps = config['sample_steps']
        self.boundary = config['boundary']
        
        self.t5_model = t5_model
        self.t5_tokenizer = t5_tokenizer
        self.vae = vae
        self.dit = dit
        
        # Initialize timing
        self.total_switch_time = 0.0
        self.step_timings = []  # 记录每步推理时间
        
        # Initialize schedulers
        if config['sample_solver'] == 'unipc':
            self.sample_scheduler = UniPCScheduler(config['num_train_timesteps'])
        else:
            self.sample_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=config['num_train_timesteps'],
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
                solver_order=2,
                prediction_type="v_prediction",
                thresholding=False,
                dynamic_thresholding_ratio=0.95,
                sample_max_value=1.0,
                algorithm_type="dpmsolver++",
                solver_type="midpoint",
                lower_order_final=True,
                use_karras_sigmas=False,
            )

    def _prepare_model_for_timestep(self, t, boundary):
        """
        Prepare the model for the given timestep by loading the appropriate expert.
        """
        switch_start_time = time.time()
        
        if t >= boundary:
            # High noise phase - load high noise model
            expert_name = "high_noise_model"
        else:
            # Low noise phase - load low noise model  
            expert_name = "low_noise_model"
            
        # Load the expert model if not already loaded
        if not hasattr(self, '_current_expert') or self._current_expert != expert_name:
            # Move previous expert to CPU if exists
            if hasattr(self, '_current_expert') and hasattr(self.dit, self._current_expert):
                prev_expert = getattr(self.dit, self._current_expert)
                if next(prev_expert.parameters()).device != torch.device('cpu'):
                    prev_expert.to('cpu')
                    torch.cuda.empty_cache()
            
            # Load new expert to GPU
            expert_model = getattr(self.dit, expert_name)
            expert_model.to(self.device, dtype=self.dtype)
            self._current_expert = expert_name
            
            switch_time = time.time() - switch_start_time
            self.total_switch_time += switch_time
            
            if self.rank == 0:
                print(f"🔄 专家切换: {expert_name} (t={t}) ⏱️ 专家切换耗时: {switch_time:.3f}秒")
        
        return getattr(self.dit, expert_name)

    def generate(self,
                 input_prompt,
                 frame_num=81,
                 image_size=(720, 1280),
                 seed=-1,
                 offload_model=True,
                 cfg_truncate_steps=5,
                 cfg_truncate_high_noise_steps=3,
                 output_dir=None,
):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                The text prompt to generate video from.
            frame_num (`int`, defaults to 81):
                Number of frames to generate.
            image_size (`tuple`, defaults to (720, 1280)):
                Size of generated frames (height, width).
            seed (`int`, defaults to -1):
                Random seed for generation. If -1, uses random seed.
            offload_model (`bool`, defaults to True):
                Whether to offload models to CPU when not in use.
            cfg_truncate_steps (`int`, defaults to 5):
                Number of final steps to skip conditional forward pass.
            cfg_truncate_high_noise_steps (`int`, defaults to 3):
                Number of high-noise final steps to skip conditional forward pass.
            output_dir (`str`, optional):
                Directory to save outputs.

        Returns:
            `tuple`: Generated video tensor and timing information.
        """
        # Set random seed
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Process text prompt
        context, context_null, seq_len = preprocess_text_prompt(
            input_prompt, self.t5_tokenizer, self.t5_model, self.device, self.dtype
        )
        
        # Initialize noise
        height, width = image_size
        latent_height = height // 8
        latent_width = width // 8
        noise_shape = (16, frame_num, latent_height, latent_width)
        
        if seed != -1:
            seed_g = torch.Generator(device=self.device).manual_seed(seed)
        else:
            seed_g = None
            
        noise = torch.randn(noise_shape, device=self.device, dtype=self.dtype, generator=seed_g)
        
        # Setup scheduler
        self.sample_scheduler.set_timesteps(self.sample_steps, device=self.device)
        timesteps = self.sample_scheduler.timesteps
        boundary = timesteps[int(len(timesteps) * self.boundary)]
        
        # Initialize sample guide scale
        if isinstance(self.config['sample_guide_scale'], tuple):
            sample_guide_scale_start, sample_guide_scale_end = self.config['sample_guide_scale']
            sample_guide_scale = sample_guide_scale_start
            scale_step = (sample_guide_scale_end - sample_guide_scale_start) / len(timesteps)
        else:
            sample_guide_scale = self.config['sample_guide_scale']
            scale_step = 0

        # sample videos
        latents = noise

        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}


        import time
        for step_idx, t in enumerate(tqdm(timesteps)):
            step_start_time = time.time()  # 记录每步开始时间
            latent_model_input = latents
            timestep = [t]

            # Update sample guide scale
            if scale_step != 0:
                sample_guide_scale += scale_step

            # Prepare model for current timestep
            model = self._prepare_model_for_timestep(t.item(), boundary.item())

            # Check if we should use CFG truncation
            is_final_steps = step_idx >= (len(timesteps) - cfg_truncate_steps)
            
            # Check if we're in high noise expert's final steps
            is_high_noise_phase = t.item() >= boundary
            high_noise_steps = [i for i, ts in enumerate(timesteps) if ts.item() >= boundary]
            is_high_noise_final = (is_high_noise_phase and 
                                 step_idx >= (max(high_noise_steps) - cfg_truncate_high_noise_steps + 1))
            

            if is_final_steps or is_high_noise_final:
                # CFG截断：只进行无条件预测（真正节省50%计算）
                noise_pred = model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if self.rank == 0:
                    if is_high_noise_final:
                        print(f"高噪声专家CFG截断: Step {step_idx+1}/{len(timesteps)}, t={t.item():.0f}")
                    else:
                        print(f"低噪声专家CFG截断: Step {step_idx+1}/{len(timesteps)}, t={t.item():.0f}")
            else:
                # 标准CFG流程
                noise_pred_cond = model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = model(
                    latent_model_input, t=timestep, **arg_null)[0]

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
                'duration': step_duration,
                'timestep': t.item(),
                'expert': 'high_noise' if is_high_noise_phase else 'low_noise'
            }
            self.step_timings.append(step_timing)
        
        # 生成视频
        videos = self.vae.decode(latents)
        

        # 返回结果和时间信息
        result_videos = videos[0] if self.rank == 0 else None
        timing_info = {
            'total_switch_time': getattr(self, 'total_switch_time', 0.0),
            'step_timings': getattr(self, 'step_timings', [])
        }
        return result_videos, timing_info