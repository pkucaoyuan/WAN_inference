# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool

EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "s2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
        "audio":
            "examples/talk.wav",
        "tts_prompt_audio":
            "examples/zero_shot_prompt.wav",
        "tts_prompt_text":
            "希望你以后能够做的比我还好呦。",
        "tts_text":
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"
    
    # Frame number validation removed - allow any frame number including 1

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]
    if args.audio is None and args.enable_tts is False and "audio" in EXAMPLE_PROMPT[args.task]:
        args.audio = EXAMPLE_PROMPT[args.task]["audio"]
    if (args.tts_prompt_audio is None or args.tts_text is None) and args.enable_tts is True and "audio" in EXAMPLE_PROMPT[args.task]:
        args.tts_prompt_audio = EXAMPLE_PROMPT[args.task]["tts_prompt_audio"]
        args.tts_prompt_text = EXAMPLE_PROMPT[args.task]["tts_prompt_text"]
        args.tts_text = EXAMPLE_PROMPT[args.task]["tts_text"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--cfg_truncate_steps",
        type=int,
        default=0,
        help="Number of final steps to skip conditional forward pass (CFG truncate). Set to 0 to disable.")
    parser.add_argument(
        "--cfg_truncate_high_noise_steps",
        type=int,
        default=0,
        help="Number of final steps in high-noise phase to skip conditional forward pass.")
    parser.add_argument(
        "--fast_loading",
        action="store_true",
        default=False,
        help="Enable fast loading optimizations: disable model offloading and keep models on GPU.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--enable_half_frame_generation",
        action="store_true",
        default=False,
        help="Enable half-frame generation: first expert generates half frames, then duplicate to full frames.")
    
    # Attention visualization parameters
    parser.add_argument(
        "--enable_attention_visualization",
        action="store_true",
        default=False,
        help="Enable attention visualization: generate average cross attention map.")
    parser.add_argument(
        "--attention_output_dir",
        type=str,
        default="attention_outputs",
        help="Directory to save attention visualization outputs.")

    # following args only works for s2v
    parser.add_argument(
        "--num_clip",
        type=int,
        default=None,
        help="Number of video clips to generate, the whole video will not exceed the length of audio."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to the audio file, e.g. wav, mp3")
    parser.add_argument(
        "--enable_tts",
        action="store_true",
        default=False,
        help="Use CosyVoice to synthesis audio")
    parser.add_argument(
        "--tts_prompt_audio",
        type=str,
        default=None,
        help="Path to the tts prompt audio file, e.g. wav, mp3. Must be greater than 16khz, and between 5s to 15s.")
    parser.add_argument(
        "--tts_prompt_text",
        type=str,
        default=None,
        help="Content to the tts prompt audio. If provided, must exactly match tts_prompt_audio")
    parser.add_argument(
        "--tts_text",
        type=str,
        default=None,
        help="Text wish to synthesize")
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="Provide Dw-pose sequence to do Pose Driven")
    parser.add_argument(
        "--start_from_ref",
        action="store_true",
        default=False,
        help="whether set the reference image as the starting point for generation"
    )
    parser.add_argument(
        "--infer_frames",
        type=int,
        default=80,
        help="Number of frames per clip, 48 or 80 or others (must be multiple of 4) for 14B s2v"
    )

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    if "t2v" in args.task:
        # GPU内存和错误处理优化
        try:
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
            # 启用内存池以减少碎片
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            if rank == 0:
                print(f"🔧 GPU内存优化: 设备{device}, 无内存分配限制")
        except Exception as e:
            if rank == 0:
                print(f"⚠️ GPU设置警告: {e}")
        
        # 快速加载优化
        if args.fast_loading:
            args.offload_model = False
            args.t5_cpu = False
            if rank == 0:
                print("🚀 快速加载模式: 禁用模型卸载，所有模型常驻GPU")
        
        # 多GPU环境优化
        if world_size > 1:
            args.offload_model = False  # 多GPU时禁用模型卸载
            # 设置更保守的内存和并发配置
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            if rank == 0:
                print(f"🔧 多GPU优化: {world_size}GPU环境，自动禁用模型卸载")
                print(f"🔧 线程优化: 设置OMP_NUM_THREADS=1避免过载")
        
        # 模型加载时间记录
        model_load_start = time.time()
        logging.info("Creating WanT2V pipeline...")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        model_load_time = time.time() - model_load_start
        if rank == 0:
            print(f"⏱️ 模型加载耗时: {model_load_time:.2f}秒")

        # 推理时间记录
        inference_start = time.time()
        logging.info(f"开始推理...")
        print(f"🎬 CFG截断配置: 低噪声专家{args.cfg_truncate_steps}步, 高噪声专家{args.cfg_truncate_high_noise_steps}步")
        # 创建输出文件夹结构（在推理前创建，以便传递给模型）
        output_base_dir = Path("./outputs")
        output_base_dir.mkdir(exist_ok=True)
        
        # 创建本次推理的子文件夹
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:30]
        run_folder = output_base_dir / f"{args.task}_{formatted_time}_{formatted_prompt}"
        run_folder.mkdir(exist_ok=True)
        
        video, timing_info = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            cfg_truncate_steps=args.cfg_truncate_steps,
            cfg_truncate_high_noise_steps=args.cfg_truncate_high_noise_steps,
            output_dir=str(run_folder),
            enable_half_frame_generation=args.enable_half_frame_generation,
            enable_attention_visualization=args.enable_attention_visualization,
            attention_output_dir=args.attention_output_dir,
)
        total_inference_time = time.time() - inference_start
        
        # 提取时间信息
        total_switch_time = timing_info.get('total_switch_time', 0.0)
        step_timings = timing_info.get('step_timings', [])
        
        pure_inference_time = total_inference_time - total_switch_time
        
        if rank == 0:
            print(f"🔄 专家切换总耗时: {total_switch_time:.2f}秒")
            print(f"⚡ 纯推理耗时: {pure_inference_time:.2f}秒")
            print(f"📊 总推理耗时: {total_inference_time:.2f}秒")
            print(f"📈 推理速度: {args.sample_steps/pure_inference_time:.3f} 步/秒")
            print(f"📈 每步耗时: {pure_inference_time/args.sample_steps:.3f} 秒/步")
            if args.frame_num > 1:
                print(f"🎬 帧生成效率: {args.frame_num/pure_inference_time:.3f} 帧/秒")
    elif "ti2v" in args.task:
        logging.info("Creating WanTI2V pipeline.")
        wan_ti2v = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        video = wan_ti2v.generate(
            args.prompt,
            img=img,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "s2v" in args.task:
        logging.info("Creating WanS2V pipeline.")
        wan_s2v = wan.WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        logging.info(f"Generating video ...")
        video = wan_s2v.generate(
            input_prompt=args.prompt,
            ref_image_path=args.image,
            audio_path=args.audio,
            enable_tts=args.enable_tts,
            tts_prompt_audio=args.tts_prompt_audio,
            tts_prompt_text=args.tts_prompt_text,
            tts_text=args.tts_text,
            num_repeat=args.num_clip,
            pose_video=args.pose_video,
            max_area=MAX_AREA_CONFIGS[args.size],
            infer_frames=args.infer_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            init_first_frame=args.start_from_ref,
        )

    else:
        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            cfg_truncate_steps=args.cfg_truncate_steps,
            cfg_truncate_high_noise_steps=args.cfg_truncate_high_noise_steps)

    if rank == 0:
        
        # 视频文件路径
        if args.save_file is None:
            video_filename = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{formatted_time}.mp4"
        else:
            video_filename = Path(args.save_file).name
        video_path = run_folder / video_filename
        
        # 保存视频
        logging.info(f"Saving generated video to {video_path}")
        save_video(
            tensor=video[None],
            save_file=str(video_path),
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        
        # 创建推理记录文件
        record_data = {
            "推理信息": {
                "时间戳": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "任务类型": args.task,
                "模型路径": args.ckpt_dir,
                "输出视频": video_filename
            },
            "参数设置": {
                "提示词": args.prompt,
                "分辨率": args.size,
                "帧数": args.frame_num,
                "采样步数": args.sample_steps,
                "CFG强度": args.sample_guide_scale,
                "CFG截断步数": args.cfg_truncate_steps,
                "高噪声CFG截断": args.cfg_truncate_high_noise_steps,
                "噪声调度偏移": args.sample_shift,
                "采样器": args.sample_solver,
                "随机种子": args.base_seed,
                "快速加载": args.fast_loading,
                "模型卸载": args.offload_model,
                "T5_CPU": args.t5_cpu,
                "数据类型转换": args.convert_model_dtype,
            },
            "分布式设置": {
                "多GPU": dist.is_initialized() if 'dist' in globals() else False,
                "GPU数量": dist.get_world_size() if dist.is_initialized() else 1,
                "DiT_FSDP": args.dit_fsdp,
                "T5_FSDP": args.t5_fsdp,
                "Ulysses大小": args.ulysses_size
            },
            "性能数据": {
                "模型加载耗时(秒)": f"{model_load_time:.2f}",
                "专家切换总耗时(秒)": f"{total_switch_time:.2f}",
                "纯推理耗时(秒)": f"{pure_inference_time:.2f}",
                "总推理耗时(秒)": f"{total_inference_time:.2f}",
                "总耗时(秒)": f"{model_load_time + total_inference_time:.2f}",
                "推理速度(步/秒)": f"{args.sample_steps/pure_inference_time:.3f}",
                "每步耗时(秒/步)": f"{pure_inference_time/args.sample_steps:.3f}",
                "帧生成效率(帧/秒)": f"{args.frame_num/pure_inference_time:.3f}" if args.frame_num > 1 else "单帧生成",
                "每帧纯推理耗时(秒)": f"{pure_inference_time/args.frame_num:.3f}",
                "每帧总耗时(秒)": f"{total_inference_time/args.frame_num:.3f}",
                "每步详细时间": step_timings  # 添加每步时间记录
            }
        }
        
        # 保存记录文件
        record_path = run_folder / "inference_record.json"
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(record_data, f, ensure_ascii=False, indent=2)
        
        print(f"📁 输出保存到: {run_folder}")
        print(f"🎬 视频文件: {video_path}")
        print(f"📊 记录文件: {record_path}")
        if "s2v" in args.task:
            if args.enable_tts is False:
                merge_video_audio(video_path=str(video_path), audio_path=args.audio)
            else:
                merge_video_audio(video_path=str(video_path), audio_path="tts.wav")
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
