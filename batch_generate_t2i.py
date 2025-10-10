"""
批量生成T2I图片脚本
从MS-COCO prompts.csv读取提示词，批量生成单帧图片用于评估
"""
import os
import sys
import argparse
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import time

# 添加项目路径 - Wan2.2子目录
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'Wan2.2'))
sys.path.insert(0, script_dir)

from generate import setup_model, parse_args


def batch_generate_t2i(
    prompts_csv_path: str,
    output_dir: str,
    num_samples: int = None,
    seed_start: int = 42,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    height: int = 480,
    width: int = 832,
    enable_half_frame: bool = False,
    cfg_truncation_step: int = None,
    model_path: str = None,
    device: str = "cuda:0"
):
    """
    批量生成T2I图片
    
    Args:
        prompts_csv_path: MS-COCO prompts CSV文件路径
        output_dir: 输出目录
        num_samples: 生成样本数量（None表示全部）
        seed_start: 起始随机种子
        num_inference_steps: 推理步数
        guidance_scale: CFG引导强度
        height: 图片高度
        width: 图片宽度
        enable_half_frame: 是否启用帧数减半优化
        cfg_truncation_step: CFG截断步数
        model_path: 模型路径
        device: 设备
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取prompts
    print(f"📖 读取prompts文件: {prompts_csv_path}")
    df = pd.read_csv(prompts_csv_path)
    
    # 检查CSV格式
    if 'prompt' not in df.columns and 'caption' not in df.columns:
        raise ValueError("CSV文件必须包含 'prompt' 或 'caption' 列")
    
    prompt_column = 'prompt' if 'prompt' in df.columns else 'caption'
    prompts = df[prompt_column].tolist()
    
    # 如果有image_id列，使用它作为文件名
    if 'image_id' in df.columns:
        image_ids = df['image_id'].tolist()
    else:
        image_ids = [f"{i:06d}" for i in range(len(prompts))]
    
    # 限制样本数量
    if num_samples is not None:
        prompts = prompts[:num_samples]
        image_ids = image_ids[:num_samples]
    
    print(f"✅ 共加载 {len(prompts)} 个prompts")
    
    # 设置模型（使用generate.py中的setup函数）
    print(f"🔧 加载模型...")
    
    # 构造参数对象
    class Args:
        def __init__(self):
            self.model_path = model_path or "path/to/model"
            self.prompt = ""  # 临时占位
            self.negative_prompt = ""
            self.height = height
            self.width = width
            self.num_frames = 1  # T2I只生成1帧
            self.num_inference_steps = num_inference_steps
            self.guidance_scale = guidance_scale
            self.seed = seed_start
            self.output_dir = output_dir
            self.enable_half_frame_generation = enable_half_frame
            self.cfg_truncation_step = cfg_truncation_step
            self.enable_debug = False
            self.debug_output_dir = None
            self.device = device
            self.dtype = "bf16"
            self.enable_tiling = False
            self.tile_sample_min_height = 0
            self.tile_sample_min_width = 0
            self.enable_vae_tiling = False
            self.vae_tile_sample_min_height = 0
            self.vae_tile_sample_min_width = 0
    
    args = Args()
    
    # 加载模型
    try:
        import wan
        from omegaconf import OmegaConf
        
        # 加载配置
        config_path = os.path.join(model_path, "config.yaml")
        if not os.path.exists(config_path):
            # 尝试默认配置路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "Wan2.2/configs/t2v_A14B.yaml")
        
        cfg = OmegaConf.load(config_path)
        
        # 提取device编号
        device_id = int(device.split(':')[1]) if ':' in device else 0
        
        # 初始化模型
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=model_path,
            device_id=device_id,
            rank=device_id,
            t5_fsdp=False
        )
        
        print(f"✅ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请确保模型路径正确，并且已安装所有依赖")
        import traceback
        traceback.print_exc()
        return
    
    # 批量生成
    print(f"\n🎨 开始批量生成 {len(prompts)} 张图片...")
    print(f"📁 输出目录: {output_dir}")
    print(f"⚙️  配置: {height}x{width}, steps={num_inference_steps}, cfg={guidance_scale}")
    
    success_count = 0
    failed_prompts = []
    
    for idx, (prompt, image_id) in enumerate(tqdm(zip(prompts, image_ids), total=len(prompts))):
        try:
            # 设置当前prompt和seed
            args.prompt = prompt
            args.seed = seed_start + idx
            
            # 输出文件名
            output_filename = f"{image_id}_seed{args.seed}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 如果文件已存在，跳过
            if os.path.exists(output_path):
                print(f"⏭️  跳过已存在的文件: {output_filename}")
                success_count += 1
                continue
            
            # 生成图片
            start_time = time.time()
            
            video, timing_info = wan_t2v.generate(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                enable_half_frame_generation=args.enable_half_frame_generation,
                cfg_truncation_step=args.cfg_truncation_step,
                enable_debug=False
            )
            
            generation_time = time.time() - start_time
            
            # 保存图片（video是tensor，shape: [B, F, C, H, W]）
            import torchvision
            frame = video[0, 0]  # 取第一帧
            torchvision.utils.save_image(frame, output_path, normalize=True, value_range=(-1, 1))
            
            success_count += 1
            
            if (idx + 1) % 10 == 0:
                print(f"\n✅ 已完成 {success_count}/{len(prompts)} 张图片")
                print(f"   最近一张耗时: {generation_time:.2f}秒")
        
        except Exception as e:
            print(f"\n❌ 生成失败 [{idx+1}/{len(prompts)}]: {prompt[:50]}...")
            print(f"   错误: {e}")
            failed_prompts.append((idx, prompt, str(e)))
            continue
    
    # 输出统计
    print(f"\n{'='*60}")
    print(f"🎉 批量生成完成!")
    print(f"✅ 成功: {success_count}/{len(prompts)}")
    print(f"❌ 失败: {len(failed_prompts)}/{len(prompts)}")
    print(f"📁 输出目录: {output_dir}")
    
    # 保存失败记录
    if failed_prompts:
        failed_log_path = os.path.join(output_dir, "failed_prompts.txt")
        with open(failed_log_path, 'w', encoding='utf-8') as f:
            for idx, prompt, error in failed_prompts:
                f.write(f"[{idx}] {prompt}\n")
                f.write(f"Error: {error}\n\n")
        print(f"📝 失败记录已保存到: {failed_log_path}")


def main():
    parser = argparse.ArgumentParser(description="批量生成T2I图片用于评估")
    
    # 输入输出
    parser.add_argument("--prompts_csv", type=str, required=True,
                        help="MS-COCO prompts CSV文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出图片目录")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="生成样本数量（默认全部）")
    
    # 模型配置
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型checkpoint目录路径（包含权重和config.yaml）")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备")
    
    # 生成参数
    parser.add_argument("--seed_start", type=int, default=42,
                        help="起始随机种子")
    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="CFG引导强度")
    parser.add_argument("--height", type=int, default=480,
                        help="图片高度")
    parser.add_argument("--width", type=int, default=832,
                        help="图片宽度")
    
    # 优化选项
    parser.add_argument("--enable_half_frame", action="store_true",
                        help="启用帧数减半优化")
    parser.add_argument("--cfg_truncation_step", type=int, default=None,
                        help="CFG截断步数")
    
    args = parser.parse_args()
    
    # 执行批量生成
    batch_generate_t2i(
        prompts_csv_path=args.prompts_csv,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed_start=args.seed_start,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        enable_half_frame=args.enable_half_frame,
        cfg_truncation_step=args.cfg_truncation_step,
        model_path=args.model_path,
        device=args.device
    )


if __name__ == "__main__":
    main()

