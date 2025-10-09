"""
多GPU批量生成T2I图片脚本
每张卡处理不同的样本，实现并行加速
"""
import os
import sys
import argparse
import pandas as pd
import torch
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def worker_generate(
    rank: int,
    world_size: int,
    prompts_csv_path: str,
    output_dir: str,
    num_samples: int,
    seed_start: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    enable_half_frame: bool,
    cfg_truncation_step: int,
    model_path: str,
    negative_prompt: str,
    dtype: str
):
    """
    单个GPU worker的生成函数
    
    Args:
        rank: GPU编号
        world_size: 总GPU数量
        其他参数同batch_generate_t2i
    """
    device = f"cuda:{rank}"
    
    print(f"[GPU {rank}] 🚀 启动worker，设备: {device}")
    
    # 读取prompts
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
    
    # 任务分配：每个GPU处理不同的样本
    # 使用简单的轮询分配策略
    my_indices = list(range(rank, len(prompts), world_size))
    my_prompts = [prompts[i] for i in my_indices]
    my_image_ids = [image_ids[i] for i in my_indices]
    
    print(f"[GPU {rank}] 📋 分配到 {len(my_prompts)} 个任务 (总共 {len(prompts)} 个)")
    print(f"[GPU {rank}] 📝 任务索引范围: {my_indices[0]} 到 {my_indices[-1]} (步长 {world_size})")
    
    # 加载模型
    print(f"[GPU {rank}] 🔧 加载模型...")
    
    try:
        import wan
        from omegaconf import OmegaConf
        
        # 加载配置
        config_path = os.path.join(model_path, "config.yaml")
        if not os.path.exists(config_path):
            # 尝试默认配置路径
            config_path = "Wan2.2/configs/t2v_A14B.yaml"
        
        cfg = OmegaConf.load(config_path)
        
        # 初始化模型
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=model_path,
            device_id=rank,
            rank=rank,
            t5_fsdp=False
        )
        
        print(f"[GPU {rank}] ✅ 模型加载完成")
    except Exception as e:
        print(f"[GPU {rank}] ❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 批量生成
    print(f"[GPU {rank}] 🎨 开始生成...")
    
    success_count = 0
    failed_prompts = []
    
    # 使用tqdm显示进度（每个GPU独立的进度条）
    pbar = tqdm(
        zip(my_prompts, my_image_ids, my_indices),
        total=len(my_prompts),
        desc=f"GPU {rank}",
        position=rank,
        leave=True
    )
    
    for prompt, image_id, original_idx in pbar:
        try:
            # 计算seed（保持与单GPU版本一致）
            seed = seed_start + original_idx
            
            # 输出文件名
            output_filename = f"{image_id}_seed{seed}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 如果文件已存在，跳过
            if os.path.exists(output_path):
                success_count += 1
                pbar.set_postfix({"success": success_count, "failed": len(failed_prompts)})
                continue
            
            # 生成图片
            video, timing_info = wan_t2v.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=1,  # T2I只生成1帧
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                enable_half_frame_generation=enable_half_frame,
                cfg_truncation_step=cfg_truncation_step,
                enable_debug=False
            )
            
            # 保存图片
            import torchvision
            frame = video[0, 0]  # 取第一帧
            torchvision.utils.save_image(frame, output_path, normalize=True, value_range=(-1, 1))
            
            success_count += 1
            pbar.set_postfix({"success": success_count, "failed": len(failed_prompts)})
        
        except Exception as e:
            print(f"\n[GPU {rank}] ❌ 生成失败 [索引 {original_idx}]: {prompt[:50]}...")
            print(f"[GPU {rank}]    错误: {e}")
            failed_prompts.append((original_idx, prompt, str(e)))
            pbar.set_postfix({"success": success_count, "failed": len(failed_prompts)})
            continue
    
    pbar.close()
    
    # 输出统计
    print(f"\n[GPU {rank}] {'='*60}")
    print(f"[GPU {rank}] 🎉 生成完成!")
    print(f"[GPU {rank}] ✅ 成功: {success_count}/{len(my_prompts)}")
    print(f"[GPU {rank}] ❌ 失败: {len(failed_prompts)}/{len(my_prompts)}")
    
    # 保存失败记录
    if failed_prompts:
        failed_log_path = os.path.join(output_dir, f"failed_prompts_gpu{rank}.txt")
        with open(failed_log_path, 'w', encoding='utf-8') as f:
            for idx, prompt, error in failed_prompts:
                f.write(f"[{idx}] {prompt}\n")
                f.write(f"Error: {error}\n\n")
        print(f"[GPU {rank}] 📝 失败记录已保存到: {failed_log_path}")


def batch_generate_t2i_multigpu(
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
    negative_prompt: str = "",
    dtype: str = "bf16",
    gpu_ids: list = None
):
    """
    多GPU批量生成T2I图片
    
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
        model_path: 模型checkpoint目录路径（包含权重和config.yaml）
        negative_prompt: 负面提示词
        dtype: 数据类型
        gpu_ids: GPU ID列表（例如[0,1,2,3]）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定使用的GPU
    if gpu_ids is None:
        # 自动检测可用GPU
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
    else:
        num_gpus = len(gpu_ids)
    
    if num_gpus == 0:
        print("❌ 未检测到可用的GPU")
        return
    
    print(f"{'='*60}")
    print(f"🚀 多GPU批量生成T2I图片")
    print(f"{'='*60}")
    print(f"📋 配置信息:")
    print(f"   Prompts文件: {prompts_csv_path}")
    print(f"   输出目录: {output_dir}")
    print(f"   样本数量: {num_samples if num_samples else '全部'}")
    print(f"   使用GPU: {gpu_ids} (共 {num_gpus} 张)")
    print(f"   推理步数: {num_inference_steps}")
    print(f"   引导强度: {guidance_scale}")
    print(f"   图片尺寸: {height}x{width}")
    print(f"   帧数减半: {enable_half_frame}")
    print(f"   CFG截断: {cfg_truncation_step if cfg_truncation_step else '不使用'}")
    print(f"{'='*60}\n")
    
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 启动多个进程
    start_time = time.time()
    
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = mp.Process(
            target=worker_generate,
            args=(
                gpu_id,  # rank使用实际的GPU ID
                num_gpus,
                prompts_csv_path,
                output_dir,
                num_samples,
                seed_start,
                num_inference_steps,
                guidance_scale,
                height,
                width,
                enable_half_frame,
                cfg_truncation_step,
                model_path,
                negative_prompt,
                dtype
            )
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    total_time = time.time() - start_time
    
    # 汇总统计
    print(f"\n{'='*60}")
    print(f"✅ 所有GPU任务完成!")
    print(f"{'='*60}")
    print(f"⏱️  总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"📁 输出目录: {output_dir}")
    
    # 统计生成的图片数量
    image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    print(f"🖼️  生成图片数量: {len(image_files)}")
    
    # 检查是否有失败记录
    failed_logs = [f for f in os.listdir(output_dir) if f.startswith('failed_prompts_gpu')]
    if failed_logs:
        print(f"⚠️  发现失败记录文件: {len(failed_logs)} 个")
        for log_file in failed_logs:
            print(f"   - {log_file}")


def main():
    parser = argparse.ArgumentParser(description="多GPU批量生成T2I图片用于评估")
    
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
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=None,
                        help="使用的GPU ID列表，例如: --gpu_ids 0 1 2 3 (默认使用所有可用GPU)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"],
                        help="数据类型")
    
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
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="负面提示词")
    
    # 优化选项
    parser.add_argument("--enable_half_frame", action="store_true",
                        help="启用帧数减半优化")
    parser.add_argument("--cfg_truncation_step", type=int, default=None,
                        help="CFG截断步数")
    
    args = parser.parse_args()
    
    # 执行批量生成
    batch_generate_t2i_multigpu(
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
        negative_prompt=args.negative_prompt,
        dtype=args.dtype,
        gpu_ids=args.gpu_ids
    )


if __name__ == "__main__":
    main()

