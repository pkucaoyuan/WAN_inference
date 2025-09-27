#!/usr/bin/env python3
"""
CFG截断方法 vs Baseline 对比脚本
比较两种方法的最终生成结果误差
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wan.text2video import WanT2V
from wan.utils import parse_size


def load_model(ckpt_dir, device):
    """加载模型"""
    print("🔄 加载模型...")
    model = WanT2V(ckpt_dir=ckpt_dir, device=device)
    print("✅ 模型加载完成")
    return model


def generate_video(model, prompt, size, frame_num, sample_steps, 
                  cfg_truncate_steps, cfg_truncate_high_noise_steps, 
                  seed, output_dir, method_name):
    """生成视频并返回结果"""
    print(f"\n🎬 开始生成 ({method_name})...")
    print(f"📝 提示词: {prompt}")
    print(f"📐 尺寸: {size}")
    print(f"🎞️ 帧数: {frame_num}")
    print(f"🔄 采样步数: {sample_steps}")
    print(f"⚙️ CFG截断步数: {cfg_truncate_steps}")
    print(f"⚙️ 高噪声CFG截断步数: {cfg_truncate_high_noise_steps}")
    print(f"🎲 种子: {seed}")
    
    # 设置随机种子确保可重复性
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 生成视频
    result = model.generate(
        prompt=prompt,
        size=size,
        frame_num=frame_num,
        sample_steps=sample_steps,
        cfg_truncate_steps=cfg_truncate_steps,
        cfg_truncate_high_noise_steps=cfg_truncate_high_noise_steps,
        output_dir=output_dir,
        seed=seed
    )
    
    print(f"✅ {method_name} 生成完成")
    return result


def calculate_error(result1, result2, method1_name, method2_name):
    """计算两种方法的误差"""
    print(f"\n📊 计算 {method1_name} vs {method2_name} 的误差...")
    
    # 获取生成的视频数据
    video1 = result1['video']  # [B, C, T, H, W]
    video2 = result2['video']  # [B, C, T, H, W]
    
    # 确保两个视频形状相同
    if video1.shape != video2.shape:
        print(f"⚠️ 警告: 视频形状不匹配")
        print(f"   {method1_name}: {video1.shape}")
        print(f"   {method2_name}: {video2.shape}")
        
        # 取最小尺寸
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(video1.shape, video2.shape))
        video1 = video1[tuple(slice(0, s) for s in min_shape)]
        video2 = video2[tuple(slice(0, s) for s in min_shape)]
        print(f"   调整后形状: {video1.shape}")
    
    # 转换为float32进行计算
    video1 = video1.float()
    video2 = video2.float()
    
    # 计算绝对误差
    absolute_error = torch.abs(video1 - video2)
    abs_error_mean = absolute_error.mean().item()
    abs_error_max = absolute_error.max().item()
    abs_error_std = absolute_error.std().item()
    
    # 计算相对误差 (避免除零)
    epsilon = 1e-8
    relative_error = absolute_error / (torch.abs(video2) + epsilon)
    rel_error_mean = relative_error.mean().item()
    rel_error_max = relative_error.max().item()
    rel_error_std = relative_error.std().item()
    
    # 计算MSE和PSNR
    mse = torch.mean((video1 - video2) ** 2).item()
    if mse > 0:
        psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()
    else:
        psnr = float('inf')
    
    # 计算SSIM (简化版本)
    def ssim_simple(x, y):
        """简化的SSIM计算"""
        mu1 = x.mean()
        mu2 = y.mean()
        sigma1 = x.var()
        sigma2 = y.var()
        sigma12 = ((x - mu1) * (y - mu2)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        return ssim.item()
    
    ssim_value = ssim_simple(video1, video2)
    
    return {
        'absolute_error': {
            'mean': abs_error_mean,
            'max': abs_error_max,
            'std': abs_error_std
        },
        'relative_error': {
            'mean': rel_error_mean,
            'max': rel_error_max,
            'std': rel_error_std
        },
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim_value
    }


def print_error_report(error_stats, method1_name, method2_name):
    """打印误差报告"""
    print(f"\n" + "="*80)
    print(f"📊 {method1_name} vs {method2_name} 误差分析报告")
    print(f"="*80)
    
    print(f"\n🔍 绝对误差 (Absolute Error):")
    print(f"   平均值: {error_stats['absolute_error']['mean']:.6f}")
    print(f"   最大值: {error_stats['absolute_error']['max']:.6f}")
    print(f"   标准差: {error_stats['absolute_error']['std']:.6f}")
    
    print(f"\n📈 相对误差 (Relative Error):")
    print(f"   平均值: {error_stats['relative_error']['mean']:.6f}")
    print(f"   最大值: {error_stats['relative_error']['max']:.6f}")
    print(f"   标准差: {error_stats['relative_error']['std']:.6f}")
    
    print(f"\n📏 图像质量指标:")
    print(f"   MSE: {error_stats['mse']:.6f}")
    print(f"   PSNR: {error_stats['psnr']:.2f} dB")
    print(f"   SSIM: {error_stats['ssim']:.6f}")
    
    print(f"\n💡 解释:")
    print(f"   - 绝对误差越小，两种方法生成的结果越相似")
    print(f"   - 相对误差越小，相对差异越小")
    print(f"   - PSNR越高，图像质量越好")
    print(f"   - SSIM越接近1，结构相似性越高")
    
    print(f"="*80)


def main():
    parser = argparse.ArgumentParser(description='CFG截断方法 vs Baseline 对比')
    parser.add_argument('--task', type=str, default='t2v-A14B', help='任务类型')
    parser.add_argument('--size', type=str, default='1280*720', help='视频尺寸')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='模型检查点目录')
    parser.add_argument('--frame_num', type=int, default=1, help='帧数')
    parser.add_argument('--sample_steps', type=int, default=20, help='采样步数')
    parser.add_argument('--prompt', type=str, required=True, help='生成提示词')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output_dir', type=str, default='comparison_outputs', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    # 解析尺寸
    size = parse_size(args.size)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 开始CFG截断方法 vs Baseline对比实验")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🎲 随机种子: {args.seed}")
    
    # 加载模型
    model = load_model(args.ckpt_dir, args.device)
    
    # 方法1: CFG截断方法
    cfg_output_dir = os.path.join(args.output_dir, "cfg_truncated")
    result_cfg = generate_video(
        model=model,
        prompt=args.prompt,
        size=size,
        frame_num=args.frame_num,
        sample_steps=args.sample_steps,
        cfg_truncate_steps=5,
        cfg_truncate_high_noise_steps=3,
        seed=args.seed,
        output_dir=cfg_output_dir,
        method_name="CFG截断方法"
    )
    
    # 方法2: Baseline (无截断)
    baseline_output_dir = os.path.join(args.output_dir, "baseline")
    result_baseline = generate_video(
        model=model,
        prompt=args.prompt,
        size=size,
        frame_num=args.frame_num,
        sample_steps=args.sample_steps,
        cfg_truncate_steps=0,
        cfg_truncate_high_noise_steps=0,
        seed=args.seed,
        output_dir=baseline_output_dir,
        method_name="Baseline方法"
    )
    
    # 计算误差
    error_stats = calculate_error(
        result_cfg, result_baseline, 
        "CFG截断方法", "Baseline方法"
    )
    
    # 打印报告
    print_error_report(error_stats, "CFG截断方法", "Baseline方法")
    
    # 保存误差统计到文件
    error_file = os.path.join(args.output_dir, "error_comparison.txt")
    with open(error_file, 'w', encoding='utf-8') as f:
        f.write("CFG截断方法 vs Baseline 误差对比\n")
        f.write("="*50 + "\n\n")
        f.write(f"绝对误差平均值: {error_stats['absolute_error']['mean']:.6f}\n")
        f.write(f"绝对误差最大值: {error_stats['absolute_error']['max']:.6f}\n")
        f.write(f"相对误差平均值: {error_stats['relative_error']['mean']:.6f}\n")
        f.write(f"相对误差最大值: {error_stats['relative_error']['max']:.6f}\n")
        f.write(f"MSE: {error_stats['mse']:.6f}\n")
        f.write(f"PSNR: {error_stats['psnr']:.2f} dB\n")
        f.write(f"SSIM: {error_stats['ssim']:.6f}\n")
    
    print(f"\n💾 误差统计已保存到: {error_file}")
    print("🎉 对比实验完成!")


if __name__ == "__main__":
    main()
