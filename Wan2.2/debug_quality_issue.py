#!/usr/bin/env python3
"""
视频质量下降调试脚本
用于诊断帧数补全过程中的问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def debug_latents_quality(latents, step_name, save_dir="debug_outputs"):
    """调试latents质量"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    if isinstance(latents, list):
        latents = latents[0]
    
    print(f"\n🔍 {step_name} - Latents质量分析:")
    print(f"  形状: {latents.shape}")
    print(f"  数据类型: {latents.dtype}")
    print(f"  设备: {latents.device}")
    print(f"  数值范围: [{latents.min():.6f}, {latents.max():.6f}]")
    print(f"  均值: {latents.mean():.6f}")
    print(f"  标准差: {latents.std():.6f}")
    print(f"  是否包含NaN: {torch.isnan(latents).any()}")
    print(f"  是否包含Inf: {torch.isinf(latents).any()}")
    
    # 保存统计信息
    stats = {
        'shape': list(latents.shape),
        'dtype': str(latents.dtype),
        'device': str(latents.device),
        'min': float(latents.min()),
        'max': float(latents.max()),
        'mean': float(latents.mean()),
        'std': float(latents.std()),
        'has_nan': bool(torch.isnan(latents).any()),
        'has_inf': bool(torch.isinf(latents).any())
    }
    
    # 保存到文件
    import json
    with open(save_dir / f"{step_name}_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def debug_timesteps_sequence(timesteps, boundary, step_name, save_dir="debug_outputs"):
    """调试时间步序列"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"\n🔍 {step_name} - 时间步序列分析:")
    print(f"  总步数: {len(timesteps)}")
    print(f"  边界值: {boundary}")
    print(f"  时间步范围: [{timesteps[0].item():.1f}, {timesteps[-1].item():.1f}]")
    
    # 分析高噪声和低噪声步数
    high_noise_steps = [i for i, ts in enumerate(timesteps) if ts.item() >= boundary]
    low_noise_steps = [i for i, ts in enumerate(timesteps) if ts.item() < boundary]
    
    print(f"  高噪声步数: {len(high_noise_steps)}")
    print(f"  低噪声步数: {len(low_noise_steps)}")
    print(f"  高噪声时间步: {[timesteps[i].item() for i in high_noise_steps[:5]]}...")
    print(f"  低噪声时间步: {[timesteps[i].item() for i in low_noise_steps[:5]]}...")
    
    # 可视化时间步分布
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(timesteps)), [t.item() for t in timesteps], 'b-', linewidth=2)
    plt.axhline(y=boundary, color='r', linestyle='--', label=f'Boundary ({boundary})')
    plt.xlabel('Step Index')
    plt.ylabel('Timestep Value')
    plt.title(f'{step_name} - Timestep Sequence')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist([t.item() for t in timesteps], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(x=boundary, color='r', linestyle='--', label=f'Boundary ({boundary})')
    plt.xlabel('Timestep Value')
    plt.ylabel('Frequency')
    plt.title(f'{step_name} - Timestep Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / f"{step_name}_timesteps.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'total_steps': len(timesteps),
        'boundary': boundary,
        'high_noise_steps': len(high_noise_steps),
        'low_noise_steps': len(low_noise_steps),
        'timestep_range': [float(timesteps[0]), float(timesteps[-1])]
    }

def debug_scheduler_state(scheduler, step_name, save_dir="debug_outputs"):
    """调试scheduler状态"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"\n🔍 {step_name} - Scheduler状态:")
    print(f"  类型: {type(scheduler).__name__}")
    print(f"  当前步骤索引: {getattr(scheduler, 'step_index', 'N/A')}")
    print(f"  时间步序列长度: {len(getattr(scheduler, 'timesteps', []))}")
    
    # 检查scheduler内部状态
    state_info = {}
    for attr in ['step_index', 'timesteps', 'sigmas', 'num_inference_steps']:
        if hasattr(scheduler, attr):
            value = getattr(scheduler, attr)
            if isinstance(value, torch.Tensor):
                state_info[attr] = {
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'device': str(value.device),
                    'min': float(value.min()) if value.numel() > 0 else None,
                    'max': float(value.max()) if value.numel() > 0 else None
                }
            else:
                state_info[attr] = value
    
    # 保存状态信息
    import json
    with open(save_dir / f"{step_name}_scheduler_state.json", 'w') as f:
        json.dump(state_info, f, indent=2)
    
    return state_info

def compare_latents_before_after(before, after, step_name, save_dir="debug_outputs"):
    """比较帧数补全前后的latents"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    if isinstance(before, list):
        before = before[0]
    if isinstance(after, list):
        after = after[0]
    
    print(f"\n🔍 {step_name} - 帧数补全前后对比:")
    print(f"  补全前形状: {before.shape}")
    print(f"  补全后形状: {after.shape}")
    
    # 计算差异
    if before.shape == after.shape:
        diff = torch.abs(after - before)
        print(f"  绝对差异 - 均值: {diff.mean():.6f}, 最大值: {diff.max():.6f}")
        
        # 可视化差异
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(before[0, 0].cpu().numpy(), cmap='viridis')
        plt.title('Before Frame Completion')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(after[0, 0].cpu().numpy(), cmap='viridis')
        plt.title('After Frame Completion')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(diff[0, 0].cpu().numpy(), cmap='hot')
        plt.title('Absolute Difference')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(save_dir / f"{step_name}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        'before_shape': list(before.shape),
        'after_shape': list(after.shape),
        'shape_changed': before.shape != after.shape
    }

def debug_frame_completion_process(latents_before, latents_after, target_frames, save_dir="debug_outputs"):
    """调试帧数补全过程"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    if isinstance(latents_before, list):
        latents_before = latents_before[0]
    if isinstance(latents_after, list):
        latents_after = latents_after[0]
    
    current_frames = latents_before.shape[1]
    
    print(f"\n🔍 帧数补全过程分析:")
    print(f"  当前帧数: {current_frames}")
    print(f"  目标帧数: {target_frames}")
    print(f"  补全后帧数: {latents_after.shape[1]}")
    
    # 分析帧间关系
    if latents_after.shape[1] >= current_frames * 2:
        # 检查复制关系
        print(f"  检查帧复制关系:")
        for i in range(min(3, current_frames)):  # 检查前3帧
            original_frame = latents_before[0, i]
            copied_frame_1 = latents_after[0, i*2]
            copied_frame_2 = latents_after[0, i*2+1] if i*2+1 < latents_after.shape[1] else None
            
            diff1 = torch.abs(copied_frame_1 - original_frame).mean()
            print(f"    帧{i} -> 帧{i*2}: 差异 {diff1:.6f}")
            
            if copied_frame_2 is not None:
                diff2 = torch.abs(copied_frame_2 - original_frame).mean()
                print(f"    帧{i} -> 帧{i*2+1}: 差异 {diff2:.6f}")
    
    return {
        'current_frames': current_frames,
        'target_frames': target_frames,
        'final_frames': latents_after.shape[1],
        'completion_ratio': latents_after.shape[1] / current_frames
    }

if __name__ == "__main__":
    print("🔍 视频质量调试工具")
    print("使用方法:")
    print("1. 在text2video.py中导入此模块")
    print("2. 在关键位置调用调试函数")
    print("3. 查看生成的调试输出和图表")
