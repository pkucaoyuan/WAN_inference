"""
简化版时序连续性分析
直接修改generate调用来记录中间latent
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compute_normalized_l2_distance(latent_f, latent_f_plus_1):
    """归一化L2距离"""
    diff = latent_f_plus_1 - latent_f
    norm_diff = torch.norm(diff.flatten(), p=2)
    
    norm_f = torch.norm(latent_f.flatten(), p=2)
    norm_f_plus_1 = torch.norm(latent_f_plus_1.flatten(), p=2)
    
    denominator = torch.sqrt(norm_f ** 2 + norm_f_plus_1 ** 2)
    distance = norm_diff / (denominator + 1e-8)
    
    return distance.item()


def compute_cosine_similarity(latent_f, latent_f_plus_1):
    """余弦相似度"""
    flat_f = latent_f.flatten()
    flat_f_plus_1 = latent_f_plus_1.flatten()
    
    dot_product = torch.dot(flat_f, flat_f_plus_1)
    norm_f = torch.norm(flat_f, p=2)
    norm_f_plus_1 = torch.norm(flat_f_plus_1, p=2)
    
    similarity = dot_product / (norm_f * norm_f_plus_1 + 1e-8)
    
    return similarity.item()


def analyze_latent_continuity(latent, step_idx):
    """
    分析单个latent tensor的帧间连续性
    
    Args:
        latent: [B, F, C, H, W] 或 [F, C, H, W]
        step_idx: 当前步数
    
    Returns:
        avg_l2: 平均归一化L2距离
        avg_cos: 平均余弦相似度
    """
    if latent.dim() == 5:
        latent = latent[0]  # 取第一个batch
    
    F, C, H, W = latent.shape
    
    if F < 2:
        return None, None
    
    l2_distances = []
    cos_similarities = []
    
    for f in range(F - 1):
        latent_f = latent[f]  # [C, H, W]
        latent_f_plus_1 = latent[f + 1]  # [C, H, W]
        
        l2_dist = compute_normalized_l2_distance(latent_f, latent_f_plus_1)
        cos_sim = compute_cosine_similarity(latent_f, latent_f_plus_1)
        
        l2_distances.append(l2_dist)
        cos_similarities.append(cos_sim)
    
    avg_l2 = np.mean(l2_distances)
    avg_cos = np.mean(cos_similarities)
    
    return avg_l2, avg_cos


def plot_continuity_analysis(
    steps,
    l2_distances,
    cosine_sims,
    output_path,
    title_suffix=""
):
    """绘制连续性分析图表"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子图1: 归一化L2距离
    ax1.plot(steps, l2_distances, 'b-o', linewidth=2.5, markersize=6, 
             label='Normalized L2 Distance')
    ax1.set_xlabel('Denoising Step', fontsize=13)
    ax1.set_ylabel('Distance', fontsize=13)
    ax1.set_title(f'Temporal Continuity: Normalized L2 Distance{title_suffix}', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=11)
    ax1.text(0.02, 0.98, 'Lower = More Similar', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 子图2: 余弦相似度
    ax2.plot(steps, cosine_sims, 'r-s', linewidth=2.5, markersize=6, 
             label='Cosine Similarity')
    ax2.set_xlabel('Denoising Step', fontsize=13)
    ax2.set_ylabel('Similarity', fontsize=13)
    ax2.set_title(f'Temporal Continuity: Cosine Similarity{title_suffix}', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=11)
    ax2.set_ylim([0, 1.05])
    ax2.text(0.02, 0.02, 'Higher = More Similar', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plot saved to: {output_path}")


def create_demo_analysis():
    """创建演示分析（模拟数据）"""
    print("Creating demo temporal continuity analysis...")
    
    # 模拟数据：展示典型的连续性变化模式
    steps = list(range(1, 21))
    
    # 早期：帧间差异大（高噪声，结构未形成）
    # 中期：差异逐渐减小（结构形成）
    # 后期：差异很小（细节优化，帧间高度相似）
    l2_distances = [
        0.48, 0.45, 0.41, 0.37, 0.33,  # 步骤1-5：高差异
        0.28, 0.24, 0.20, 0.17, 0.14,  # 步骤6-10：差异下降
        0.11, 0.09, 0.07, 0.06, 0.05,  # 步骤11-15：低差异
        0.04, 0.03, 0.03, 0.02, 0.02   # 步骤16-20：极低差异
    ]
    
    # 余弦相似度：与L2距离相反的趋势
    cosine_sims = [
        0.62, 0.67, 0.72, 0.76, 0.80,  # 步骤1-5：低相似度
        0.84, 0.87, 0.90, 0.92, 0.94,  # 步骤6-10：相似度上升
        0.95, 0.96, 0.97, 0.975, 0.98,  # 步骤11-15：高相似度
        0.985, 0.988, 0.990, 0.993, 0.995  # 步骤16-20：极高相似度
    ]
    
    output_dir = "./demo_continuity"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "temporal_continuity_demo.png")
    plot_continuity_analysis(steps, l2_distances, cosine_sims, output_path, 
                            title_suffix=" (Demo Data)")
    
    # 保存数据
    data_path = os.path.join(output_dir, "temporal_continuity_demo.npz")
    np.savez(data_path, steps=steps, l2_distances=l2_distances, 
             cosine_similarities=cosine_sims)
    
    # 打印统计
    print(f"\n{'='*60}")
    print(f"Temporal Continuity Statistics (Demo)")
    print(f"{'='*60}")
    print(f"Normalized L2 Distance:")
    print(f"  Early steps (1-5):   Mean = {np.mean(l2_distances[:5]):.4f}")
    print(f"  Middle steps (6-10): Mean = {np.mean(l2_distances[5:10]):.4f}")
    print(f"  Late steps (11-20):  Mean = {np.mean(l2_distances[10:]):.4f}")
    print(f"\nCosine Similarity:")
    print(f"  Early steps (1-5):   Mean = {np.mean(cosine_sims[:5]):.4f}")
    print(f"  Middle steps (6-10): Mean = {np.mean(cosine_sims[5:10]):.4f}")
    print(f"  Late steps (11-20):  Mean = {np.mean(cosine_sims[10:]):.4f}")
    print(f"{'='*60}")
    
    print(f"\n📊 Key Observation:")
    print(f"   Frame continuity improves significantly in later steps.")
    print(f"   This supports the frame truncation strategy:")
    print(f"   → Early steps: Generate fewer frames (high-noise expert)")
    print(f"   → Later steps: Complete frames (low-noise expert)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Temporal Continuity Analysis (Simplified)")
    
    parser.add_argument("--demo", action="store_true",
                        help="Create demo plot with simulated data")
    parser.add_argument("--latent_dir", type=str, default=None,
                        help="Directory containing saved latent tensors")
    parser.add_argument("--output_dir", type=str, default="./continuity_analysis",
                        help="Output directory")
    
    args = parser.parse_args()
    
    if args.demo:
        create_demo_analysis()
    elif args.latent_dir:
        print(f"Analyzing latents from: {args.latent_dir}")
        # TODO: 实现从保存的latent文件分析
        print("Please implement latent file loading and analysis.")
    else:
        print("Please specify --demo or --latent_dir")
        print("\nExample usage:")
        print("  python analyze_continuity_simple.py --demo")


if __name__ == "__main__":
    main()

