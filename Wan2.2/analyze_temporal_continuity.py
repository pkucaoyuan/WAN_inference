"""
Temporal Continuity Analysis in Latent Space
分析生成过程中相邻帧的latent连续性变化
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compute_x0_prediction(x_t, eps_pred, alpha_bar_t):
    """
    从噪声latent和预测噪声计算x0估计
    
    x̂₀ = (x_t - sqrt(1 - ᾱ_t) * ε_θ) / sqrt(ᾱ_t)
    
    Args:
        x_t: 当前噪声latent [B, F, C, H, W]
        eps_pred: 预测的噪声 [B, F, C, H, W]
        alpha_bar_t: 累积alpha值
    
    Returns:
        x0_pred: x0估计 [B, F, C, H, W]
    """
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
    
    # 确保维度正确
    while len(sqrt_alpha_bar.shape) < len(x_t.shape):
        sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
    
    x0_pred = (x_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar
    
    return x0_pred


def compute_normalized_l2_distance(latent_f, latent_f_plus_1):
    """
    计算归一化L2距离（强度型指标）
    
    d = ||x̂₀⁽ᶠ⁺¹⁾ - x̂₀⁽ᶠ⁾||₂ / sqrt(||x̂₀⁽ᶠ⁾||₂² + ||x̂₀⁽ᶠ⁺¹⁾||₂²)
    
    Args:
        latent_f: 帧f的latent [C, H, W]
        latent_f_plus_1: 帧f+1的latent [C, H, W]
    
    Returns:
        distance: 归一化距离标量
    """
    diff = latent_f_plus_1 - latent_f
    norm_diff = torch.norm(diff.flatten(), p=2)
    
    norm_f = torch.norm(latent_f.flatten(), p=2)
    norm_f_plus_1 = torch.norm(latent_f_plus_1.flatten(), p=2)
    
    denominator = torch.sqrt(norm_f ** 2 + norm_f_plus_1 ** 2)
    
    distance = norm_diff / (denominator + 1e-8)
    
    return distance.item()


def compute_cosine_similarity(latent_f, latent_f_plus_1):
    """
    计算余弦相似度（方向型指标）
    
    s = cos(x̂₀⁽ᶠ⁾, x̂₀⁽ᶠ⁺¹⁾)
    
    Args:
        latent_f: 帧f的latent [C, H, W]
        latent_f_plus_1: 帧f+1的latent [C, H, W]
    
    Returns:
        similarity: 余弦相似度标量
    """
    flat_f = latent_f.flatten()
    flat_f_plus_1 = latent_f_plus_1.flatten()
    
    dot_product = torch.dot(flat_f, flat_f_plus_1)
    norm_f = torch.norm(flat_f, p=2)
    norm_f_plus_1 = torch.norm(flat_f_plus_1, p=2)
    
    similarity = dot_product / (norm_f * norm_f_plus_1 + 1e-8)
    
    return similarity.item()


def analyze_temporal_continuity_single_video(
    prompt: str,
    output_dir: str,
    num_frames: int = 49,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int = 42,
    model_path: str = None,
    device: str = "cuda:0",
    use_x0_space: bool = True,
    enable_half_frame: bool = False
):
    """
    分析单个视频生成过程中的时序连续性
    
    Args:
        prompt: 文本提示
        output_dir: 输出目录
        num_frames: 帧数
        num_inference_steps: 推理步数
        guidance_scale: CFG引导强度
        seed: 随机种子
        model_path: 模型路径
        device: 设备
        use_x0_space: True使用x0空间，False使用epsilon空间
        enable_half_frame: 是否启用帧数减半
    """
    import wan
    from omegaconf import OmegaConf
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"Temporal Continuity Analysis")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Frames: {num_frames}")
    print(f"Steps: {num_inference_steps}")
    print(f"Space: {'x0' if use_x0_space else 'epsilon'}")
    print(f"Half-frame: {enable_half_frame}")
    print(f"{'='*60}\n")
    
    # 加载模型
    print("Loading model...")
    config_path = os.path.join(model_path, "config.yaml")
    if not os.path.exists(config_path):
        config_path = "Wan2.2/configs/t2v_A14B.yaml"
    
    cfg = OmegaConf.load(config_path)
    device_id = int(device.split(':')[1]) if ':' in device else 0
    
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=model_path,
        device_id=device_id,
        rank=device_id,
        t5_fsdp=False
    )
    
    print("Model loaded.\n")
    
    # 存储每步的指标
    step_indices = []
    timesteps_list = []
    normalized_l2_distances = []  # 每步的平均归一化L2距离
    cosine_similarities = []  # 每步的平均余弦相似度
    
    # Hook函数：捕获每步的latent和预测
    step_data = {'step': 0}
    
    def capture_step_hook(module, input_data, output_data):
        """捕获每步的x_t和eps_pred"""
        # 这个hook需要根据实际的模型结构调整
        # 暂时使用简化版本
        pass
    
    # 由于直接hook比较复杂，我们修改generate函数来记录数据
    # 这里我们创建一个修改版的生成函数
    
    print("Generating video with continuity tracking...")
    
    # 调用generate并记录中间结果
    # 注意：需要修改wan.text2video.py来支持返回中间latent
    # 这里我们假设已经修改，或者使用调试模式
    
    video, timing_info = wan_t2v.generate(
        prompt=prompt,
        negative_prompt="",
        height=480,
        width=832,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        enable_half_frame_generation=enable_half_frame,
        enable_debug=True,  # 启用调试模式以获取中间结果
        debug_output_dir=output_dir
    )
    
    print("\nVideo generation completed.")
    print("Note: This script requires modifications to text2video.py to capture intermediate latents.")
    print("Please add latent recording functionality to the generate() method.\n")
    
    # 如果有调试输出，读取并分析
    debug_latents_dir = os.path.join(output_dir, "debug_latents")
    if os.path.exists(debug_latents_dir):
        print("Analyzing saved latents...")
        analyze_saved_latents(debug_latents_dir, output_dir, use_x0_space)
    else:
        print("⚠️  No debug latents found. Please enable latent saving in text2video.py")
        print("    Add code to save x_t and eps_pred at each step.")


def analyze_saved_latents(debug_dir: str, output_dir: str, use_x0_space: bool = True):
    """
    分析保存的latent文件
    
    Args:
        debug_dir: 包含保存的latent的目录
        output_dir: 输出目录
        use_x0_space: 使用x0空间还是epsilon空间
    """
    # 查找所有保存的latent文件
    latent_files = sorted([f for f in os.listdir(debug_dir) if f.startswith('latent_step_')])
    
    if not latent_files:
        print("No latent files found.")
        return
    
    print(f"Found {len(latent_files)} latent files.")
    
    step_indices = []
    normalized_l2_distances = []
    cosine_similarities = []
    
    for latent_file in tqdm(latent_files, desc="Analyzing latents"):
        # 加载latent
        latent_path = os.path.join(debug_dir, latent_file)
        data = torch.load(latent_path)
        
        step = data['step']
        latent = data['x0_pred'] if use_x0_space else data['eps_pred']
        
        # latent shape: [B, F, C, H, W]
        B, F, C, H, W = latent.shape
        
        if F < 2:
            continue  # 需要至少2帧
        
        # 计算相邻帧的指标
        l2_distances = []
        cos_similarities = []
        
        for f in range(F - 1):
            latent_f = latent[0, f]  # [C, H, W]
            latent_f_plus_1 = latent[0, f + 1]  # [C, H, W]
            
            # 计算归一化L2距离
            l2_dist = compute_normalized_l2_distance(latent_f, latent_f_plus_1)
            l2_distances.append(l2_dist)
            
            # 计算余弦相似度
            cos_sim = compute_cosine_similarity(latent_f, latent_f_plus_1)
            cos_similarities.append(cos_sim)
        
        # 取平均
        avg_l2 = np.mean(l2_distances)
        avg_cos = np.mean(cos_similarities)
        
        step_indices.append(step)
        normalized_l2_distances.append(avg_l2)
        cosine_similarities.append(avg_cos)
    
    # 绘图
    plot_continuity_metrics(
        step_indices,
        normalized_l2_distances,
        cosine_similarities,
        output_dir,
        use_x0_space
    )


def plot_continuity_metrics(
    steps: list,
    l2_distances: list,
    cosine_sims: list,
    output_dir: str,
    use_x0_space: bool
):
    """
    绘制时序连续性指标
    
    Args:
        steps: 步数列表
        l2_distances: 归一化L2距离列表
        cosine_sims: 余弦相似度列表
        output_dir: 输出目录
        use_x0_space: 是否使用x0空间
    """
    space_name = "x̂₀" if use_x0_space else "ε"
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子图1: 归一化L2距离
    ax1.plot(steps, l2_distances, 'b-o', linewidth=2.5, markersize=6, label=f'Normalized L2 Distance')
    ax1.set_xlabel('Denoising Step', fontsize=13)
    ax1.set_ylabel('Distance', fontsize=13)
    ax1.set_title(f'Temporal Continuity: Normalized L2 Distance in {space_name} Space', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=11)
    
    # 添加说明文字
    ax1.text(0.02, 0.98, 'Lower = More Similar', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 子图2: 余弦相似度
    ax2.plot(steps, cosine_sims, 'r-s', linewidth=2.5, markersize=6, label=f'Cosine Similarity')
    ax2.set_xlabel('Denoising Step', fontsize=13)
    ax2.set_ylabel('Similarity', fontsize=13)
    ax2.set_title(f'Temporal Continuity: Cosine Similarity in {space_name} Space', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=11)
    ax2.set_ylim([0, 1.05])
    
    # 添加说明文字
    ax2.text(0.02, 0.02, 'Higher = More Similar', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, f"temporal_continuity_{space_name.replace('̂', 'hat')}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Continuity plot saved to: {output_path}")
    
    # 保存数据
    data_path = os.path.join(output_dir, f"temporal_continuity_data_{space_name.replace('̂', 'hat')}.npz")
    np.savez(data_path, 
             steps=steps, 
             l2_distances=l2_distances, 
             cosine_similarities=cosine_sims)
    print(f"✅ Data saved to: {data_path}")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"Temporal Continuity Statistics ({space_name} space)")
    print(f"{'='*60}")
    print(f"Normalized L2 Distance:")
    print(f"  Mean: {np.mean(l2_distances):.6f}")
    print(f"  Std:  {np.std(l2_distances):.6f}")
    print(f"  Min:  {np.min(l2_distances):.6f} (step {steps[np.argmin(l2_distances)]})")
    print(f"  Max:  {np.max(l2_distances):.6f} (step {steps[np.argmax(l2_distances)]})")
    print(f"\nCosine Similarity:")
    print(f"  Mean: {np.mean(cosine_sims):.6f}")
    print(f"  Std:  {np.std(cosine_sims):.6f}")
    print(f"  Min:  {np.min(cosine_sims):.6f} (step {steps[np.argmin(cosine_sims)]})")
    print(f"  Max:  {np.max(cosine_sims):.6f} (step {steps[np.argmax(cosine_sims)]})")
    print(f"{'='*60}\n")


def create_demo_plot():
    """
    创建一个演示图表（使用模拟数据）
    用于展示分析结果的格式
    """
    print("Creating demo plot with simulated data...")
    
    # 模拟数据：早期步骤相邻帧差异大，后期趋于稳定
    steps = list(range(1, 21))
    
    # 归一化L2距离：早期高（不连续），后期低（连续）
    l2_distances = [
        0.45, 0.42, 0.38, 0.35, 0.30, 
        0.25, 0.22, 0.18, 0.15, 0.12,
        0.10, 0.08, 0.07, 0.06, 0.05,
        0.04, 0.04, 0.03, 0.03, 0.03
    ]
    
    # 余弦相似度：早期低，后期高（趋近1）
    cosine_sims = [
        0.65, 0.70, 0.74, 0.78, 0.82,
        0.85, 0.88, 0.90, 0.92, 0.94,
        0.95, 0.96, 0.97, 0.975, 0.98,
        0.985, 0.988, 0.990, 0.992, 0.994
    ]
    
    output_dir = "./demo_continuity_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_continuity_metrics(steps, l2_distances, cosine_sims, output_dir, use_x0_space=True)
    
    print(f"\n✅ Demo plot created in: {output_dir}")
    print("This demonstrates the expected pattern:")
    print("  - Early steps: High L2 distance, low cosine similarity (frames differ)")
    print("  - Later steps: Low L2 distance, high cosine similarity (frames converge)")


def main():
    parser = argparse.ArgumentParser(description="Temporal Continuity Analysis in Latent Space")
    
    parser.add_argument("--prompt", type=str, 
                        default="A young woman in a red jacket is walking across a busy crosswalk in a modern city at night.",
                        help="Text prompt for generation")
    parser.add_argument("--output_dir", type=str, default="./continuity_analysis",
                        help="Output directory")
    parser.add_argument("--num_frames", type=int, default=49,
                        help="Number of frames")
    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device")
    parser.add_argument("--use_epsilon", action="store_true",
                        help="Use epsilon space instead of x0 space")
    parser.add_argument("--enable_half_frame", action="store_true",
                        help="Enable half-frame generation")
    parser.add_argument("--demo", action="store_true",
                        help="Create demo plot with simulated data")
    
    args = parser.parse_args()
    
    if args.demo:
        create_demo_plot()
    else:
        if args.model_path is None:
            print("Error: --model_path is required (unless using --demo)")
            return
        
        analyze_temporal_continuity_single_video(
            prompt=args.prompt,
            output_dir=args.output_dir,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            model_path=args.model_path,
            device=args.device,
            use_x0_space=not args.use_epsilon,
            enable_half_frame=args.enable_half_frame
        )


if __name__ == "__main__":
    main()

