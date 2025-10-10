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


def compute_mse_distance(latent_f, latent_f_plus_1):
    """
    计算MSE距离（均方误差）
    
    MSE = mean((x̂₀⁽ᶠ⁺¹⁾ - x̂₀⁽ᶠ⁾)²)
    
    Args:
        latent_f: 帧f的latent [C, H, W]
        latent_f_plus_1: 帧f+1的latent [C, H, W]
    
    Returns:
        mse: MSE标量
    """
    diff = latent_f_plus_1 - latent_f
    mse = torch.mean(diff ** 2)
    
    return mse.item()


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
    from wan.configs import WAN_CONFIGS
    
    # 使用Python配置对象（与generate.py相同的方式）
    cfg = WAN_CONFIGS['t2v-A14B']
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
        input_prompt=prompt,
        size=(832, 480),
        frame_num=num_frames,
        shift=12.0,
        sample_solver='unipc',
        sampling_steps=num_inference_steps,
        guide_scale=guidance_scale,
        n_prompt="",
        seed=seed,
        offload_model=True,
        cfg_truncate_steps=0,  # 禁用CFG截断
        cfg_truncate_high_noise_steps=0,  # 禁用高噪声CFG截断
        enable_half_frame_generation=enable_half_frame,
        enable_debug=True,  # 启用调试模式以获取中间结果
        debug_output_dir=output_dir
    )
    
    print(f"\n📋 分析说明:")
    print(f"   将分析每步传递的实际latent (x_t)")
    print(f"   而不是中间的x0预测")
    print(f"   x_t是真正影响下一步的量")
    
    print("\n✅ Video generation completed.")
    
    # 读取并分析保存的latent
    debug_latents_dir = os.path.join(output_dir, "debug_latents")
    if os.path.exists(debug_latents_dir):
        print(f"\n📊 分析保存的latents...")
        print(f"   Latents目录: {debug_latents_dir}")
        analyze_saved_latents(debug_latents_dir, output_dir, use_x0_space)
    else:
        print(f"\n⚠️  未找到debug latents目录: {debug_latents_dir}")
        print("    请确保enable_debug=True")
        print("    Latents会自动保存到: {output_dir}/debug_latents/")


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
    
    # 打印第一个文件的信息用于调试
    first_file_path = os.path.join(debug_dir, latent_files[0])
    first_data = torch.load(first_file_path)
    print(f"\n📋 Latent文件格式:")
    print(f"   Keys: {list(first_data.keys())}")
    print(f"   x0_pred shape: {first_data['x0_pred'].shape}")
    print(f"   eps_pred shape: {first_data['eps_pred'].shape}")
    print(f"   使用空间: {'x0' if use_x0_space else 'epsilon'}")
    
    # 检查latent的维度顺序
    test_latent = first_data['x_t']  # 使用x_t而不是x0_pred
    print(f"   x_t shape: {first_data['x_t'].shape}")
    print(f"   分析对象: x_t (实际传递的噪声latent)")
    print(f"   测试latent shape: {test_latent.shape}")
    print(f"   测试latent维度: {test_latent.dim()}")
    if test_latent.dim() == 5:
        print(f"   假设维度顺序: [Batch, Channel, Frame, Height, Width]")
    elif test_latent.dim() == 4:
        print(f"   假设维度顺序: [Channel, Frame, Height, Width]")
    print()
    
    for latent_file in tqdm(latent_files, desc="Analyzing latents"):
        # 加载latent
        latent_path = os.path.join(debug_dir, latent_file)
        data = torch.load(latent_path)
        
        step = data['step']
        # 使用x_t（实际传递的latent），而不是x0_pred（中间估计）
        latent = data['x_t']
        
        # 处理维度：WAN格式是 [B, C, F, H, W] 或 [C, F, H, W]
        if latent.dim() == 5:
            # [B, C, F, H, W]，取第一个batch
            B, C, F, H, W = latent.shape
            latent = latent[0]  # 现在是 [C, F, H, W]
        elif latent.dim() == 4:
            # [C, F, H, W]
            C, F, H, W = latent.shape
        else:
            print(f"⚠️  Unexpected latent shape: {latent.shape}")
            continue
        
        if F < 2:
            if step == 1:
                print(f"⚠️  Step {step}: 只有{F}帧，无法计算相邻帧连续性")
            continue  # 需要至少2帧
        
        # 计算相邻帧的指标（注意：这是分析每步内的相邻帧，不是相邻步骤）
        mse_distances = []
        cos_similarities = []
        
        # 第一步时打印详细信息
        if step == 1:
            print(f"\n🔍 Step {step} 详细信息:")
            print(f"   Latent shape after processing: [C={C}, F={F}, H={H}, W={W}]")
            print(f"   将分析 {F-1} 对相邻帧")
            print(f"   帧0 vs 帧1, 帧1 vs 帧2, ..., 帧{F-2} vs 帧{F-1}")
        
        for f in range(F - 1):
            latent_f = latent[:, f, :, :]  # [C, H, W]
            latent_f_plus_1 = latent[:, f + 1, :, :]  # [C, H, W]
            
            # 第一步时打印前两对帧的详细信息
            if step == 1 and f < 2:
                print(f"\n   帧{f} vs 帧{f+1}:")
                print(f"     帧{f} 范围: [{latent_f.min():.4f}, {latent_f.max():.4f}]")
                print(f"     帧{f+1} 范围: [{latent_f_plus_1.min():.4f}, {latent_f_plus_1.max():.4f}]")
                print(f"     帧{f} 均值: {latent_f.mean():.4f}, 标准差: {latent_f.std():.4f}")
                print(f"     帧{f+1} 均值: {latent_f_plus_1.mean():.4f}, 标准差: {latent_f_plus_1.std():.4f}")
            
            # 计算MSE距离
            mse_dist = compute_mse_distance(latent_f, latent_f_plus_1)
            mse_distances.append(mse_dist)
            
            # 第一步时打印计算结果
            if step == 1 and f < 2:
                print(f"     MSE距离: {mse_dist:.6f}")
            
            # 计算余弦相似度
            cos_sim = compute_cosine_similarity(latent_f, latent_f_plus_1)
            cos_similarities.append(cos_sim)
            
            # 第一步时打印计算结果
            if step == 1 and f < 2:
                print(f"     余弦相似度: {cos_sim:.6f}")
        
        # 取平均
        avg_mse = np.mean(mse_distances)
        avg_cos = np.mean(cos_similarities)
        
        step_indices.append(step)
        normalized_l2_distances.append(avg_mse)  # 复用变量名，实际存MSE
        cosine_similarities.append(avg_cos)
    
    # 绘图（注意：normalized_l2_distances变量实际存储的是mse_distances）
    plot_continuity_metrics(
        step_indices,
        normalized_l2_distances,  # 实际是mse_distances
        cosine_similarities,
        output_dir,
        use_x0_space
    )


def plot_continuity_metrics(
    steps: list,
    mse_distances: list,
    cosine_sims: list,
    output_dir: str,
    use_x0_space: bool
):
    """
    绘制时序连续性指标
    
    Args:
        steps: 步数列表
        mse_distances: MSE距离列表
        cosine_sims: 余弦相似度列表
        output_dir: 输出目录
        use_x0_space: 是否使用x0空间
    """
    space_name = "x_t (Noisy Latent)"
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子图1: MSE距离
    ax1.plot(steps, mse_distances, 'b-o', linewidth=2.5, markersize=6, label=f'MSE Distance')
    ax1.set_xlabel('Denoising Step', fontsize=13)
    ax1.set_ylabel('MSE', fontsize=13)
    ax1.set_title(f'Temporal Continuity: MSE Between Adjacent Frames in {space_name}', 
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
    ax2.set_title(f'Temporal Continuity: Cosine Similarity in {space_name}', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=11)
    ax2.set_ylim([-0.1, 1.05])  # 允许负值，因为早期可能是负相关
    
    # 添加说明文字
    ax2.text(0.02, 0.02, 'Higher = More Similar', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, "temporal_continuity_x_t.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Continuity plot saved to: {output_path}")
    
    # 保存数据
    data_path = os.path.join(output_dir, "temporal_continuity_data_x_t.npz")
    np.savez(data_path, 
             steps=steps, 
             mse_distances=mse_distances, 
             cosine_similarities=cosine_sims)
    print(f"✅ Data saved to: {data_path}")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"Temporal Continuity Statistics (x_t - Noisy Latent)")
    print(f"{'='*60}")
    print(f"MSE Between Adjacent Frames:")
    print(f"  Mean: {np.mean(mse_distances):.6f}")
    print(f"  Std:  {np.std(mse_distances):.6f}")
    print(f"  Min:  {np.min(mse_distances):.6f} (step {steps[np.argmin(mse_distances)]})")
    print(f"  Max:  {np.max(mse_distances):.6f} (step {steps[np.argmax(mse_distances)]})")
    print(f"\nCosine Similarity:")
    print(f"  Mean: {np.mean(cosine_sims):.6f}")
    print(f"  Std:  {np.std(cosine_sims):.6f}")
    print(f"  Min:  {np.min(cosine_sims):.6f} (step {steps[np.argmin(cosine_sims)]})")
    print(f"  Max:  {np.max(cosine_sims):.6f} (step {steps[np.argmax(cosine_sims)]})")
    print(f"\n📋 说明:")
    print(f"  - 分析的是每步实际传递的噪声latent (x_t)")
    print(f"  - x_t是真正影响下一步计算的量")
    print(f"  - 预期趋势: MSE从高到低, 余弦相似度从低到高")
    print(f"{'='*60}\n")


def create_demo_plot():
    """
    创建一个演示图表（使用模拟数据）
    用于展示分析结果的格式
    """
    print("Creating demo plot with simulated data...")
    
    # 模拟数据：早期步骤相邻帧差异大，后期趋于稳定
    steps = list(range(1, 21))
    
    # MSE距离：早期高（不连续），后期低（连续）
    mse_distances = [
        0.25, 0.20, 0.17, 0.14, 0.11,
        0.08, 0.06, 0.04, 0.03, 0.02,
        0.012, 0.008, 0.005, 0.004, 0.003,
        0.002, 0.001, 0.001, 0.0005, 0.0005
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
    
    plot_continuity_metrics(steps, mse_distances, cosine_sims, output_dir, use_x0_space=True)
    
    print(f"\n✅ Demo plot created in: {output_dir}")
    print("This demonstrates the expected pattern:")
    print("  - Early steps: High MSE, low cosine similarity (frames differ)")
    print("  - Later steps: Low MSE, high cosine similarity (frames converge)")


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

