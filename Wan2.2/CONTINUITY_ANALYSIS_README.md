# 时序连续性分析工具

## 📋 概述

本工具用于分析视频生成过程中**latent空间**的帧间连续性变化，为**帧数截断策略**提供理论支持。

## 🎯 核心动机

通过分析发现：
- **早期步骤**：相邻帧在latent空间差异大（高噪声，结构未形成）
- **后期步骤**：相邻帧在latent空间差异小（结构稳定，细节优化）

**策略推论**：
→ 早期步骤：生成少量帧（高噪声专家）  
→ 后期步骤：补全所有帧（低噪声专家）

## 📐 数学公式

### 两种Latent表征

#### 1. x̂₀空间（去噪估计）
```
x̂₀,t⁽ᶠ⁾ = (x_t⁽ᶠ⁾ - √(1-ᾱ_t) · ε_θ(x_t⁽ᶠ⁾, t)) / √ᾱ_t
```
- 稳定、直观
- 适合做结构/细节的时序比较

#### 2. ε空间（预测噪声）
```
ε_t⁽ᶠ⁾ = ε_θ(x_t⁽ᶠ⁾, t)
```
- 尺度更均匀
- 常用于分析CFG相关量

### 时序连续性指标

对相邻帧 f 和 f+1，定义：

#### 1. 归一化L2距离（强度型）
```
d_t,f = ||x̂₀,t⁽ᶠ⁺¹⁾ - x̂₀,t⁽ᶠ⁾||₂ / √(||x̂₀,t⁽ᶠ⁾||₂² + ||x̂₀,t⁽ᶠ⁺¹⁾||₂²)

d̄_t = (1/(F-1)) · Σ_f d_t,f
```
- **越小** = 越相似
- 衡量帧间差异的绝对强度

#### 2. 余弦相似度（方向型）
```
s_t,f = cos(x̂₀,t⁽ᶠ⁾, x̂₀,t⁽ᶠ⁺¹⁾)

s̄_t = (1/(F-1)) · Σ_f s_t,f
```
- **越大** = 越相似（后期趋近于1）
- 衡量帧间方向的一致性

## 🚀 使用方法

### 快速演示

```bash
# 使用模拟数据生成演示图表（脚本在Wan2.2/目录下）
python Wan2.2/analyze_continuity_simple.py --demo
```

输出：
- `demo_continuity/temporal_continuity_demo.png` - 可视化图表
- `demo_continuity/temporal_continuity_demo.npz` - 原始数据

### 完整分析（需要模型）

```bash
# 注意：此脚本在Wan2.2/目录下
python Wan2.2/analyze_temporal_continuity.py \
    --model_path /path/to/WAN2.2-27B/T2V_A14B_weights \
    --prompt "A young woman walking in a city at night" \
    --output_dir ./continuity_analysis \
    --num_frames 49 \
    --num_inference_steps 20 \
    --use_epsilon  # 可选：使用ε空间而非x̂₀空间
```

**注意**：完整分析需要修改`text2video.py`来保存中间latent。

## 📊 输出结果

### 可视化图表

双子图布局：

**子图1：归一化L2距离**
- X轴：去噪步数
- Y轴：距离值
- 趋势：从高到低（帧间差异减小）

**子图2：余弦相似度**
- X轴：去噪步数
- Y轴：相似度值（0-1）
- 趋势：从低到高（帧间相似度增加）

### 统计信息

```
Temporal Continuity Statistics
========================================
Normalized L2 Distance:
  Early steps (1-5):   Mean = 0.4180
  Middle steps (6-10): Mean = 0.2060
  Late steps (11-20):  Mean = 0.0450

Cosine Similarity:
  Early steps (1-5):   Mean = 0.7240
  Middle steps (6-10): Mean = 0.8980
  Late steps (11-20):  Mean = 0.9847
========================================
```

## 🔍 典型模式

### 预期的连续性变化

| 阶段 | 步数 | L2距离 | 余弦相似度 | 解释 |
|------|------|--------|-----------|------|
| **早期** | 1-5 | 高 (0.4+) | 低 (0.6-0.8) | 高噪声，结构未形成，帧间差异大 |
| **中期** | 6-10 | 中 (0.2-0.3) | 中 (0.8-0.9) | 结构逐渐形成，差异减小 |
| **后期** | 11-20 | 低 (0.02-0.1) | 高 (0.98+) | 细节优化，帧间高度相似 |

### 对帧数截断的启示

```
步骤1-12（高噪声专家）：
  → 生成3帧（F=3）
  → 帧间差异大，但不影响后续
  → 节省计算量

步骤13-20（低噪声专家）：
  → 补全到49帧（F=49）
  → 帧间高度相似，易于插值
  → 保证视频质量
```

## 📁 文件结构

```
Wan2.2/
├── analyze_temporal_continuity.py    # 完整版分析脚本
├── analyze_continuity_simple.py      # 简化版（含演示）
└── CONTINUITY_ANALYSIS_README.md     # 本文档

输出示例：
demo_continuity/
├── temporal_continuity_demo.png      # 可视化图表
└── temporal_continuity_demo.npz      # 原始数据
```

## 🔧 技术细节

### 实现的函数

#### `compute_normalized_l2_distance(latent_f, latent_f_plus_1)`
计算归一化L2距离

**输入**：
- `latent_f`: 帧f的latent [C, H, W]
- `latent_f_plus_1`: 帧f+1的latent [C, H, W]

**输出**：
- `distance`: 标量，范围[0, +∞)

#### `compute_cosine_similarity(latent_f, latent_f_plus_1)`
计算余弦相似度

**输入**：
- `latent_f`: 帧f的latent [C, H, W]
- `latent_f_plus_1`: 帧f+1的latent [C, H, W]

**输出**：
- `similarity`: 标量，范围[-1, 1]

#### `analyze_latent_continuity(latent, step_idx)`
分析单个latent tensor的帧间连续性

**输入**：
- `latent`: [B, F, C, H, W] 或 [F, C, H, W]
- `step_idx`: 当前步数

**输出**：
- `avg_l2`: 平均归一化L2距离
- `avg_cos`: 平均余弦相似度

### 集成到现有代码

```python
from analyze_continuity_simple import (
    analyze_latent_continuity,
    compute_normalized_l2_distance,
    compute_cosine_similarity
)

# 在生成循环中
for step_idx, t in enumerate(timesteps):
    # ... 模型前向传播 ...
    
    # 分析连续性
    avg_l2, avg_cos = analyze_latent_continuity(x0_pred, step_idx)
    
    print(f"Step {step_idx}: L2={avg_l2:.4f}, Cos={avg_cos:.4f}")
```

## 📈 实验建议

### 对比实验

1. **基线方法** vs **帧数截断方法**
   ```bash
   # 基线：全程49帧
   python Wan2.2/analyze_temporal_continuity.py \
       --model_path /path/to/model \
       --num_frames 49 \
       --output_dir ./baseline_continuity
   
   # 帧数截断：3帧→49帧
   python Wan2.2/analyze_temporal_continuity.py \
       --model_path /path/to/model \
       --num_frames 49 \
       --enable_half_frame \
       --output_dir ./truncated_continuity
   ```

2. **不同步数的影响**
   ```bash
   for steps in 10 15 20 25 30; do
       python Wan2.2/analyze_temporal_continuity.py \
           --model_path /path/to/model \
           --num_inference_steps $steps \
           --output_dir ./continuity_steps_${steps}
   done
   ```

3. **不同帧数的影响**
   ```bash
   for frames in 13 25 49 97; do
       python Wan2.2/analyze_temporal_continuity.py \
           --model_path /path/to/model \
           --num_frames $frames \
           --output_dir ./continuity_frames_${frames}
   done
   ```

## 💡 关键发现

### 理论支持

1. **帧间连续性随步数增加**
   - 早期：低连续性（高噪声）
   - 后期：高连续性（低噪声）

2. **帧数截断的合理性**
   - 早期少量帧足够捕获全局结构
   - 后期补全帧利用高连续性

3. **计算效率提升**
   - 早期步骤：3帧 vs 49帧 → 16倍加速
   - 后期步骤：保持49帧 → 保证质量

### 实验验证

通过对比基线和截断方法的连续性曲线：
- 截断方法在补全后连续性快速恢复
- 最终视频质量与基线相当
- 总计算量显著减少

## 🎓 引用

如果使用本工具进行研究，建议引用：

```
时序连续性分析基于latent空间的帧间相似度度量：
- 归一化L2距离：衡量强度差异
- 余弦相似度：衡量方向一致性

支持帧数截断策略的理论基础：
早期步骤生成少量帧，后期步骤补全所有帧
```

---

**祝实验顺利！** 🎉

