# WAN2.2-T2V-A14B (27B MOE) 推理计算过程详解

## 🧮 **核心计算公式**

### **1. 专家路由函数**
```math
Expert(t) = \begin{cases} 
E_{high} & \text{if } t \geq \tau \cdot T_{max} \\
E_{low} & \text{if } t < \tau \cdot T_{max}
\end{cases}
```
其中：
- $t$: 当前扩散时间步 ∈ [0, 1000]
- $\tau = 0.875$: 专家切换边界系数
- $T_{max} = 1000$: 最大训练时间步
- $E_{high}$: 高噪声专家模型 (14B参数)
- $E_{low}$: 低噪声专家模型 (14B参数)

### **2. 扩散噪声预测**
```math
\epsilon_\theta(x_t, t, c) = Expert(t)(x_t, t, c)
```

### **3. 分类器无关引导 (CFG)**
```math
\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s(t) \cdot [\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)]
```
其中：
```math
s(t) = \begin{cases} 
4.0 & \text{if } t \geq 875 \text{ (高噪声阶段)} \\
3.0 & \text{if } t < 875 \text{ (低噪声阶段)}
\end{cases}
```

### **4. Flow Matching采样**
```math
x_{t-1} = x_t - \Delta t \cdot \tilde{\epsilon}_\theta(x_t, t, c)
```
其中时间步长：
```math
\Delta t = \frac{\sigma_{shift} \cdot (1-\alpha_t)}{1 + (\sigma_{shift}-1) \cdot (1-\alpha_t)}
```

## 🔢 **Transformer计算详解**

### **5. 多头自注意力机制**
对于每个专家模型 $E \in \{E_{high}, E_{low}\}$：
```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
```
其中每个注意力头：
```math
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
```
```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

参数维度：
- $h = 40$ (注意力头数)
- $d_{model} = 5120$ (模型维度)
- $d_k = d_v = d_{model}/h = 128$ (每头维度)

### **6. RoPE位置编码**
```math
\text{RoPE}(x, pos) = x \odot \cos(pos \cdot \theta) + \text{rotate}(x) \odot \sin(pos \cdot \theta)
```
其中频率向量：
```math
\theta_i = 10000^{-2i/d}, \quad i = 0, 1, ..., d/2-1
```

### **7. Feed-Forward网络**
```math
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
```
参数维度：
- $W_1 \in \mathbb{R}^{5120 \times 13824}$
- $W_2 \in \mathbb{R}^{13824 \times 5120}$

## 📐 **多卡并行计算**

### **8. FSDP权重分片**
对于P张GPU，每张GPU存储参数：
```math
W^{(p)} = \text{shard}(W, p), \quad p = 0, 1, ..., P-1
```
其中总参数量：
```math
|W_{total}| = |W_{high}| + |W_{low}| = 14B + 14B = 27B
```
每张GPU参数量：
```math
|W^{(p)}| = \frac{27B}{P}
```

### **9. Ulysses序列并行**
将序列长度 $L$ 分割到 $P$ 张GPU：
```math
L^{(p)} = \frac{L}{P}, \quad p = 0, 1, ..., P-1
```
All-to-All通信模式：
```math
\text{AllToAll}(X) = \text{Gather}(\text{Scatter}(X, \text{dim}=2), \text{dim}=1)
```

### **10. 分布式注意力计算**
```math
\text{Attention}^{(p)}(Q^{(p)}, K^{all}, V^{all}) = \text{softmax}\left(\frac{Q^{(p)}(K^{all})^T}{\sqrt{d_k}}\right)V^{all}
```
其中 $K^{all}, V^{all}$ 通过All-to-All聚合所有GPU的K,V。

## 🧮 **计算复杂度分析**

### **11. 单层Transformer计算量**
对于序列长度 $L$，模型维度 $d$：

**自注意力计算**：
```math
\text{FLOPs}_{attn} = 4Ld^2 + 2L^2d
```

**Feed-Forward计算**：
```math
\text{FLOPs}_{ffn} = 8Ld^2
```

**单层总计算量**：
```math
\text{FLOPs}_{layer} = 12Ld^2 + 2L^2d
```

### **12. 完整模型计算量**
对于40层Transformer：
```math
\text{FLOPs}_{model} = 40 \times (12Ld^2 + 2L^2d) = 480Ld^2 + 80L^2d
```

具体数值 ($L = 75600$, $d = 5120$)：
```math
\text{FLOPs}_{model} = 480 \times 75600 \times 5120^2 + 80 \times 75600^2 \times 5120 \approx 9.5 \times 10^{16}
```

### **13. 内存占用计算**
**模型参数内存** (bfloat16)：
```math
\text{Memory}_{params} = 14 \times 10^9 \times 2 \text{ bytes} = 26 \text{ GB}
```

**激活内存**：
```math
\text{Memory}_{act} = L \times d \times \text{layers} \times 2 \text{ bytes} \approx 30 \text{ GB}
```

**KV缓存内存**：
```math
\text{Memory}_{kv} = 2 \times L \times d \times \text{layers} \times 2 \text{ bytes} \approx 60 \text{ GB}
```

## 📊 **完整推理算法**

### **14. 主推理循环**
```math
\begin{align}
&\text{输入: } \text{prompt } p, \text{ 采样步数 } T \\
&\text{初始化: } x_T \sim \mathcal{N}(0, I), \text{ } c = \text{T5}(p) \\
&\text{For } t = T, T-1, ..., 1: \\
&\quad \text{1. 专家选择: } E_t = \begin{cases} E_{high} & \text{if } t \geq 875 \\ E_{low} & \text{if } t < 875 \end{cases} \\
&\quad \text{2. 噪声预测: } \epsilon_{cond} = E_t(x_t, t, c), \text{ } \epsilon_{uncond} = E_t(x_t, t, \emptyset) \\
&\quad \text{3. CFG引导: } \tilde{\epsilon} = \epsilon_{uncond} + s(t) \cdot (\epsilon_{cond} - \epsilon_{uncond}) \\
&\quad \text{4. 采样更新: } x_{t-1} = x_t - \Delta t \cdot \tilde{\epsilon} \\
&\text{输出: } \text{video} = \text{VAE}_{decode}(x_0)
\end{align}
```

### **15. 数值参数总结**
```math
\begin{array}{|l|c|}
\hline
\text{参数} & \text{数值} \\
\hline
\text{总参数量} & 27 \times 10^9 \\
\text{激活参数量} & 14 \times 10^9 \\
\text{模型维度 } d & 5120 \\
\text{FFN维度} & 13824 \\
\text{注意力头数 } h & 40 \\
\text{Transformer层数} & 40 \\
\text{序列长度 } L & 75600 \\
\text{专家切换边界 } \tau & 0.875 \\
\text{CFG缩放(高噪声)} & 4.0 \\
\text{CFG缩放(低噪声)} & 3.0 \\
\text{采样步数} & 40 \\
\text{噪声调度偏移} & 12.0 \\
\hline
\end{array}
```

### **16. 计算复杂度汇总**
```math
\begin{align}
\text{单步FLOPs} &= 480Ld^2 + 80L^2d \\
&= 480 \times 75600 \times 5120^2 + 80 \times 75600^2 \times 5120 \\
&\approx 9.5 \times 10^{16} \text{ FLOPs} \\
\\
\text{总推理FLOPs} &= 40 \times 9.5 \times 10^{16} = 3.8 \times 10^{18} \text{ FLOPs} \\
\\
\text{单卡内存需求} &= 26 + 30 + 60 = 116 \text{ GB} \\
\text{多卡内存需求} &= \frac{116}{P} + \text{通信开销} \text{ GB}
\end{align}
```

---

**WAN2.2-T2V-A14B通过MOE双专家架构，实现了27B参数模型在14B激活参数下的高效推理，是视频生成领域的重要技术突破。**
