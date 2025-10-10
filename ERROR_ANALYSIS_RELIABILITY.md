# Error Analysis 可靠性分析

## 🎯 核心问题

1. **相邻两步的CFG差值变化分析可靠吗？**
2. **有条件/无条件输出会随时间步变化量级吗？**
3. **Noise如何根据有条件/无条件输出更新？**

---

## 📊 问题1：相邻两步CFG差值变化的可靠性

### **当前实现**

```python
# Wan2.2/wan/text2video.py (第1357-1361行)
cfg_diff_changes = []
for i in range(1, len(cfg_diffs)):
    change = abs(cfg_diffs[i] - cfg_diffs[i-1])
    cfg_diff_changes.append(change)
```

其中：
```python
cfg_diff = noise_pred_cond - noise_pred_uncond
```

### **⚠️ 可靠性问题**

#### **问题：不同时间步的量级不可比**

在扩散模型中，noise prediction的量级会随时间步变化：

**早期步骤（高噪声，t ≈ 1000）**：
- `noise_pred` 的量级较大（因为噪声水平高）
- `|noise_pred_cond - noise_pred_uncond|` 也较大

**后期步骤（低噪声，t ≈ 0）**：
- `noise_pred` 的量级较小（因为噪声水平低）
- `|noise_pred_cond - noise_pred_uncond|` 也较小

**结果**：直接比较不同时间步的绝对差值是**不公平的**！

---

### **✅ 改进方案：归一化CFG差值**

#### **方案1：相对于噪声水平归一化**

```python
# 使用sigma_t归一化
sigma_t = scheduler.sigmas[step_index]
cfg_diff_normalized = cfg_diff / (sigma_t + 1e-8)
```

**原理**：
- `sigma_t` 表示当前时间步的噪声水平
- 早期 `sigma_t` 大，后期 `sigma_t` 小
- 归一化后的CFG差值可以跨时间步比较

#### **方案2：相对于无条件输出归一化**

```python
# 相对误差
cfg_diff_relative = cfg_diff / (torch.abs(noise_pred_uncond) + 1e-8)
cfg_diff_relative_mean = cfg_diff_relative.mean().item()
```

**原理**：
- 以无条件输出的量级作为基准
- 衡量条件输出相对于无条件输出的偏离程度

#### **方案3：标准化（Z-score）**

```python
# 在整个生成过程中标准化
cfg_diffs_array = np.array([d['cfg_diff_mean'] for d in error_history])
cfg_diffs_normalized = (cfg_diffs_array - cfg_diffs_array.mean()) / (cfg_diffs_array.std() + 1e-8)
```

**原理**：
- 消除量级差异，只关注相对变化
- 适合分析趋势和异常值

---

## 📐 问题2：有条件/无条件输出的量级变化

### **理论分析**

在Flow Matching / Rectified Flow框架中：

#### **前向过程（加噪）**

```
x_t = (1 - t) * x_0 + t * ε
```

其中：
- `t ∈ [0, 1]` 是归一化时间步
- `x_0` 是干净数据
- `ε` 是纯噪声

#### **模型预测**

模型预测的是**velocity**（速度场）：

```
v_θ(x_t, t) = x_0 - ε
```

或者在实际实现中，模型预测noise：

```
ε_θ(x_t, t) ≈ ε
```

#### **量级随时间步的变化**

**早期（t ≈ 1，高噪声）**：
```
x_t ≈ ε  （几乎全是噪声）
ε_θ(x_t, t) 的量级 ≈ ||ε|| （大）
```

**后期（t ≈ 0，低噪声）**：
```
x_t ≈ x_0  （几乎是干净数据）
ε_θ(x_t, t) 的量级 ≈ 0 （小）
```

### **实际验证**

从代码中可以看到，scheduler使用sigma来控制噪声水平：

```python
# Wan2.2/wan/utils/fm_solvers_unipc.py (第317-323行)
sigma = self.sigmas[self.step_index]
alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

if self.config.prediction_type == "flow_prediction":
    sigma_t = self.sigmas[self.step_index]
    x0_pred = sample - sigma_t * model_output  # ✅ 使用sigma_t缩放
```

**结论**：
- ✅ **有条件和无条件输出的量级确实会随时间步变化**
- ✅ **早期量级大，后期量级小**
- ✅ **这是扩散模型的固有特性**

---

## 🔄 问题3：Noise如何根据输出更新

### **完整的更新流程**

#### **步骤1：模型预测（CFG引导）**

```python
# text2video.py (第574-586行)
noise_pred_cond = model(latent, t, context=prompt)      # 条件预测
noise_pred_uncond = model(latent, t, context=empty)     # 无条件预测

# CFG引导
noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
```

**数学形式**：
```
ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
         = (1 - w) * ε_uncond + w * ε_cond
```

其中 `w = guide_scale`（通常是3.0-7.5）

#### **步骤2：Scheduler更新latent**

```python
# text2video.py (第588-593行)
temp_x0 = sample_scheduler.step(
    noise_pred.unsqueeze(0),
    t,
    latents[0].unsqueeze(0),
    return_dict=False,
    generator=seed_g
)[0]
```

**Scheduler内部逻辑**：

```python
# fm_solvers_unipc.py (第697-730行)
def step(self, model_output, timestep, sample):
    # 1. 转换模型输出
    model_output_convert = self.convert_model_output(model_output, sample=sample)
    
    # 2. 多步更新（UniPC算法）
    prev_sample = self.multistep_uni_p_bh_update(
        model_output=model_output,
        sample=sample,
        order=self.this_order,
    )
    
    return prev_sample
```

**convert_model_output的关键**：

```python
# fm_solvers_unipc.py (第317-323行)
sigma = self.sigmas[self.step_index]
alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

if self.config.prediction_type == "flow_prediction":
    sigma_t = self.sigmas[self.step_index]
    x0_pred = sample - sigma_t * model_output  # ✅ 关键公式
```

**数学推导**：

在Flow Matching中：
```
x_t = (1 - σ_t) * x_0 + σ_t * ε

模型预测: v_θ = x_0 - ε

因此:
x_0 = x_t - σ_t * v_θ
```

#### **步骤3：更新latent**

```python
latents[0] = temp_x0.squeeze(0)
```

---

### **完整数学流程**

给定当前latent `x_t` 和时间步 `t`：

**1. 条件预测**：
```
ε_cond = Model(x_t, t, prompt)
```

**2. 无条件预测**：
```
ε_uncond = Model(x_t, t, empty)
```

**3. CFG引导**：
```
ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
```

**4. 估计x_0**：
```
x̂_0 = x_t - σ_t * ε_guided
```

**5. 更新到下一步**（UniPC多步算法）：
```
x_{t-1} = UniPC_Update(x̂_0, x_t, t)
```

---

## 🔧 改进建议

### **1. 归一化CFG差值**

修改error analysis代码：

```python
def _call_model_with_error_analysis(self, model, latent_model_input, timestep, model_kwargs, step_idx, record_error=True):
    # ... 现有代码 ...
    
    if record_error:
        # 获取当前sigma
        sigma_t = self.sample_scheduler.sigmas[self.sample_scheduler.step_index]
        
        # 计算归一化的CFG差值
        cfg_diff = current_output - noise_pred_uncond
        cfg_diff_normalized = cfg_diff / (sigma_t + 1e-8)  # ✅ 归一化
        
        # 计算相对CFG差值
        cfg_diff_relative = cfg_diff / (torch.abs(noise_pred_uncond) + 1e-8)
        
        error_data = {
            # ... 现有字段 ...
            'cfg_diff_mean': cfg_diff.mean().item(),
            'cfg_diff_std': cfg_diff.std().item(),
            'cfg_diff_normalized_mean': cfg_diff_normalized.mean().item(),  # ✅ 新增
            'cfg_diff_normalized_std': cfg_diff_normalized.std().item(),    # ✅ 新增
            'cfg_diff_relative_mean': cfg_diff_relative.mean().item(),      # ✅ 新增
            'cfg_diff_relative_std': cfg_diff_relative.std().item(),        # ✅ 新增
            'sigma_t': sigma_t.item(),  # ✅ 记录sigma
        }
```

### **2. 改进可视化**

```python
def _create_error_visualization(self):
    # 提取归一化数据
    cfg_diffs_normalized = [data['cfg_diff_normalized_mean'] for data in self.error_history]
    sigmas = [data['sigma_t'] for data in self.error_history]
    
    # 创建3个子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # 子图1: 原始CFG差值（会随时间步变化）
    ax1.plot(steps, cfg_diffs, label='Raw CFG Difference')
    ax1.set_title('Raw CFG Difference (Time-dependent Scale)')
    
    # 子图2: 归一化CFG差值（消除时间步影响）
    ax2.plot(steps, cfg_diffs_normalized, label='Normalized CFG Difference')
    ax2.set_title('Normalized CFG Difference (Scale-invariant)')
    
    # 子图3: Sigma变化（噪声水平）
    ax3.plot(steps, sigmas, label='Noise Level (σ_t)')
    ax3.set_title('Noise Schedule')
```

### **3. 相邻步骤变化分析**

```python
# 使用归一化值计算变化
cfg_diff_normalized_changes = []
for i in range(1, len(cfg_diffs_normalized)):
    change = abs(cfg_diffs_normalized[i] - cfg_diffs_normalized[i-1])
    cfg_diff_normalized_changes.append(change)

# 这样的变化才是可比的！
```

---

## 📊 总结

### **当前实现的问题**

| 问题 | 影响 | 严重性 |
|------|------|--------|
| 未归一化CFG差值 | 不同时间步不可比 | ⚠️ 中等 |
| 直接比较绝对值 | 早期步骤主导分析 | ⚠️ 中等 |
| 忽略噪声水平 | 误导性结论 | ⚠️ 中等 |

### **改进后的优势**

| 改进 | 优势 |
|------|------|
| ✅ Sigma归一化 | 消除时间步量级差异 |
| ✅ 相对误差 | 关注相对变化而非绝对值 |
| ✅ 记录sigma | 理解噪声水平的影响 |
| ✅ 多维度分析 | 更全面的误差理解 |

### **核心结论**

1. **✅ Error Analysis是真实计算的**（每步都调用模型）
2. **⚠️ 相邻步骤的CFG差值变化需要归一化才可靠**
3. **✅ 有条件/无条件输出量级确实随时间步变化**
4. **✅ Noise更新通过CFG引导的预测和scheduler完成**

### **推荐做法**

对于可靠的误差分析：
1. 使用**归一化CFG差值**（除以sigma_t）
2. 计算**相对误差**（相对于无条件输出）
3. 分析**归一化后的相邻步骤变化**
4. 同时可视化**原始值和归一化值**以理解完整图景

---

**参考文献**：
- Flow Matching: [Lipman et al., 2023]
- Rectified Flow: [Liu et al., 2023]
- Classifier-Free Guidance: [Ho & Salimans, 2022]

