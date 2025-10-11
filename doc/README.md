# WAN2.2 项目文档

本文件夹包含WAN2.2模型的完整文档和实验说明。

---

## 📚 文档目录

### **使用指南**

| 文档 | 描述 | 适用人群 |
|------|------|---------|
| [使用说明.md](./使用说明.md) | WAN2.2模型完整使用指南 | 所有用户 |
| [CFG截断加速技术说明.md](./CFG截断加速技术说明.md) | CFG截断优化详解 | 优化研究者 |
| [帧数截断优化说明.md](./帧数截断优化说明.md) | 帧数截断实现详解（含代码位置） | 优化研究者 |
| [WAN2.2模型加载优化分析.md](./WAN2.2模型加载优化分析.md) | 模型加载机制分析 | 系统优化者 |

### **实验研究**

| 文档 | 描述 | 适用人群 |
|------|------|---------|
| [实验说明.md](./实验说明.md) | 完整的实验文档（核心） | 研究者 |
| [FLOW_MATCHING_ANALYSIS.md](./FLOW_MATCHING_ANALYSIS.md) | Flow Matching理论分析 | 算法研究者 |
| [CONTINUITY_ANALYSIS_README.md](./CONTINUITY_ANALYSIS_README.md) | 帧连续性分析文档 | 视频质量研究者 |

### **评估工具**

| 文档 | 描述 | 适用人群 |
|------|------|---------|
| [T2I_EVALUATION_README.md](./T2I_EVALUATION_README.md) | T2I评估完整说明 | 评估研究者 |
| [T2I_EVALUATION_QUICKSTART.md](./T2I_EVALUATION_QUICKSTART.md) | T2I评估快速开始 | 快速验证者 |

---

## 🚀 快速导航

### **我想...**

#### **...使用模型生成视频**
→ 阅读 [使用说明.md](./使用说明.md)

#### **...加速推理**
→ 阅读 [CFG截断加速技术说明.md](./CFG截断加速技术说明.md)
→ 阅读 [帧数截断优化说明.md](./帧数截断优化说明.md)
→ 参考 [使用说明.md](./使用说明.md) 的"推理加速技巧"部分

#### **...了解实验和分析方法**
→ 阅读 [实验说明.md](./实验说明.md)（推荐从这里开始）

#### **...理解Flow Matching算法**
→ 阅读 [FLOW_MATCHING_ANALYSIS.md](./FLOW_MATCHING_ANALYSIS.md)

#### **...分析视频连续性**
→ 阅读 [CONTINUITY_ANALYSIS_README.md](./CONTINUITY_ANALYSIS_README.md)

#### **...评估T2I生成质量**
→ 阅读 [T2I_EVALUATION_QUICKSTART.md](./T2I_EVALUATION_QUICKSTART.md)

---

## 🔬 实验概览

本项目包含三大核心实验：

### **1. 误差分析（Error Analysis）**
- **脚本**: `Wan2.2/wan/text2video.py`
- **目的**: 量化CFG的作用机制
- **启用**: `--enable_debug`

### **2. 方法对比（Method Comparison）**
- **脚本**: `Wan2.2/compare_cfg_baseline.py`
- **目的**: 比较优化方法与基线的质量差异
- **指标**: MSE、PSNR、SSIM

### **3. 帧连续性分析（Temporal Continuity）**
- **脚本**: `Wan2.2/analyze_temporal_continuity.py`
- **目的**: 评估时序一致性
- **分析**: 相邻帧MSE

详细说明请参考 [实验说明.md](./实验说明.md)

---

## 📊 关键发现

### **实验验证的结论**

| 结论 | 证据 | 文档 |
|------|------|------|
| CFG截断质量无损 | MSE≈0.01, PSNR>35dB | [实验说明.md](./实验说明.md) |
| 帧截断保持连续性 | SSIM>0.95 | [实验说明.md](./实验说明.md) |
| Step 13的x_t包含75%噪声 | MSE≈1.4符合理论预期 | [FLOW_MATCHING_ANALYSIS.md](./FLOW_MATCHING_ANALYSIS.md) |

---

## 🗂️ 文档结构

```
doc/
├── README.md                              # 本文件（文档导航）
├── 使用说明.md                            # 模型使用完整指南
├── 实验说明.md                            # 实验研究核心文档 ⭐
├── CFG截断加速技术说明.md                  # CFG截断详解
├── WAN2.2模型加载优化分析.md               # 模型加载机制
├── FLOW_MATCHING_ANALYSIS.md              # Flow Matching理论
├── CONTINUITY_ANALYSIS_README.md          # 连续性分析
├── T2I_EVALUATION_README.md               # T2I评估完整说明
└── T2I_EVALUATION_QUICKSTART.md           # T2I评估快速开始
```

---

## 🎯 推荐阅读顺序

### **初学者**
1. [使用说明.md](./使用说明.md) - 了解基本使用
2. [实验说明.md](./实验说明.md) - 理解实验设计
3. [CFG截断加速技术说明.md](./CFG截断加速技术说明.md) - 学习优化技巧

### **研究者**
1. [实验说明.md](./实验说明.md) - 实验方法和结果
2. [FLOW_MATCHING_ANALYSIS.md](./FLOW_MATCHING_ANALYSIS.md) - 理论基础
3. [CONTINUITY_ANALYSIS_README.md](./CONTINUITY_ANALYSIS_README.md) - 深入分析

### **系统优化者**
1. [使用说明.md](./使用说明.md) - 优化参数
2. [WAN2.2模型加载优化分析.md](./WAN2.2模型加载优化分析.md) - 加载机制
3. [CFG截断加速技术说明.md](./CFG截断加速技术说明.md) - 加速策略

---

## 🔧 实验复现

所有实验都可以通过以下命令复现：

```bash
# 误差分析
python generate.py --task t2v-A14B --enable_debug --prompt "Your prompt"

# 方法对比
python Wan2.2/compare_cfg_baseline.py --ckpt_dir ./weights --prompt "Your prompt"

# 连续性分析
python Wan2.2/analyze_temporal_continuity.py --model_path ./weights --prompt "Your prompt"
```

详细参数和说明请参考 [实验说明.md](./实验说明.md)

---

## 📝 更新日志

- **2025-01**: 完成三大核心实验
  - 误差分析实验
  - 方法对比实验
  - 帧连续性分析实验
- **2025-01**: 验证CFG截断和帧截断的有效性
- **2025-01**: 理论分析Flow Matching机制

---

## 💡 贡献指南

如果你想添加新的实验或文档：

1. 在相应的脚本中实现实验逻辑
2. 在 `doc/实验说明.md` 中添加实验描述
3. 如有需要，创建独立的详细文档
4. 更新本 `README.md` 的文档目录

---

## 📧 联系方式

如有问题或建议，请：
1. 参考相关文档
2. 查看代码注释
3. 运行示例脚本验证

---

**最后更新**: 2025年1月
**文档版本**: 1.0

