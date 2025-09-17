# WAN2.2-T2V-A14B (27B MOE) 模型使用说明

## 概述
这是WAN2.2-T2V-A14B（Text-to-Video）模型的代码仓库。此仓库**仅包含模型代码和配置文件**，不包含模型权重，以避免本地存储和运行大型MOE模型。

## 🏗️ 模型架构特点

### MOE（Mixture of Experts）架构
- **总参数量**: 27B（270亿参数）
- **激活参数**: 14B（每次推理只激活一个专家）
- **专家分工**:
  - **高噪声专家**: 处理扩散过程早期阶段（整体布局和运动）
  - **低噪声专家**: 处理扩散过程后期阶段（细节优化）
- **切换机制**: 基于信噪比(SNR)自动选择专家

## 📁 目录结构
```
WAN2.2-27B/
├── Wan2.2-T2V-A14B/                    # T2V-A14B模型仓库
│   ├── high_noise_model/               # 高噪声专家模型
│   │   ├── config.json                 # 高噪声专家配置
│   │   └── diffusion_pytorch_model.safetensors.index.json  # 权重索引
│   ├── low_noise_model/                # 低噪声专家模型
│   │   ├── config.json                 # 低噪声专家配置
│   │   └── diffusion_pytorch_model.safetensors.index.json  # 权重索引
│   ├── google/umt5-xxl/                # T5文本编码器
│   ├── assets/                         # 资源文件
│   ├── configuration.json              # 模型总配置
│   └── README.md                       # 官方文档
├── download_T2V_A14B_weights.py        # Python下载脚本
├── download_T2V_A14B_weights.bat       # Windows下载脚本
└── WAN2.2-T2V-A14B使用说明.md          # 本文档
```

## 📥 下载模型权重

### 方法一：使用Python脚本（推荐）
```bash
python download_T2V_A14B_weights.py
```

### 方法二：使用Windows批处理脚本
```bash
download_T2V_A14B_weights.bat
```

### 方法三：手动使用Hugging Face CLI
```bash
# 安装依赖
pip install huggingface_hub[cli]

# 下载到指定目录
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./T2V_A14B_weights
```

### 方法四：使用Git LFS（如果在仓库目录中）
```bash
# 安装Git LFS
git lfs install

# 进入模型目录并拉取LFS文件
cd Wan2.2-T2V-A14B
git lfs pull
```

## ⚠️ 重要注意事项

### 🚫 本地运行限制
- **严禁在本地运行此模型**
- 模型文件非常大（约111GB）
- 需要80GB+的GPU显存
- 推荐仅在远程服务器或云平台上运行

### 💾 存储需求
| 组件 | 大小 | 说明 |
|------|------|------|
| 高噪声专家模型 | ~53GB | 处理早期扩散阶段 |
| 低噪声专家模型 | ~53GB | 处理后期扩散阶段 |
| T5编码器 | ~4GB | 文本理解组件 |
| VAE编码器 | ~1GB | 视觉编码组件 |
| **总计** | **~111GB** | **完整模型大小** |

### 🖥️ 硬件要求（仅供参考，请勿本地运行）
- **GPU**: NVIDIA A100 80GB 或更高
- **RAM**: 64GB+
- **存储**: 150GB+ 可用空间
- **多GPU**: 支持2-8卡并行推理
- **CUDA**: 11.8+
- **Python**: 3.8+

## 🌐 推荐的远程运行平台

1. **启智算力平台** - 提供A100等高端GPU资源
2. **共绩算力** - 支持容器化部署
3. **阿里云PAI** - 企业级AI计算平台
4. **腾讯云TI** - 专业AI训练推理平台
5. **AWS/Azure/GCP** - 国际云计算平台

## 🚀 推理性能

### 单GPU推理（A100 80GB）
- **生成时间**: 约8-12分钟/视频（720P, 5秒）
- **内存占用**: 70-75GB显存
- **推荐参数**: `--offload_model True --convert_model_dtype`

### 多GPU推理（2-8卡A100）
- **生成时间**: 约3-6分钟/视频（720P, 5秒）
- **并行策略**: FSDP + DeepSpeed Ulysses
- **推荐参数**: `--dit_fsdp --t5_fsdp --ulysses_size 4`

## 🎯 MOE专家切换机制

```python
# 专家选择基于信噪比(SNR)
if SNR >= threshold:
    # 使用高噪声专家（早期阶段）
    active_expert = "high_noise_model"
else:
    # 使用低噪声专家（后期阶段）  
    active_expert = "low_noise_model"
```

## 📊 与其他模型对比

| 模型 | 参数量 | 架构 | 激活参数 | 推理效率 |
|------|--------|------|----------|----------|
| WAN2.2-TI2V-5B | 5B | Dense | 5B | ⭐⭐⭐⭐⭐ |
| **WAN2.2-T2V-A14B** | **27B** | **MOE** | **14B** | ⭐⭐⭐ |
| 其他T2V模型 | 10-20B | Dense | 10-20B | ⭐⭐ |

## 🔧 推理命令示例

```bash
# 单GPU推理（需要80GB+显存）
python generate.py --task t2v-A14B --size 1280*720 \
    --ckpt_dir ./T2V_A14B_weights \
    --offload_model True --convert_model_dtype \
    --prompt "A beautiful sunset over the ocean"

# 多GPU推理（4卡）
torchrun --nproc_per_node=4 generate.py --task t2v-A14B \
    --size 1280*720 --ckpt_dir ./T2V_A14B_weights \
    --dit_fsdp --t5_fsdp --ulysses_size 4 \
    --prompt "A beautiful sunset over the ocean"
```

## 📄 许可证
请参考原始仓库的Apache 2.0许可证文件。

## 🆘 技术支持
如有问题，请参考：
1. 官方README.md文档
2. Hugging Face模型页面: https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B
3. WAN官方技术论文: arXiv:2503.20314

---
**再次提醒：此仓库仅用于代码研究，请勿在本地运行27B MOE模型！**
