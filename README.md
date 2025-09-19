# WAN推理项目 (WAN Inference Project)

这是一个用于研究和部署WAN2.2系列视频生成模型的完整项目，包含5B Dense模型和27B MOE模型的代码、配置和文档。

## 🏗️ 项目结构

```
WAN_inference/
├── Wan2.2/                           # WAN2.2推理代码仓库
├── Wan2.2-TI2V-5B/                   # 5B Dense模型
│   ├── 配置文件和README
│   └── 模型索引文件
├── WAN2.2-27B/                       # 27B MOE模型
│   ├── Wan2.2-T2V-A14B/             # T2V-A14B模型仓库
│   ├── download_T2V_A14B_weights.*   # 权重下载脚本
│   ├── WAN2.2-T2V-A14B使用说明.md   # 使用说明
│   └── WAN2.2_T2V_A14B_推理结构详解.md # 技术详解
├── download_model_weights.*          # 5B模型权重下载脚本
└── 使用说明.md                       # 总体使用说明
```

## 📊 模型对比

| 模型 | 参数量 | 架构类型 | 激活参数 | 支持分辨率 | 推荐用途 |
|------|--------|----------|----------|------------|----------|
| **WAN2.2-TI2V-5B** | 5B | Dense | 5B | 720P | 研究和轻量部署 |
| **WAN2.2-T2V-A14B** | 27B | MOE | 14B | 720P | 高质量生成 |

## 🚀 快速开始

### 1. 环境安装
```bash
# 克隆项目
git clone https://github.com/pkucaoyuan/WAN_inference.git
cd WAN_inference

# 进入推理代码目录
cd Wan2.2

# 安装依赖（确保torch >= 2.4.0）
pip install -r requirements.txt

# 如果需要语音转视频功能，额外安装：
pip install -r requirements_s2v.txt

# 注意：如果flash_attn安装失败，先安装其他包，最后安装flash_attn
```

### 2. 下载模型权重

**5B模型（后台下载）**:
```bash
cd ..  # 回到项目根目录
python download_model_weights.py
# 监控下载进度: tail -f ./model_weights/download.log
```

**27B MOE模型（后台下载）**:
```bash
cd WAN2.2-27B
python download_T2V_A14B_weights.py
# 监控下载进度: tail -f ./T2V_A14B_weights/download.log
```

### 3. 推理示例

**5B模型推理（RTX 4090可运行）**:
```bash
cd Wan2.2  # 进入推理代码目录
python generate.py --task ti2v-5B --size 1280*704 \
    --ckpt_dir ../model_weights \
    --offload_model True --convert_model_dtype --t5_cpu \
    --frame_num 81 \
    --prompt "A beautiful sunset over the ocean"
```

**27B MOE模型单GPU推理（需要80GB+显存）**:
```bash
python generate.py --task t2v-A14B --size 1280*720 \
    --ckpt_dir ../WAN2.2-27B/T2V_A14B_weights \
    --offload_model True --convert_model_dtype \
    --frame_num 81 \
    --prompt "A beautiful sunset over the ocean"
```

**27B MOE模型多GPU推理（推荐）**:
```bash
torchrun --nproc_per_node=4 generate.py --task t2v-A14B \
    --size 1280*720 --ckpt_dir ../WAN2.2-27B/T2V_A14B_weights \
    --dit_fsdp --t5_fsdp --ulysses_size 4 \
    --frame_num 81 \
    --cfg_truncate_steps 5 \
    --prompt "A beautiful sunset over the ocean"
```

## ⚡ **推理加速优化**

### **CFG截断技术（新功能）**
为了加速WAN2.2推理，我们实现了CFG截断技术，在最后几步跳过条件前传：

```bash
# 基础CFG截断使用
--cfg_truncate_steps 5    # 在最后5步跳过条件前传

# 不同加速等级
--cfg_truncate_steps 3    # 激进加速（20-25%时间减少）
--cfg_truncate_steps 5    # 平衡模式（25-35%时间减少）推荐
--cfg_truncate_steps 8    # 保守加速（35-40%时间减少）
--cfg_truncate_steps 0    # 禁用CFG截断
```

### **完整的加速推理示例**
```bash
# 5B模型 + CFG截断
cd Wan2.2
python generate.py --task ti2v-5B --size 1280*704 \
    --ckpt_dir ../model_weights \
    --cfg_truncate_steps 5 \
    --frame_num 81 \
    --prompt "A beautiful sunset over the ocean"

# 27B MOE模型 + CFG截断 + 多GPU
torchrun --nproc_per_node=4 generate.py --task t2v-A14B \
    --size 1280*720 --ckpt_dir ../WAN2.2-27B/T2V_A14B_weights \
    --dit_fsdp --t5_fsdp --ulysses_size 4 \
    --cfg_truncate_steps 5 \
    --frame_num 81 \
    --prompt "A beautiful sunset over the ocean"
```

## ⚠️ 重要说明

- **本项目仅供研究使用，严禁在本地运行大型模型**
- **模型权重未包含在仓库中，需要单独下载**
- **推荐在远程服务器或云平台上运行模型**
- **27B MOE模型需要80GB+显存的GPU**

## 📚 技术文档

- [5B模型使用说明](使用说明.md)
- [27B MOE模型使用说明](WAN2.2-27B/WAN2.2-T2V-A14B使用说明.md)
- [27B MOE模型技术详解](WAN2.2-27B/WAN2.2_T2V_A14B_推理结构详解.md)

## 🔧 系统要求

### 最低要求 (5B模型)
- **GPU**: RTX 4090 24GB
- **RAM**: 32GB
- **存储**: 50GB
- **Python**: 3.8+
- **PyTorch**: >= 2.4.0

### 推荐配置 (27B MOE模型)
- **GPU**: A100 80GB × 4-8
- **RAM**: 128GB
- **存储**: 200GB
- **Python**: 3.8+
- **PyTorch**: >= 2.4.0

## 📝 重要参数说明

### **帧数限制**
- `--frame_num` 必须满足公式：**4n+1**
- 有效帧数：5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, **81**, 85...
- **不能设置为1帧**，最小为5帧

### **分辨率设置**
- **5B模型**: `1280*704` 或 `704*1280`（720P）
- **27B模型**: `1280*720`（标准720P）
- **I2V任务**: 宽高比跟随输入图像

### **内存优化参数**
- `--offload_model True`: 模型卸载到CPU
- `--convert_model_dtype`: 转换模型数据类型
- `--t5_cpu`: T5编码器在CPU运行
- `--dit_fsdp`: DiT模型FSDP分片
- `--ulysses_size N`: 序列并行度

## 📄 许可证

本项目遵循Apache 2.0许可证。模型权重请参考官方许可证。

## 🙏 致谢

感谢阿里巴巴WAN团队开源的优秀工作：
- [WAN2.2官方仓库](https://github.com/Wan-Video/Wan2.2)
- [Hugging Face模型页面](https://huggingface.co/Wan-AI)

---

**再次提醒：请勿在本地运行大型模型，仅用于代码研究和远程部署！**
