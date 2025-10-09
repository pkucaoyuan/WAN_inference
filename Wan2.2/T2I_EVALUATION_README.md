# T2I评估工具使用指南

本工具用于在MS-COCO数据集上评估文本到图像(T2I)生成模型的质量。

## 📋 功能特性

- ✅ 批量生成：从MS-COCO prompts批量生成单帧图片
- ✅ 自动下载：自动下载MS-COCO验证集和标注
- ✅ 多指标评估：支持FID、IS、CLIP Score、LPIPS等多种指标
- ✅ 灵活配置：支持自定义生成参数和评估选项

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装评估依赖
pip install -r requirements_evaluation.txt

# 安装CLIP（用于CLIP Score）
pip install git+https://github.com/openai/CLIP.git
```

### 2. 下载MS-COCO数据

```bash
# 下载验证集图片、标注和创建prompts CSV
python download_mscoco.py \
    --output_dir ./mscoco_data \
    --num_samples 5000

# 如果只需要prompts（不下载图片，节省空间）
python download_mscoco.py \
    --output_dir ./mscoco_data \
    --num_samples 5000 \
    --skip_images
```

**参数说明：**
- `--output_dir`: 输出目录
- `--num_samples`: 采样prompts数量（默认全部约40k，建议5000用于快速测试）
- `--skip_images`: 跳过下载图片（只下载标注）
- `--skip_fid_stats`: 跳过FID统计文件

### 3. 批量生成图片

```bash
# 基础生成（默认配置）
python batch_generate_t2i.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./generated_images \
    --model_path /path/to/your/model \
    --num_samples 1000

# 完整配置示例
python batch_generate_t2i.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./generated_images_baseline \
    --model_path /path/to/your/model \
    --num_samples 1000 \
    --seed_start 42 \
    --num_inference_steps 20 \
    --guidance_scale 7.5 \
    --height 480 \
    --width 832 \
    --device cuda:0

# 使用帧数减半优化
python batch_generate_t2i.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./generated_images_half_frame \
    --model_path /path/to/your/model \
    --num_samples 1000 \
    --enable_half_frame

# 使用CFG截断
python batch_generate_t2i.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./generated_images_cfg_truncate \
    --model_path /path/to/your/model \
    --num_samples 1000 \
    --cfg_truncation_step 15
```

**参数说明：**
- `--prompts_csv`: MS-COCO prompts CSV文件路径（必需）
- `--output_dir`: 输出图片目录（必需）
- `--model_path`: 模型路径
- `--num_samples`: 生成样本数量
- `--seed_start`: 起始随机种子
- `--num_inference_steps`: 推理步数（默认20）
- `--guidance_scale`: CFG引导强度（默认7.5）
- `--height/width`: 图片尺寸
- `--enable_half_frame`: 启用帧数减半优化
- `--cfg_truncation_step`: CFG截断步数

### 4. 计算评估指标

```bash
# 计算所有指标
python evaluate_t2i.py \
    --generated_dir ./generated_images \
    --real_dir ./mscoco_data/images/val2014 \
    --prompts_csv ./mscoco_data/prompts.csv \
    --metrics all

# 只计算FID和CLIP Score
python evaluate_t2i.py \
    --generated_dir ./generated_images \
    --real_dir ./mscoco_data/images/val2014 \
    --prompts_csv ./mscoco_data/prompts.csv \
    --metrics fid clip

# 指定输出文件
python evaluate_t2i.py \
    --generated_dir ./generated_images \
    --real_dir ./mscoco_data/images/val2014 \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_json ./results/evaluation_baseline.json
```

**参数说明：**
- `--generated_dir`: 生成图片目录（必需）
- `--real_dir`: 真实图片目录（用于FID和LPIPS）
- `--prompts_csv`: Prompts CSV文件（用于CLIP Score）
- `--metrics`: 要计算的指标（fid, is, clip, lpips, all）
- `--batch_size`: 批次大小（默认32）
- `--device`: 设备（默认cuda）
- `--output_json`: 输出JSON文件路径

## 📊 评估指标说明

### FID (Fréchet Inception Distance)
- **范围**: [0, +∞)，越低越好
- **含义**: 衡量生成图片与真实图片在特征空间的分布距离
- **优点**: 广泛使用，与人类感知相关性较好
- **缺点**: 需要大量样本（建议≥10k）

### IS (Inception Score)
- **范围**: [1, +∞)，越高越好
- **含义**: 衡量生成图片的质量和多样性
- **优点**: 不需要真实图片
- **缺点**: 对模式崩溃不敏感

### CLIP Score
- **范围**: [-1, 1]，越高越好
- **含义**: 衡量生成图片与文本提示的一致性
- **优点**: 直接衡量文本-图像对齐
- **缺点**: 可能忽略图片质量

### LPIPS (Learned Perceptual Image Patch Similarity)
- **范围**: [0, +∞)，越低越好
- **含义**: 衡量感知相似度
- **优点**: 与人类感知相关性好
- **缺点**: 需要配对比较

## 📁 文件结构

```
Wan2.2/
├── batch_generate_t2i.py       # 批量生成脚本
├── download_mscoco.py           # 数据下载脚本
├── evaluate_t2i.py              # 评估脚本
├── requirements_evaluation.txt  # 评估依赖
└── T2I_EVALUATION_README.md    # 本文档

mscoco_data/                     # MS-COCO数据目录
├── prompts.csv                  # Prompts文件
├── annotations/                 # 标注文件
│   └── captions_val2014.json
└── images/                      # 图片目录
    └── val2014/

generated_images/                # 生成图片目录
├── 000001_seed42.png
├── 000002_seed43.png
└── evaluation_results.json      # 评估结果
```

## 🔧 常见问题

### Q1: 内存不足怎么办？
**A**: 减小`--batch_size`或`--num_samples`

### Q2: FID计算很慢？
**A**: FID需要大量样本，建议使用GPU加速，或减少样本数量

### Q3: CLIP Score需要什么依赖？
**A**: 需要安装CLIP：`pip install git+https://github.com/openai/CLIP.git`

### Q4: 如何对比不同方法？
**A**: 使用不同的`--output_dir`生成多组图片，然后分别评估

```bash
# 方法1：基线
python batch_generate_t2i.py --output_dir ./gen_baseline ...
python evaluate_t2i.py --generated_dir ./gen_baseline ...

# 方法2：帧数减半
python batch_generate_t2i.py --output_dir ./gen_half_frame --enable_half_frame ...
python evaluate_t2i.py --generated_dir ./gen_half_frame ...

# 方法3：CFG截断
python batch_generate_t2i.py --output_dir ./gen_cfg_truncate --cfg_truncation_step 15 ...
python evaluate_t2i.py --generated_dir ./gen_cfg_truncate ...
```

### Q5: 生成失败的prompts怎么处理？
**A**: 失败记录会保存在`{output_dir}/failed_prompts.txt`，可以单独重新生成

## 📈 建议的评估流程

1. **小规模测试**（100-500样本）：快速验证流程
2. **中等规模评估**（1000-5000样本）：初步评估性能
3. **大规模评估**（10000+样本）：最终评估和论文结果

## 🎯 性能优化建议

1. **使用多GPU**：修改脚本支持`torch.distributed`
2. **批量推理**：增大`batch_size`（如果内存允许）
3. **混合精度**：使用`bf16`或`fp16`
4. **缓存特征**：预计算Inception特征用于FID

## 📞 技术支持

如有问题，请检查：
1. 模型路径是否正确
2. 依赖包是否完整安装
3. GPU内存是否充足
4. CUDA版本是否兼容

---

**祝评估顺利！** 🎉

