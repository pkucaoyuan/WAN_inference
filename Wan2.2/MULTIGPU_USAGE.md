# 多GPU并行生成使用指南

## 🚀 核心优势

使用`batch_generate_t2i_multigpu.py`可以实现：
- **线性加速**：4张GPU约4倍速度，8张GPU约8倍速度
- **任务隔离**：每张GPU处理不同样本，互不干扰
- **结果一致**：与单GPU版本完全相同的seed和输出
- **自动恢复**：中断后重新运行会跳过已生成的图片

## 📊 性能对比

| GPU数量 | 样本数 | 单GPU耗时 | 多GPU耗时 | 加速比 |
|---------|--------|-----------|-----------|--------|
| 1       | 1000   | ~2小时    | ~2小时    | 1x     |
| 4       | 1000   | ~2小时    | ~30分钟   | 4x     |
| 8       | 1000   | ~2小时    | ~15分钟   | 8x     |
| 4       | 5000   | ~10小时   | ~2.5小时  | 4x     |
| 8       | 10000  | ~20小时   | ~2.5小时  | 8x     |

*注：实际耗时取决于模型大小、硬件配置和生成参数*

## 🎯 使用方法

### 基础用法

```bash
# 自动使用所有可用GPU
python batch_generate_t2i_multigpu.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./generated_images \
    --model_path /path/to/your/model \
    --num_samples 5000
```

### 指定GPU

```bash
# 使用GPU 0,1,2,3
python batch_generate_t2i_multigpu.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./generated_images \
    --model_path /path/to/your/model \
    --num_samples 5000 \
    --gpu_ids 0 1 2 3

# 只使用GPU 2和3（跳过0,1，可能被其他任务占用）
python batch_generate_t2i_multigpu.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./generated_images \
    --model_path /path/to/your/model \
    --num_samples 5000 \
    --gpu_ids 2 3
```

### 完整配置示例

```bash
# 使用4张GPU + 帧数减半优化 + 自定义参数
python batch_generate_t2i_multigpu.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./generated_images_optimized \
    --model_path /path/to/your/model \
    --num_samples 10000 \
    --gpu_ids 0 1 2 3 \
    --seed_start 42 \
    --num_inference_steps 20 \
    --guidance_scale 7.5 \
    --height 480 \
    --width 832 \
    --enable_half_frame \
    --dtype bf16
```

### 使用环境变量（一键评估脚本）

```bash
# 启用多GPU模式
export USE_MULTIGPU=true
export GPU_IDS="0 1 2 3"
export MODEL_PATH="/path/to/your/model"
export NUM_SAMPLES=5000

# 运行完整评估
bash run_full_evaluation.sh
```

## 🔍 工作原理

### 任务分配策略

采用**轮询（Round-Robin）**分配：
- GPU 0: 处理样本 0, 4, 8, 12, 16, ...
- GPU 1: 处理样本 1, 5, 9, 13, 17, ...
- GPU 2: 处理样本 2, 6, 10, 14, 18, ...
- GPU 3: 处理样本 3, 7, 11, 15, 19, ...

### Seed计算

每个样本的seed计算方式：
```python
seed = seed_start + sample_index
```

例如：
- 样本0: seed = 42 + 0 = 42
- 样本1: seed = 42 + 1 = 43
- 样本4: seed = 42 + 4 = 46

**保证与单GPU版本完全一致！**

### 进度显示

每个GPU独立显示进度条：
```
GPU 0: 100%|██████████| 250/250 [30:00<00:00, success: 248, failed: 2]
GPU 1: 100%|██████████| 250/250 [30:15<00:00, success: 250, failed: 0]
GPU 2: 100%|██████████| 250/250 [30:10<00:00, success: 249, failed: 1]
GPU 3: 100%|██████████| 250/250 [30:05<00:00, success: 250, failed: 0]
```

## 📝 输出文件

### 生成的图片

文件名格式：`{image_id}_seed{seed}.png`

例如：
```
generated_images/
├── 000001_seed42.png
├── 000002_seed43.png
├── 000003_seed44.png
└── ...
```

### 失败记录

每个GPU独立保存失败记录：
```
generated_images/
├── failed_prompts_gpu0.txt
├── failed_prompts_gpu1.txt
├── failed_prompts_gpu2.txt
└── failed_prompts_gpu3.txt
```

## ⚠️ 注意事项

### 1. 内存管理

每张GPU会加载一份完整的模型，确保：
- 每张GPU有足够的显存（建议≥24GB）
- 如果显存不足，减少使用的GPU数量

### 2. 中断恢复

如果生成中断：
```bash
# 重新运行相同命令，会自动跳过已生成的图片
python batch_generate_t2i_multigpu.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./generated_images \
    --model_path /path/to/your/model \
    --num_samples 5000 \
    --gpu_ids 0 1 2 3
```

### 3. GPU选择

检查GPU可用性：
```bash
# 查看GPU状态
nvidia-smi

# 查看GPU数量
python -c "import torch; print(torch.cuda.device_count())"
```

选择空闲的GPU：
```bash
# 如果GPU 0,1被占用，使用GPU 2,3,4,5
--gpu_ids 2 3 4 5
```

### 4. 进程管理

多GPU版本使用`torch.multiprocessing`：
- 每个GPU运行在独立的Python进程中
- 进程间不共享内存
- 可以通过`Ctrl+C`中断所有进程

## 🐛 故障排查

### 问题1: CUDA out of memory

**原因**: 单张GPU显存不足

**解决**:
1. 减少使用的GPU数量
2. 使用更小的模型或更低的分辨率
3. 检查是否有其他进程占用GPU

### 问题2: 进程卡住不动

**原因**: 可能是模型加载或初始化问题

**解决**:
1. 检查模型路径是否正确
2. 确认每张GPU都能正常访问
3. 查看各个GPU的日志输出

### 问题3: 生成结果不一致

**原因**: Seed设置或任务分配问题

**解决**:
1. 确认使用相同的`--seed_start`
2. 检查文件名中的seed是否正确
3. 对比单GPU和多GPU生成的相同样本

### 问题4: 某些GPU没有工作

**原因**: GPU ID设置错误或GPU不可用

**解决**:
```bash
# 检查可用的GPU
nvidia-smi

# 确认GPU ID正确
--gpu_ids 0 1 2 3  # 使用前4张GPU
```

## 📈 性能优化建议

### 1. 合理分配样本数

确保样本数能被GPU数量整除，避免负载不均：
```bash
# 好的例子：1000样本，4张GPU，每张250个
--num_samples 1000 --gpu_ids 0 1 2 3

# 不好的例子：1001样本，4张GPU，分配不均（250,250,250,251）
--num_samples 1001 --gpu_ids 0 1 2 3
```

### 2. 使用帧数减半优化

对于T2I任务，帧数减半优化可能不适用，但可以尝试：
```bash
--enable_half_frame
```

### 3. CFG截断

在后期步骤跳过CFG计算，加速生成：
```bash
--cfg_truncation_step 15  # 在第15步后截断CFG
```

### 4. 混合精度

使用bf16或fp16减少显存占用：
```bash
--dtype bf16  # 推荐
# 或
--dtype fp16
```

## 🎓 最佳实践

### 小规模测试（100-500样本）

```bash
# 使用2张GPU快速验证
python batch_generate_t2i_multigpu.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./test_output \
    --model_path /path/to/model \
    --num_samples 100 \
    --gpu_ids 0 1
```

### 中等规模评估（1000-5000样本）

```bash
# 使用4张GPU
python batch_generate_t2i_multigpu.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./eval_output \
    --model_path /path/to/model \
    --num_samples 5000 \
    --gpu_ids 0 1 2 3
```

### 大规模评估（10000+样本）

```bash
# 使用8张GPU + 优化选项
python batch_generate_t2i_multigpu.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./large_eval_output \
    --model_path /path/to/model \
    --num_samples 10000 \
    --gpu_ids 0 1 2 3 4 5 6 7 \
    --enable_half_frame \
    --cfg_truncation_step 15 \
    --dtype bf16
```

---

**享受多GPU带来的加速吧！** 🚀

