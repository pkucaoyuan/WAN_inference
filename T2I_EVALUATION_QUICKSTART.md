# T2I评估工具 - 快速开始指南

## 📁 新的目录结构

所有评估相关文件已移至项目根目录，使用更简洁：

```
WAN_inference/
├── Wan2.2/                          # 模型代码
│   ├── generate.py
│   ├── configs/
│   └── wan/
├── batch_generate_t2i.py            # 单GPU批量生成 ✨
├── batch_generate_t2i_multigpu.py   # 多GPU批量生成 ✨
├── download_mscoco.py               # 数据下载 ✨
├── evaluate_t2i.py                  # 评估指标 ✨
├── run_full_evaluation.sh           # 完整流程 ✨
├── requirements_evaluation.txt      # 评估依赖 ✨
└── T2I_EVALUATION_README.md         # 详细文档
```

## 🚀 一键运行完整评估

从项目根目录运行：

```bash
# 设置模型路径
export MODEL_PATH="/home/caoyuan/efs/cy/WAN_inference/WAN2.2-27B/T2V_A14B_weights"

# 快速测试（16个样本，多GPU）
export NUM_SAMPLES=16
export USE_MULTIGPU=true
export GPU_IDS="0 1 2 3"

bash run_full_evaluation.sh
```

## 📝 分步执行

### 步骤1：下载数据

```bash
python download_mscoco.py \
    --output_dir ./mscoco_data \
    --num_samples 16 \
    --skip_images
```

### 步骤2：多GPU批量生成

```bash
python batch_generate_t2i_multigpu.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./evaluation_results/generated_baseline \
    --model_path $MODEL_PATH \
    --num_samples 16 \
    --num_inference_steps 20 \
    --guidance_scale 7.5 \
    --gpu_ids "0 1 2 3"
```

### 步骤3：评估

```bash
python evaluate_t2i.py \
    --generated_dir ./evaluation_results/generated_baseline \
    --real_dir ./mscoco_data/images/val2014 \
    --prompts_csv ./mscoco_data/prompts.csv \
    --metrics fid clip \
    --output_file ./evaluation_results/metrics_baseline.json
```

## ✅ 修复的问题

### 之前的错误
```
FileNotFoundError: [Errno 2] No such file or directory: 
'/home/caoyuan/efs/cy/WAN_inference/Wan2.2/Wan2.2/configs/t2v_A14B.yaml'
                                      ^^^^^^ 路径重复
```

### 解决方案
1. ✅ 将所有评估脚本移至根目录
2. ✅ 修复配置文件路径查找逻辑
3. ✅ 更新sys.path设置

### 新的路径逻辑
```python
# batch_generate_t2i_multigpu.py
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'Wan2.2'))  # 添加Wan2.2到路径

# 配置文件查找
config_path = os.path.join(model_path, "config.yaml")
if not os.path.exists(config_path):
    config_path = os.path.join(script_dir, "Wan2.2/configs/t2v_A14B.yaml")
```

## 🔧 使用提示

### 1. 确保从根目录运行

```bash
# ✅ 正确
cd /home/caoyuan/efs/cy/WAN_inference
python batch_generate_t2i_multigpu.py --help

# ❌ 错误
cd /home/caoyuan/efs/cy/WAN_inference/Wan2.2
python ../batch_generate_t2i_multigpu.py --help  # 路径会混乱
```

### 2. 检查GPU可用性

```bash
# 查看GPU状态
nvidia-smi

# 测试单个样本
python batch_generate_t2i_multigpu.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./test_output \
    --model_path $MODEL_PATH \
    --num_samples 1 \
    --gpu_ids "0"
```

### 3. 调试模式

```bash
# 启用Python调试输出
python -u batch_generate_t2i_multigpu.py \
    --prompts_csv ./mscoco_data/prompts.csv \
    --output_dir ./debug_output \
    --model_path $MODEL_PATH \
    --num_samples 4 \
    --gpu_ids "0 1" 2>&1 | tee debug.log
```

## 📊 预期输出

### 生成阶段
```
📋 配置信息:
   Prompts文件: ./mscoco_data/prompts.csv
   输出目录: ./evaluation_results/generated_baseline
   样本数量: 16
   使用GPU: [0, 1, 2, 3] (共 4 张)
   ...

[GPU 0] 🚀 启动worker，设备: cuda:0
[GPU 1] 🚀 启动worker，设备: cuda:1
[GPU 2] 🚀 启动worker，设备: cuda:2
[GPU 3] 🚀 启动worker，设备: cuda:3

[GPU 0] 📋 分配到 4 个任务 (总共 16 个)
[GPU 0] 🔧 加载模型...
[GPU 0] ✅ 模型加载成功

[GPU 0] 🎨 生成 1/4: 000000 - "A cat sitting on a windowsill"
...
```

### 评估阶段
```
📊 T2I评估指标计算
============================================================
生成图片: ./evaluation_results/generated_baseline
真实图片: ./mscoco_data/images/val2014
Prompts: ./mscoco_data/prompts.csv
指标: ['fid', 'clip']
============================================================

计算 FID...
✅ FID: 23.45

计算 CLIP Score...
✅ CLIP Score: 0.285

结果保存到: ./evaluation_results/metrics_baseline.json
```

## 📖 更多信息

详细文档请参考：`T2I_EVALUATION_README.md`

---

**祝评估顺利！** 🎉

