#!/bin/bash
# 完整的T2I评估流程脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "T2I完整评估流程"
echo "=========================================="

# 配置参数
MODEL_PATH=${MODEL_PATH:-"/path/to/your/model"}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
SEED_START=${SEED_START:-42}
NUM_STEPS=${NUM_STEPS:-20}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7.5}
DEVICE=${DEVICE:-"cuda:0"}
USE_MULTIGPU=${USE_MULTIGPU:-false}  # 是否使用多GPU
GPU_IDS=${GPU_IDS:-"0 1 2 3"}  # 多GPU时使用的GPU ID

# 目录配置
DATA_DIR="./mscoco_data"
PROMPTS_CSV="${DATA_DIR}/prompts.csv"
REAL_IMAGES="${DATA_DIR}/images/val2014"

# 输出目录
OUTPUT_BASE="./evaluation_results"
mkdir -p ${OUTPUT_BASE}

echo ""
echo "配置信息:"
echo "  模型路径: ${MODEL_PATH}"
echo "  样本数量: ${NUM_SAMPLES}"
echo "  推理步数: ${NUM_STEPS}"
echo "  引导强度: ${GUIDANCE_SCALE}"
echo "  使用多GPU: ${USE_MULTIGPU}"
if [ "${USE_MULTIGPU}" = "true" ]; then
    echo "  GPU IDs: ${GPU_IDS}"
else
    echo "  设备: ${DEVICE}"
fi
echo ""

# 步骤1: 下载MS-COCO数据（如果不存在）
if [ ! -f "${PROMPTS_CSV}" ]; then
    echo "=========================================="
    echo "步骤 1/4: 下载MS-COCO数据"
    echo "=========================================="
    python download_mscoco.py \
        --output_dir ${DATA_DIR} \
        --num_samples ${NUM_SAMPLES} \
        --skip_images
    echo "✅ 数据下载完成"
else
    echo "⏭️  跳过数据下载（已存在）"
fi

# 步骤2: 生成基线图片
echo ""
echo "=========================================="
echo "步骤 2/3: 生成基线图片"
echo "=========================================="
GEN_DIR_BASELINE="${OUTPUT_BASE}/generated_baseline"

if [ "${USE_MULTIGPU}" = "true" ]; then
    python batch_generate_t2i_multigpu.py \
        --prompts_csv ${PROMPTS_CSV} \
        --output_dir ${GEN_DIR_BASELINE} \
        --model_path ${MODEL_PATH} \
        --num_samples ${NUM_SAMPLES} \
        --seed_start ${SEED_START} \
        --num_inference_steps ${NUM_STEPS} \
        --guidance_scale ${GUIDANCE_SCALE} \
        --gpu_ids ${GPU_IDS}
else
    python batch_generate_t2i.py \
        --prompts_csv ${PROMPTS_CSV} \
        --output_dir ${GEN_DIR_BASELINE} \
        --model_path ${MODEL_PATH} \
        --num_samples ${NUM_SAMPLES} \
        --seed_start ${SEED_START} \
        --num_inference_steps ${NUM_STEPS} \
        --guidance_scale ${GUIDANCE_SCALE} \
        --device ${DEVICE}
fi
echo "✅ 基线图片生成完成"

# 步骤3: 评估基线方法
echo ""
echo "=========================================="
echo "步骤 3/3: 评估基线方法"
echo "=========================================="

echo ""
echo "评估基线方法..."
python evaluate_t2i.py \
    --generated_dir ${GEN_DIR_BASELINE} \
    --real_dir ${REAL_IMAGES} \
    --prompts_csv ${PROMPTS_CSV} \
    --metrics fid clip \
    --output_json ${OUTPUT_BASE}/results_baseline.json \
    --device ${DEVICE}

# 输出结果
echo ""
echo "=========================================="
echo "✅ 评估完成！"
echo "=========================================="
echo ""
echo "结果文件:"
echo "  ${OUTPUT_BASE}/results_baseline.json"
echo ""
echo "生成图片:"
echo "  ${GEN_DIR_BASELINE}"
echo ""

# 显示结果（如果安装了jq）
if command -v jq &> /dev/null; then
    echo "评估结果:"
    echo "----------------------------------------"
    jq '.metrics' ${OUTPUT_BASE}/results_baseline.json
    echo "----------------------------------------"
fi

echo ""
echo "🎉 全部完成！"

