#!/bin/bash
# WAN2.2 多机分布式推理启动脚本

# 配置参数
MASTER_IP="192.168.1.100"      # 主节点IP地址
MASTER_PORT="29500"            # 通信端口
TOTAL_NODES=2                  # 机器总数
GPUS_PER_NODE=4               # 每台机器GPU数量
TOTAL_GPUS=$((TOTAL_NODES * GPUS_PER_NODE))

# 模型配置
MODEL_TASK="t2v-A14B"
MODEL_DIR="../WAN2.2-27B/T2V_A14B_weights"
RESOLUTION="1280*720"
FRAME_NUM=81
PROMPT="A beautiful sunset over the ocean with waves crashing on the shore"

echo "🌐 WAN2.2 多机分布式推理配置"
echo "=================================="
echo "总机器数: $TOTAL_NODES"
echo "每机GPU数: $GPUS_PER_NODE" 
echo "总GPU数: $TOTAL_GPUS"
echo "主节点IP: $MASTER_IP"
echo "通信端口: $MASTER_PORT"
echo "=================================="

# 检查当前机器的节点编号
if [ -z "$NODE_RANK" ]; then
    echo "请设置NODE_RANK环境变量 (0为主节点, 1,2,3...为工作节点)"
    echo "例如: export NODE_RANK=0  # 主节点"
    echo "     export NODE_RANK=1  # 工作节点1"
    exit 1
fi

echo "当前节点编号: $NODE_RANK"

# 设置环境变量
export MASTER_ADDR=$MASTER_IP
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$TOTAL_GPUS
export NODE_RANK=$NODE_RANK

# 检查网络连通性（仅工作节点）
if [ "$NODE_RANK" -ne 0 ]; then
    echo "🔗 检查与主节点的网络连通性..."
    if ! ping -c 1 $MASTER_IP > /dev/null 2>&1; then
        echo "❌ 无法连接到主节点 $MASTER_IP"
        exit 1
    fi
    echo "✅ 网络连通性正常"
fi

# 启动分布式推理
echo "🚀 启动节点 $NODE_RANK 的分布式推理..."

cd "$(dirname "$0")/../Wan2.2"

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$TOTAL_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_IP \
    --master_port=$MASTER_PORT \
    generate.py \
    --task $MODEL_TASK \
    --size $RESOLUTION \
    --ckpt_dir $MODEL_DIR \
    --frame_num $FRAME_NUM \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size $TOTAL_GPUS \
    --cfg_truncate_steps 5 \
    --cfg_truncate_high_noise_steps 3 \
    --prompt "$PROMPT"

echo "✅ 节点 $NODE_RANK 推理完成"
