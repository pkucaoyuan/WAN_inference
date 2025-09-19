#!/usr/bin/env python3
"""
WAN2.2 多机分布式推理Python启动脚本
支持自动化的多机推理配置和启动
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

def check_network_connectivity(master_ip, master_port):
    """检查网络连通性"""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((master_ip, int(master_port)))
        sock.close()
        return result == 0
    except:
        return False

def start_multi_node_inference(args):
    """启动多机分布式推理"""
    
    # 计算总GPU数量
    total_gpus = args.nnodes * args.nproc_per_node
    
    print("🌐 WAN2.2 多机分布式推理配置")
    print("=" * 50)
    print(f"总机器数: {args.nnodes}")
    print(f"每机GPU数: {args.nproc_per_node}")
    print(f"总GPU数: {total_gpus}")
    print(f"主节点IP: {args.master_addr}")
    print(f"通信端口: {args.master_port}")
    print(f"当前节点: {args.node_rank}")
    print("=" * 50)
    
    # 设置环境变量
    env = os.environ.copy()
    env.update({
        'MASTER_ADDR': args.master_addr,
        'MASTER_PORT': str(args.master_port),
        'WORLD_SIZE': str(total_gpus),
        'NODE_RANK': str(args.node_rank)
    })
    
    # 检查网络连通性（工作节点）
    if args.node_rank != 0:
        print("🔗 检查与主节点的网络连通性...")
        if not check_network_connectivity(args.master_addr, args.master_port):
            print(f"❌ 无法连接到主节点 {args.master_addr}:{args.master_port}")
            return False
        print("✅ 网络连通性正常")
    
    # 构建torchrun命令
    cmd = [
        'torchrun',
        f'--nproc_per_node={args.nproc_per_node}',
        f'--nnodes={args.nnodes}',
        f'--node_rank={args.node_rank}',
        f'--master_addr={args.master_addr}',
        f'--master_port={args.master_port}',
        'generate.py',
        '--task', args.task,
        '--size', args.size,
        '--ckpt_dir', args.ckpt_dir,
        '--frame_num', str(args.frame_num),
        '--dit_fsdp',
        '--t5_fsdp',
        '--ulysses_size', str(total_gpus),
        '--prompt', args.prompt
    ]
    
    # 添加可选参数
    if args.cfg_truncate_steps > 0:
        cmd.extend(['--cfg_truncate_steps', str(args.cfg_truncate_steps)])
    if args.cfg_truncate_high_noise_steps > 0:
        cmd.extend(['--cfg_truncate_high_noise_steps', str(args.cfg_truncate_high_noise_steps)])
    if args.fast_loading:
        cmd.append('--fast_loading')
    
    print(f"🚀 启动节点 {args.node_rank} 的分布式推理...")
    print(f"📝 执行命令: {' '.join(cmd)}")
    
    # 切换到Wan2.2目录
    wan2_dir = Path(__file__).parent.parent / "Wan2.2"
    
    try:
        # 启动推理
        start_time = time.time()
        result = subprocess.run(cmd, cwd=wan2_dir, env=env, check=True)
        total_time = time.time() - start_time
        
        print(f"✅ 节点 {args.node_rank} 推理完成")
        print(f"⏱️ 节点总耗时: {total_time:.2f}秒")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 节点 {args.node_rank} 推理失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="WAN2.2 多机分布式推理启动器")
    
    # 分布式配置
    parser.add_argument("--master_addr", type=str, default="127.0.0.1",
                       help="主节点IP地址")
    parser.add_argument("--master_port", type=int, default=29500,
                       help="通信端口")
    parser.add_argument("--nnodes", type=int, default=2,
                       help="机器总数")
    parser.add_argument("--nproc_per_node", type=int, default=4,
                       help="每台机器的GPU数量")
    parser.add_argument("--node_rank", type=int, required=True,
                       help="当前机器编号 (0为主节点)")
    
    # 模型配置
    parser.add_argument("--task", type=str, default="t2v-A14B",
                       help="推理任务类型")
    parser.add_argument("--size", type=str, default="1280*720",
                       help="视频分辨率")
    parser.add_argument("--ckpt_dir", type=str, 
                       default="../WAN2.2-27B/T2V_A14B_weights",
                       help="模型权重目录")
    parser.add_argument("--frame_num", type=int, default=81,
                       help="生成帧数")
    parser.add_argument("--prompt", type=str,
                       default="A beautiful sunset over the ocean",
                       help="生成提示词")
    
    # 优化参数
    parser.add_argument("--cfg_truncate_steps", type=int, default=0,
                       help="CFG截断步数")
    parser.add_argument("--cfg_truncate_high_noise_steps", type=int, default=0,
                       help="高噪声专家CFG截断步数")
    parser.add_argument("--fast_loading", action="store_true",
                       help="启用快速加载模式")
    
    args = parser.parse_args()
    
    success = start_multi_node_inference(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
