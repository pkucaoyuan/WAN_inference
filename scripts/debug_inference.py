#!/usr/bin/env python3
"""
WAN2.2 推理调试和错误恢复脚本
用于诊断和解决常见的推理问题
"""

import argparse
import os
import subprocess
import sys
import torch
from pathlib import Path

def check_gpu_status():
    """检查GPU状态"""
    print("🔍 GPU状态检查")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 个GPU")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_total = props.total_memory / 1024**3
        memory_free = torch.cuda.get_device_properties(i).total_memory / 1024**3
        try:
            torch.cuda.set_device(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            memory_free = memory_total - memory_allocated
        except:
            memory_allocated = 0
            memory_cached = 0
            memory_free = memory_total
            
        print(f"  GPU {i}: {props.name}")
        print(f"    总内存: {memory_total:.1f}GB")
        print(f"    已用内存: {memory_allocated:.1f}GB")
        print(f"    缓存内存: {memory_cached:.1f}GB") 
        print(f"    可用内存: {memory_free:.1f}GB")
    
    return True

def test_simple_inference(args):
    """测试简单推理"""
    print("\n🧪 简单推理测试")
    print("-" * 40)
    
    # 最保守的配置
    cmd = [
        sys.executable, "generate.py",
        "--task", args.task,
        "--size", "1280*720", 
        "--ckpt_dir", args.ckpt_dir,
        "--frame_num", "1",
        "--offload_model", "True",
        "--convert_model_dtype",
        "--t5_cpu",
        "--prompt", "test"
    ]
    
    print(f"📝 测试命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd="Wan2.2", capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print("✅ 简单推理测试成功")
            return True
        else:
            print("❌ 简单推理测试失败")
            print(f"错误输出: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时（10分钟）")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def suggest_optimal_config(gpu_memory_gb):
    """根据GPU内存建议最优配置"""
    print(f"\n💡 基于{gpu_memory_gb:.1f}GB GPU内存的推荐配置")
    print("-" * 50)
    
    if gpu_memory_gb >= 80:
        print("🚀 高端配置 (A100 80GB+):")
        print("  python generate.py --task t2v-A14B --fast_loading")
        print("  # 或多GPU: torchrun --nproc_per_node=4 --dit_fsdp --t5_fsdp")
        
    elif gpu_memory_gb >= 40:
        print("⚡ 中端配置 (A100 40GB):")
        print("  python generate.py --task t2v-A14B --offload_model True --convert_model_dtype")
        
    elif gpu_memory_gb >= 24:
        print("💾 轻量配置 (RTX 4090):")
        print("  python generate.py --task ti2v-5B --offload_model True --t5_cpu")
        
    else:
        print("⚠️ 内存不足，建议使用云平台")

def main():
    parser = argparse.ArgumentParser(description="WAN2.2推理调试工具")
    parser.add_argument("--task", default="t2v-A14B", help="任务类型")
    parser.add_argument("--ckpt_dir", default="../WAN2.2-27B/T2V_A14B_weights", help="模型目录")
    parser.add_argument("--test_inference", action="store_true", help="运行推理测试")
    
    args = parser.parse_args()
    
    print("🔧 WAN2.2 推理调试工具")
    print("=" * 50)
    
    # 检查GPU状态
    if not check_gpu_status():
        return
    
    # 获取GPU内存信息
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    suggest_optimal_config(gpu_memory)
    
    # 可选的推理测试
    if args.test_inference:
        test_simple_inference(args)
    
    print("\n📋 常见错误解决方案:")
    print("1. SIGSEGV (-11): 使用 --offload_model True --convert_model_dtype")
    print("2. OOM: 降低并行度或使用 --t5_cpu")
    print("3. 加载慢: 使用 --fast_loading 或多GPU分片")
    print("4. 网络错误: 检查防火墙和NCCL配置")

if __name__ == "__main__":
    main()
