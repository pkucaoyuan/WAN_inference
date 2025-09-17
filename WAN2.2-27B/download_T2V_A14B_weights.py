#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAN2.2-T2V-A14B (27B MOE) 模型权重下载脚本
此脚本用于下载WAN2.2-T2V-A14B模型的完整权重文件
包含高噪声专家模型和低噪声专家模型
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """安装必要的依赖包"""
    print("正在安装必要的依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub[cli]", "git-lfs"])
        print("依赖包安装完成!")
    except subprocess.CalledProcessError as e:
        print(f"依赖包安装失败: {e}")
        return False
    return True

def download_with_huggingface_cli():
    """使用Hugging Face CLI下载模型权重"""
    print("正在使用Hugging Face CLI下载T2V-A14B模型权重...")
    
    # 创建下载目录
    download_dir = Path("./T2V_A14B_weights")
    download_dir.mkdir(exist_ok=True)
    
    try:
        # 下载完整模型权重
        cmd = [
            "huggingface-cli", "download", 
            "Wan-AI/Wan2.2-T2V-A14B",
            "--local-dir", str(download_dir),
            "--local-dir-use-symlinks", "False"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        print("注意: T2V-A14B是27B参数的MOE模型，包含:")
        print("  - 高噪声专家模型: ~53GB")
        print("  - 低噪声专家模型: ~53GB") 
        print("  - T5编码器: ~4GB")
        print("  - VAE模型: ~1GB")
        print("  总计约: ~111GB")
        print()
        
        user_input = input("确认下载? 这将需要大量存储空间和时间 (y/N): ").strip().lower()
        if user_input != 'y':
            print("下载已取消")
            return False
            
        subprocess.check_call(cmd)
        print(f"T2V-A14B模型权重已下载到: {download_dir.absolute()}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"下载失败: {e}")
        return False

def download_with_git_lfs():
    """使用Git LFS下载当前仓库的权重文件"""
    print("正在使用Git LFS下载当前仓库的权重文件...")
    
    try:
        # 进入模型目录
        os.chdir("Wan2.2-T2V-A14B")
        
        # 拉取LFS文件
        subprocess.check_call(["git", "lfs", "pull"])
        print("Git LFS文件下载完成!")
        
        # 返回上级目录
        os.chdir("..")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Git LFS下载失败: {e}")
        os.chdir("..")  # 确保返回上级目录
        return False

def check_model_structure():
    """检查模型结构"""
    model_dir = Path("Wan2.2-T2V-A14B")
    if not model_dir.exists():
        print("错误: 未找到Wan2.2-T2V-A14B目录")
        return False
    
    print("=== T2V-A14B MOE模型结构 ===")
    
    # 检查高噪声模型
    high_noise_dir = model_dir / "high_noise_model"
    if high_noise_dir.exists():
        print(f"✓ 高噪声专家模型目录: {high_noise_dir}")
        index_file = high_noise_dir / "diffusion_pytorch_model.safetensors.index.json"
        if index_file.exists():
            print("  - 权重索引文件存在")
        else:
            print("  - ⚠️ 权重文件未下载")
    
    # 检查低噪声模型
    low_noise_dir = model_dir / "low_noise_model"
    if low_noise_dir.exists():
        print(f"✓ 低噪声专家模型目录: {low_noise_dir}")
        index_file = low_noise_dir / "diffusion_pytorch_model.safetensors.index.json"
        if index_file.exists():
            print("  - 权重索引文件存在")
        else:
            print("  - ⚠️ 权重文件未下载")
    
    # 检查其他组件
    components = {
        "T5编码器": "google/umt5-xxl",
        "配置文件": "configuration.json",
        "README": "README.md"
    }
    
    for name, path in components.items():
        if (model_dir / path).exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"⚠️ {name}: {path} 不存在")
    
    return True

def main():
    """主函数"""
    print("=" * 70)
    print("WAN2.2-T2V-A14B (27B MOE) 模型权重下载脚本")
    print("=" * 70)
    
    # 检查模型结构
    if not check_model_structure():
        print("请确保在正确的目录中运行此脚本")
        return False
    
    print("\n模型信息:")
    print("- 模型类型: Text-to-Video MOE (Mixture of Experts)")
    print("- 总参数: 27B (高噪声专家14B + 低噪声专家14B)")
    print("- 激活参数: 14B (每次推理只激活一个专家)")
    print("- 支持分辨率: 480P & 720P")
    print("- 存储需求: ~111GB")
    
    # 安装依赖
    if not install_requirements():
        print("依赖安装失败，退出")
        return False
    
    print("\n请选择下载方式:")
    print("1. 使用Hugging Face CLI下载到新目录 (推荐)")
    print("2. 使用Git LFS下载到当前仓库")
    
    while True:
        choice = input("\n请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            success = download_with_huggingface_cli()
            break
        elif choice == "2":
            success = download_with_git_lfs()
            break
        else:
            print("无效选择，请输入 1 或 2")
            continue
    
    if success:
        print("\n" + "=" * 70)
        print("T2V-A14B模型权重下载完成!")
        print("注意事项:")
        print("- 这是一个27B参数的MOE模型，需要大量GPU内存")
        print("- 推荐在具有80GB+显存的GPU上运行")
        print("- 可以使用多GPU并行推理")
        print("- 请在远程服务器或云平台上运行，不要在本地运行")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("模型权重下载失败!")
        print("请检查网络连接和存储空间")
        print("=" * 70)
    
    return success

if __name__ == "__main__":
    main()
