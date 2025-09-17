#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAN2.2-5B 模型权重下载脚本
此脚本用于下载WAN2.2-5B模型的完整权重文件
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
    print("正在使用Hugging Face CLI下载模型权重...")
    
    # 创建下载目录
    download_dir = Path("./model_weights")
    download_dir.mkdir(exist_ok=True)
    
    try:
        # 下载模型权重
        cmd = [
            "huggingface-cli", "download", 
            "Wan-AI/Wan2.2-TI2V-5B",
            "--local-dir", str(download_dir),
            "--local-dir-use-symlinks", "False"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        print(f"模型权重已下载到: {download_dir.absolute()}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"下载失败: {e}")
        return False

def download_with_git_lfs():
    """使用Git LFS下载当前仓库的权重文件"""
    print("正在使用Git LFS下载当前仓库的权重文件...")
    
    try:
        # 拉取LFS文件
        subprocess.check_call(["git", "lfs", "pull"])
        print("Git LFS文件下载完成!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Git LFS下载失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("WAN2.2-5B 模型权重下载脚本")
    print("=" * 60)
    
    # 检查当前目录
    if not Path("README.md").exists():
        print("警告: 当前目录似乎不是WAN2.2-5B模型仓库")
        print("请确保在正确的目录中运行此脚本")
    
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
        print("\n" + "=" * 60)
        print("模型权重下载完成!")
        print("注意: 请在远程服务器或云平台上运行模型，不要在本地运行")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("模型权重下载失败!")
        print("请检查网络连接和权限设置")
        print("=" * 60)
    
    return success

if __name__ == "__main__":
    main()

