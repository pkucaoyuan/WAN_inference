#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAN2.2 推理环境安装脚本
此脚本用于自动安装WAN2.2推理所需的所有依赖
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        print("请升级到Python 3.8或更高版本")
        return False
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True

def install_pytorch():
    """安装PyTorch"""
    print("\n正在安装PyTorch...")
    try:
        # 检查是否已安装torch
        import torch
        version = torch.__version__
        print(f"✅ PyTorch已安装: {version}")
        
        # 检查版本是否满足要求
        major, minor = map(int, version.split('.')[:2])
        if major < 2 or (major == 2 and minor < 4):
            print(f"⚠️ PyTorch版本过低: {version}")
            print("正在升级到PyTorch >= 2.4.0...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch>=2.4.0", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ])
        return True
        
    except ImportError:
        print("正在安装PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch>=2.4.0", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
        return True
    except Exception as e:
        print(f"❌ PyTorch安装失败: {e}")
        return False

def install_wan_requirements():
    """安装WAN2.2依赖"""
    print("\n正在安装WAN2.2基础依赖...")
    
    req_file = Path("Wan2.2/requirements.txt")
    if not req_file.exists():
        print(f"❌ 未找到依赖文件: {req_file}")
        print("请确保已克隆Wan2.2推理代码")
        return False
    
    try:
        # 先安装其他依赖
        print("安装基础依赖包...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "diffusers", "transformers", "accelerate", 
            "sentencepiece", "protobuf", "pillow", 
            "opencv-python", "tqdm", "easydict"
        ])
        
        # 最后安装flash_attn（可能失败）
        print("尝试安装flash_attn...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "flash_attn"
            ])
            print("✅ flash_attn安装成功")
        except subprocess.CalledProcessError:
            print("⚠️ flash_attn安装失败，但不影响基本功能")
            print("如需flash_attn，请手动安装或使用预编译版本")
        
        print("✅ WAN2.2基础依赖安装完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def install_s2v_requirements():
    """安装语音转视频依赖（可选）"""
    print("\n是否安装语音转视频(S2V)功能依赖？")
    choice = input("输入 y 安装，其他键跳过: ").strip().lower()
    
    if choice != 'y':
        print("跳过S2V依赖安装")
        return True
    
    req_file = Path("Wan2.2/requirements_s2v.txt")
    if not req_file.exists():
        print(f"❌ 未找到S2V依赖文件: {req_file}")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(req_file)
        ])
        print("✅ S2V依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ S2V依赖安装失败: {e}")
        return False

def install_download_tools():
    """安装下载工具"""
    print("\n正在安装模型下载工具...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "huggingface_hub[cli]", "git-lfs"
        ])
        print("✅ 下载工具安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载工具安装失败: {e}")
        return False

def verify_installation():
    """验证安装"""
    print("\n=== 验证安装 ===")
    
    # 检查关键包
    packages = {
        "torch": "PyTorch",
        "diffusers": "Diffusers",
        "transformers": "Transformers", 
        "huggingface_hub": "HuggingFace Hub"
    }
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name}: 已安装")
        except ImportError:
            print(f"❌ {name}: 未安装")
    
    # 检查可选包
    try:
        import flash_attn
        print(f"✅ FlashAttention: 已安装")
    except ImportError:
        print(f"⚠️ FlashAttention: 未安装（可选）")

def main():
    """主函数"""
    print("=" * 60)
    print("WAN2.2 推理环境安装脚本")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        return False
    
    # 安装PyTorch
    if not install_pytorch():
        return False
    
    # 安装WAN依赖
    if not install_wan_requirements():
        return False
    
    # 安装S2V依赖（可选）
    if not install_s2v_requirements():
        return False
    
    # 安装下载工具
    if not install_download_tools():
        return False
    
    # 验证安装
    verify_installation()
    
    print("\n" + "=" * 60)
    print("🎉 环境安装完成！")
    print("\n下一步操作：")
    print("1. 下载模型权重:")
    print("   python download_model_weights.py  # 5B模型")
    print("   cd WAN2.2-27B && python download_T2V_A14B_weights.py  # 27B模型")
    print("\n2. 开始推理:")
    print("   cd Wan2.2")
    print("   python generate.py --task ti2v-5B --ckpt_dir ../model_weights ...")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    main()
