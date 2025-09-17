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
import threading
import time
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
    """使用Hugging Face CLI下载模型权重（后台模式）"""
    print("正在启动T2V-A14B模型后台下载...")
    
    # 创建下载目录
    download_dir = Path("./T2V_A14B_weights")
    download_dir.mkdir(exist_ok=True)
    log_file = download_dir / "download.log"
    
    print("注意: T2V-A14B是27B参数的MOE模型，包含:")
    print("  - 高噪声专家模型: ~53GB")
    print("  - 低噪声专家模型: ~53GB") 
    print("  - T5编码器: ~4GB")
    print("  - VAE模型: ~1GB")
    print("  总计约: ~111GB")
    print()
    
    user_input = input("确认后台下载? 这将需要大量存储空间和时间 (y/N): ").strip().lower()
    if user_input != 'y':
        print("下载已取消")
        return False
    
    def background_download():
        """后台下载函数"""
        try:
            # 下载完整模型权重
            cmd = [
                "huggingface-cli", "download", 
                "Wan-AI/Wan2.2-T2V-A14B",
                "--local-dir", str(download_dir),
                "--local-dir-use-symlinks", "False"
            ]
            
            print(f"后台执行命令: {' '.join(cmd)}")
            print(f"日志文件: {log_file}")
            
            # 重定向输出到日志文件
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"开始下载时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"执行命令: {' '.join(cmd)}\n")
                f.write("模型信息:\n")
                f.write("  - 高噪声专家模型: ~53GB\n")
                f.write("  - 低噪声专家模型: ~53GB\n")
                f.write("  - T5编码器: ~4GB\n")
                f.write("  - VAE模型: ~1GB\n")
                f.write("  - 总计约: ~111GB\n")
                f.write("=" * 50 + "\n")
                f.flush()
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # 实时写入日志
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                
                process.wait()
                
                if process.returncode == 0:
                    f.write(f"\n下载完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"T2V-A14B模型权重已下载到: {download_dir.absolute()}\n")
                    print(f"\n✅ T2V-A14B下载完成！权重保存在: {download_dir.absolute()}")
                else:
                    f.write(f"\n下载失败，返回码: {process.returncode}\n")
                    print(f"\n❌ T2V-A14B下载失败，请检查日志: {log_file}")
                    
        except Exception as e:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n下载异常: {str(e)}\n")
            print(f"❌ T2V-A14B下载异常: {e}")
    
    # 启动后台线程
    download_thread = threading.Thread(target=background_download, daemon=True)
    download_thread.start()
    
    print(f"📥 T2V-A14B后台下载已启动")
    print(f"📄 实时日志: {log_file}")
    print(f"💡 您可以使用以下命令监控下载进度:")
    print(f"   tail -f {log_file}")
    print(f"🔄 下载线程已在后台运行，您可以继续其他操作")
    
    return True

def download_with_git_lfs():
    """使用Git LFS下载当前仓库的权重文件（后台模式）"""
    print("正在启动Git LFS后台下载...")
    
    log_file = Path("./git_lfs_download.log")
    
    def background_git_lfs():
        """后台Git LFS下载函数"""
        original_dir = os.getcwd()
        try:
            # 进入模型目录
            os.chdir("Wan2.2-T2V-A14B")
            
            cmd = ["git", "lfs", "pull"]
            print(f"后台执行命令: {' '.join(cmd)}")
            print(f"日志文件: {log_file}")
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"开始下载时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"执行命令: {' '.join(cmd)}\n")
                f.write(f"工作目录: {os.getcwd()}\n")
                f.write("=" * 50 + "\n")
                f.flush()
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # 实时写入日志
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                
                process.wait()
                
                if process.returncode == 0:
                    f.write(f"\n下载完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("Git LFS文件下载完成!\n")
                    print(f"\n✅ Git LFS下载完成！")
                else:
                    f.write(f"\n下载失败，返回码: {process.returncode}\n")
                    print(f"\n❌ Git LFS下载失败，请检查日志: {log_file}")
                    
        except Exception as e:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n下载异常: {str(e)}\n")
            print(f"❌ Git LFS下载异常: {e}")
        finally:
            # 确保返回原目录
            os.chdir(original_dir)
    
    # 启动后台线程
    download_thread = threading.Thread(target=background_git_lfs, daemon=True)
    download_thread.start()
    
    print(f"📥 Git LFS后台下载已启动")
    print(f"📄 实时日志: {log_file}")
    print(f"💡 您可以使用以下命令监控下载进度:")
    print(f"   tail -f {log_file}")
    print(f"🔄 下载线程已在后台运行，您可以继续其他操作")
    
    return True

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
