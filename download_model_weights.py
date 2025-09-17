#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAN2.2-5B 模型权重下载脚本
此脚本用于下载WAN2.2-5B模型的完整权重文件
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
    print("正在启动后台下载...")
    
    # 创建下载目录
    download_dir = Path("./model_weights")
    download_dir.mkdir(exist_ok=True)
    log_file = download_dir / "download.log"
    
    def background_download():
        """后台下载函数"""
        try:
            # 下载模型权重
            cmd = [
                "huggingface-cli", "download", 
                "Wan-AI/Wan2.2-TI2V-5B",
                "--local-dir", str(download_dir),
                "--local-dir-use-symlinks", "False"
            ]
            
            print(f"后台执行命令: {' '.join(cmd)}")
            print(f"日志文件: {log_file}")
            
            # 重定向输出到日志文件
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"开始下载时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"执行命令: {' '.join(cmd)}\n")
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
                    f.write(f"模型权重已下载到: {download_dir.absolute()}\n")
                    print(f"\n✅ 下载完成！权重保存在: {download_dir.absolute()}")
                else:
                    f.write(f"\n下载失败，返回码: {process.returncode}\n")
                    print(f"\n❌ 下载失败，请检查日志: {log_file}")
                    
        except Exception as e:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n下载异常: {str(e)}\n")
            print(f"❌ 下载异常: {e}")
    
    # 启动后台线程
    download_thread = threading.Thread(target=background_download, daemon=True)
    download_thread.start()
    
    print(f"📥 后台下载已启动")
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
        try:
            cmd = ["git", "lfs", "pull"]
            print(f"后台执行命令: {' '.join(cmd)}")
            print(f"日志文件: {log_file}")
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"开始下载时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"执行命令: {' '.join(cmd)}\n")
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
    
    # 启动后台线程
    download_thread = threading.Thread(target=background_git_lfs, daemon=True)
    download_thread.start()
    
    print(f"📥 Git LFS后台下载已启动")
    print(f"📄 实时日志: {log_file}")
    print(f"💡 您可以使用以下命令监控下载进度:")
    print(f"   tail -f {log_file}")
    print(f"🔄 下载线程已在后台运行，您可以继续其他操作")
    
    return True

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

