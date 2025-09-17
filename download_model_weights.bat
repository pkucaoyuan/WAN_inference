@echo off
REM WAN2.2-5B 模型权重下载批处理脚本 (Windows)
REM 此脚本用于在Windows系统上下载WAN2.2-5B模型权重

echo ========================================
echo WAN2.2-5B 模型权重下载脚本 (Windows)
echo ========================================

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python
    pause
    exit /b 1
)

REM 安装依赖包
echo 正在安装必要的依赖包...
pip install huggingface_hub[cli] git-lfs

if errorlevel 1 (
    echo 依赖包安装失败
    pause
    exit /b 1
)

echo 依赖包安装完成!

REM 创建下载目录
if not exist "model_weights" mkdir model_weights

echo.
echo 请选择下载方式:
echo 1. 使用Hugging Face CLI下载到新目录 (推荐)
echo 2. 使用Git LFS下载到当前仓库
echo.

set /p choice="请输入选择 (1 或 2): "

if "%choice%"=="1" (
    echo 正在使用Hugging Face CLI下载模型权重...
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./model_weights --local-dir-use-symlinks False
    if errorlevel 1 (
        echo 下载失败!
        pause
        exit /b 1
    )
    echo 模型权重已下载到: %cd%\model_weights
) else if "%choice%"=="2" (
    echo 正在使用Git LFS下载当前仓库的权重文件...
    git lfs pull
    if errorlevel 1 (
        echo Git LFS下载失败!
        pause
        exit /b 1
    )
    echo Git LFS文件下载完成!
) else (
    echo 无效选择
    pause
    exit /b 1
)

echo.
echo ========================================
echo 模型权重下载完成!
echo 注意: 请在远程服务器或云平台上运行模型，不要在本地运行
echo ========================================
pause

