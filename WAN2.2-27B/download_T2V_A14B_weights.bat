@echo off
REM WAN2.2-T2V-A14B (27B MOE) 模型权重下载批处理脚本 (Windows)
REM 此脚本用于在Windows系统上下载WAN2.2-T2V-A14B模型权重

echo ========================================================================
echo WAN2.2-T2V-A14B (27B MOE) 模型权重下载脚本 (Windows)
echo ========================================================================

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python
    pause
    exit /b 1
)

echo 模型信息:
echo - 模型类型: Text-to-Video MOE (Mixture of Experts)
echo - 总参数: 27B (高噪声专家14B + 低噪声专家14B)
echo - 激活参数: 14B (每次推理只激活一个专家)
echo - 支持分辨率: 480P ^& 720P
echo - 存储需求: ~111GB
echo.

REM 安装依赖包
echo 正在安装必要的依赖包...
pip install huggingface_hub[cli] git-lfs

if errorlevel 1 (
    echo 依赖包安装失败
    pause
    exit /b 1
)

echo 依赖包安装完成!
echo.

REM 检查模型目录
if not exist "Wan2.2-T2V-A14B" (
    echo 错误: 未找到Wan2.2-T2V-A14B目录
    echo 请确保在正确的目录中运行此脚本
    pause
    exit /b 1
)

echo 检测到T2V-A14B模型目录结构:
if exist "Wan2.2-T2V-A14B\high_noise_model" echo   ✓ 高噪声专家模型目录
if exist "Wan2.2-T2V-A14B\low_noise_model" echo   ✓ 低噪声专家模型目录
if exist "Wan2.2-T2V-A14B\google\umt5-xxl" echo   ✓ T5编码器目录
echo.

echo 请选择下载方式:
echo 1. 使用Hugging Face CLI下载到新目录 (推荐)
echo 2. 使用Git LFS下载到当前仓库
echo.

set /p choice="请输入选择 (1 或 2): "

if "%choice%"=="1" (
    echo 正在使用Hugging Face CLI下载T2V-A14B模型权重...
    echo.
    echo 警告: 此模型非常大！
    echo   - 高噪声专家模型: ~53GB
    echo   - 低噪声专家模型: ~53GB
    echo   - T5编码器: ~4GB
    echo   - VAE模型: ~1GB
    echo   - 总计约: ~111GB
    echo.
    
    set /p confirm="确认下载? 这将需要大量存储空间和时间 (y/N): "
    if /i not "%confirm%"=="y" (
        echo 下载已取消
        pause
        exit /b 0
    )
    
    REM 创建下载目录
    if not exist "T2V_A14B_weights" mkdir T2V_A14B_weights
    
    huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./T2V_A14B_weights --local-dir-use-symlinks False
    if errorlevel 1 (
        echo 下载失败!
        pause
        exit /b 1
    )
    echo T2V-A14B模型权重已下载到: %cd%\T2V_A14B_weights
    
) else if "%choice%"=="2" (
    echo 正在使用Git LFS下载当前仓库的权重文件...
    cd Wan2.2-T2V-A14B
    git lfs pull
    if errorlevel 1 (
        echo Git LFS下载失败!
        cd ..
        pause
        exit /b 1
    )
    cd ..
    echo Git LFS文件下载完成!
    
) else (
    echo 无效选择
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo T2V-A14B模型权重下载完成!
echo.
echo 注意事项:
echo - 这是一个27B参数的MOE模型，需要大量GPU内存
echo - 推荐在具有80GB+显存的GPU上运行
echo - 可以使用多GPU并行推理
echo - 请在远程服务器或云平台上运行，不要在本地运行
echo ========================================================================
pause
