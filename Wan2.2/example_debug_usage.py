#!/usr/bin/env python3
"""
调试代码使用示例
展示如何在视频生成过程中使用调试功能
"""

import os
import sys
import torch

# 添加Wan2.2到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'Wan2.2'))

from wan import WanT2V
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS

def example_debug_generation():
    """示例：使用调试功能生成视频"""
    
    # 配置参数
    task = "t2v-A14B"
    size = SIZE_CONFIGS["832*480"]
    ckpt_dir = "/path/to/your/checkpoints"  # 替换为实际路径
    frame_num = 49
    sample_steps = 20
    prompt = "A young woman in a red jacket is walking across a busy crosswalk in a modern city at night."
    seed = 42
    
    # 加载模型
    print("🔄 加载模型...")
    cfg = WAN_CONFIGS[task]
    model = WanT2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    )
    print("✅ 模型加载完成")
    
    # 生成视频（启用调试模式）
    print("\n🎬 开始生成视频（调试模式）...")
    video, timing_info = model.generate(
        input_prompt=prompt,
        size=size,
        frame_num=frame_num,
        sampling_steps=sample_steps,
        cfg_truncate_steps=3,
        cfg_truncate_high_noise_steps=5,
        seed=seed,
        enable_half_frame_generation=True,  # 启用帧数减半
        enable_debug=True,                  # 启用调试模式
        debug_output_dir="debug_analysis",  # 调试输出目录
        output_dir="generated_videos"
    )
    
    print("✅ 视频生成完成")
    print(f"📊 调试分析结果保存在: debug_analysis/")
    
    return video, timing_info

def analyze_debug_results(debug_dir="debug_analysis"):
    """分析调试结果"""
    import json
    from pathlib import Path
    
    debug_path = Path(debug_dir)
    if not debug_path.exists():
        print(f"❌ 调试目录不存在: {debug_dir}")
        return
    
    print(f"\n🔍 分析调试结果: {debug_dir}")
    
    # 查找所有JSON文件
    json_files = list(debug_path.glob("*.json"))
    print(f"📁 找到 {len(json_files)} 个调试文件")
    
    for json_file in json_files:
        print(f"\n📄 分析文件: {json_file.name}")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if 'shape' in data:
                print(f"  形状: {data['shape']}")
            if 'min' in data and 'max' in data:
                print(f"  数值范围: [{data['min']:.6f}, {data['max']:.6f}]")
            if 'mean' in data:
                print(f"  均值: {data['mean']:.6f}")
            if 'std' in data:
                print(f"  标准差: {data['std']:.6f}")
            if 'has_nan' in data:
                print(f"  包含NaN: {data['has_nan']}")
            if 'has_inf' in data:
                print(f"  包含Inf: {data['has_inf']}")
                
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
    
    # 查找PNG文件（可视化结果）
    png_files = list(debug_path.glob("*.png"))
    if png_files:
        print(f"\n📊 找到 {len(png_files)} 个可视化文件:")
        for png_file in png_files:
            print(f"  - {png_file.name}")

def debug_specific_issue():
    """调试特定问题：帧数补全质量下降"""
    
    print("\n🔍 调试帧数补全质量下降问题")
    
    # 1. 生成baseline（无帧数减半）
    print("\n1️⃣ 生成baseline视频...")
    # ... baseline生成代码 ...
    
    # 2. 生成帧数减半视频（启用调试）
    print("\n2️⃣ 生成帧数减半视频（调试模式）...")
    # ... 帧数减半生成代码 ...
    
    # 3. 对比分析
    print("\n3️⃣ 对比分析结果...")
    # ... 对比分析代码 ...

if __name__ == "__main__":
    print("🔍 调试代码使用示例")
    print("\n使用方法:")
    print("1. 修改ckpt_dir为实际路径")
    print("2. 运行: python example_debug_usage.py")
    print("3. 查看debug_analysis/目录中的分析结果")
    
    # 示例：分析现有调试结果
    if os.path.exists("debug_analysis"):
        analyze_debug_results("debug_analysis")
    else:
        print("\n💡 提示: 先运行视频生成以创建调试结果")
        print("   在generate.py中添加 --enable_debug 参数")
