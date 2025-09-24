#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
演示如何通过参数控制注意力可视化的示例
"""

import torch
from wan.text2video import WanT2V


def main():
    """主函数 - 演示参数控制"""
    
    print("=== WAN模型参数控制注意力可视化示例 ===\n")
    
    # 初始化模型
    print("正在加载WAN模型...")
    try:
        model = WanT2V.from_pretrained(
            "pkucaoyuan/WAN2.2_T2V_A14B",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✅ 模型加载成功\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 测试提示词
    test_prompt = "A beautiful sunset over the ocean with waves"
    
    print("=" * 60)
    print("示例1: 禁用注意力可视化 (默认)")
    print("=" * 60)
    
    try:
        video, timing_info = model.generate(
            input_prompt=test_prompt,
            frame_num=8,
            size=(256, 256),
            sampling_steps=10,
            guide_scale=7.5,
            # enable_attention_visualization=False,  # 默认值，可以不写
            # attention_output_dir="attention_outputs"  # 默认值，可以不写
        )
        print("✅ 禁用可视化测试完成")
        print(f"   视频形状: {video.shape}")
        print("   注意: 没有生成注意力可视化文件")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("示例2: 启用注意力可视化")
    print("=" * 60)
    
    try:
        video, timing_info = model.generate(
            input_prompt=test_prompt,
            frame_num=8,
            size=(256, 256),
            sampling_steps=10,
            guide_scale=7.5,
            enable_attention_visualization=True,  # 启用注意力可视化
            attention_output_dir="my_attention_outputs"  # 自定义输出目录
        )
        print("✅ 启用可视化测试完成")
        print(f"   视频形状: {video.shape}")
        print("   注意: 已生成注意力可视化文件")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("示例3: 使用默认输出目录")
    print("=" * 60)
    
    try:
        video, timing_info = model.generate(
            input_prompt=test_prompt,
            frame_num=8,
            size=(256, 256),
            sampling_steps=10,
            guide_scale=7.5,
            enable_attention_visualization=True
            # attention_output_dir 使用默认值 "attention_outputs"
        )
        print("✅ 使用默认目录测试完成")
        print(f"   视频形状: {video.shape}")
        print("   注意: 使用默认输出目录 'attention_outputs'")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print("\n=== 参数说明 ===")
    print("enable_attention_visualization:")
    print("  - True:  启用注意力可视化，生成 average_cross_attention_map.png")
    print("  - False: 禁用注意力可视化，不生成任何可视化文件 (默认)")
    print()
    print("attention_output_dir:")
    print("  - 指定注意力可视化文件的输出目录")
    print("  - 默认值: 'attention_outputs'")
    print("  - 只在 enable_attention_visualization=True 时有效")


if __name__ == "__main__":
    main()
