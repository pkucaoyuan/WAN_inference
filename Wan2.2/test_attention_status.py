#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
测试注意力可视化状态的脚本
演示如何通过命令行确认是否启动可视化
"""

import argparse
import torch
from wan.text2video import WanT2V


def test_attention_visualization_status():
    """测试注意力可视化状态"""
    
    print("=== WAN模型注意力可视化状态测试 ===\n")
    
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
    test_prompt = "A beautiful sunset over the ocean"
    
    print("=" * 50)
    print("测试1: 禁用注意力可视化")
    print("=" * 50)
    
    try:
        video, timing_info = model.generate(
            input_prompt=test_prompt,
            frame_num=4,  # 很少的帧数用于快速测试
            size=(256, 256),
            sampling_steps=5,  # 很少的步数用于快速测试
            guide_scale=7.5,
            enable_attention_visualization=False,  # 禁用可视化
            attention_output_dir="test_no_attention"
        )
        print("✅ 禁用可视化测试完成")
        print(f"   视频形状: {video.shape}")
    except Exception as e:
        print(f"❌ 禁用可视化测试失败: {e}")
    
    print("\n" + "=" * 50)
    print("测试2: 启用注意力可视化")
    print("=" * 50)
    
    try:
        video, timing_info = model.generate(
            input_prompt=test_prompt,
            frame_num=4,  # 很少的帧数用于快速测试
            size=(256, 256),
            sampling_steps=5,  # 很少的步数用于快速测试
            guide_scale=7.5,
            enable_attention_visualization=True,  # 启用可视化
            attention_output_dir="test_with_attention"
        )
        print("✅ 启用可视化测试完成")
        print(f"   视频形状: {video.shape}")
    except Exception as e:
        print(f"❌ 启用可视化测试失败: {e}")
    
    print("\n=== 测试完成 ===")
    print("请检查以下目录:")
    print("- test_no_attention/: 应该为空或不存在")
    print("- test_with_attention/: 应该包含 average_cross_attention_map.png")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试WAN模型注意力可视化状态")
    parser.add_argument("--prompt", type=str, 
                       default="A beautiful sunset over the ocean",
                       help="测试提示词")
    parser.add_argument("--frames", type=int, default=4, help="测试帧数")
    parser.add_argument("--steps", type=int, default=5, help="测试步数")
    parser.add_argument("--enable", action="store_true", help="启用注意力可视化")
    parser.add_argument("--disable", action="store_true", help="禁用注意力可视化")
    
    args = parser.parse_args()
    
    if args.enable and args.disable:
        print("❌ 不能同时指定 --enable 和 --disable")
        return
    
    if args.enable or args.disable:
        # 单次测试
        print("=== WAN模型注意力可视化单次测试 ===\n")
        
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
        
        # 确定可视化状态
        enable_attention = args.enable
        output_dir = "test_attention_enabled" if enable_attention else "test_attention_disabled"
        
        print(f"🔍 注意力可视化: {'启用' if enable_attention else '禁用'}")
        print(f"📁 输出目录: {output_dir}")
        print(f"📝 提示词: {args.prompt}")
        print(f"🎬 帧数: {args.frames}")
        print(f"🔄 步数: {args.steps}")
        print()
        
        try:
            video, timing_info = model.generate(
                input_prompt=args.prompt,
                frame_num=args.frames,
                size=(256, 256),
                sampling_steps=args.steps,
                guide_scale=7.5,
                enable_attention_visualization=enable_attention,
                attention_output_dir=output_dir
            )
            print("✅ 测试完成")
            print(f"   视频形状: {video.shape}")
            print(f"   输出目录: {output_dir}")
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 完整测试
        test_attention_visualization_status()


if __name__ == "__main__":
    main()
