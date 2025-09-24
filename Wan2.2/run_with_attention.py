#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
命令行工具：通过参数控制注意力可视化
"""

import argparse
import torch
from wan.text2video import WanT2V


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="WAN模型视频生成 - 支持注意力可视化控制")
    
    # 基本参数
    parser.add_argument("--prompt", type=str, required=True, help="输入提示词")
    parser.add_argument("--frames", type=int, default=16, help="视频帧数")
    parser.add_argument("--width", type=int, default=256, help="视频宽度")
    parser.add_argument("--height", type=int, default=256, help="视频高度")
    parser.add_argument("--steps", type=int, default=25, help="去噪步数")
    parser.add_argument("--guidance", type=float, default=7.5, help="CFG引导尺度")
    
    # 注意力可视化参数
    parser.add_argument("--enable-attention", action="store_true", 
                       help="启用注意力可视化")
    parser.add_argument("--attention-dir", type=str, default="attention_outputs",
                       help="注意力可视化输出目录")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=-1, help="随机种子")
    parser.add_argument("--output-dir", type=str, help="视频输出目录")
    
    args = parser.parse_args()
    
    print("=== WAN模型视频生成 ===")
    print(f"📝 提示词: {args.prompt}")
    print(f"🎬 帧数: {args.frames}")
    print(f"📐 尺寸: {args.width}x{args.height}")
    print(f"🔄 步数: {args.steps}")
    print(f"🎯 引导尺度: {args.guidance}")
    print(f"🔍 注意力可视化: {'启用' if args.enable_attention else '禁用'}")
    if args.enable_attention:
        print(f"📁 注意力输出目录: {args.attention_dir}")
    print()
    
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
    
    # 生成视频
    try:
        video, timing_info = model.generate(
            input_prompt=args.prompt,
            frame_num=args.frames,
            size=(args.width, args.height),
            sampling_steps=args.steps,
            guide_scale=args.guidance,
            seed=args.seed,
            enable_attention_visualization=args.enable_attention,
            attention_output_dir=args.attention_dir
        )
        
        print("✅ 视频生成完成!")
        print(f"   视频形状: {video.shape}")
        print(f"   生成时间: {timing_info}")
        
        if args.enable_attention:
            print(f"   注意力可视化已保存到: {args.attention_dir}/")
            print("   生成文件:")
            print("   - average_cross_attention_map.png")
            print("   - attention_analysis_report.md")
        else:
            print("   注意: 未启用注意力可视化")
            
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
