#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
WAN模型注意力可视化使用示例
展示如何使用注意力可视化功能
"""

import torch
from wan.text2video import WanT2V


def main():
    """主函数 - 演示注意力可视化功能"""
    
    print("=== WAN模型注意力可视化示例 ===")
    
    # 初始化模型
    print("正在加载WAN模型...")
    try:
        model = WanT2V.from_pretrained(
            "pkucaoyuan/WAN2.2_T2V_A14B",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 测试提示词
    test_prompts = [
        "A beautiful sunset over the ocean with waves crashing on the shore",
        "A cat playing with a ball of yarn in slow motion",
        "A person walking through a forest in autumn"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n=== 测试 {i+1}/{len(test_prompts)} ===")
        print(f"提示词: {prompt}")
        
        try:
            # 使用注意力可视化生成视频
            video, timing_info = model.generate_with_attention_visualization(
                prompt=prompt,
                num_frames=8,  # 减少帧数以加快测试
                height=256,
                width=256,
                num_inference_steps=10,  # 减少步数以加快测试
                guidance_scale=7.5,
                output_dir=f"attention_demo_{i+1}"
            )
            
            print(f"✅ 视频生成成功!")
            print(f"   视频形状: {video.shape}")
            print(f"   生成时间: {timing_info}")
            print(f"   注意力可视化已保存到: attention_demo_{i+1}/")
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== 注意力可视化示例完成 ===")
    print("请查看生成的attention_demo_*目录中的可视化结果:")
    print("- attention_step_*.png: 每步的注意力权重热力图")
    print("- attention_montage.png: 所有步骤的蒙太奇图像")
    print("- attention_animation.gif: 注意力权重动画")
    print("- attention_analysis_report.md: 详细的分析报告")


if __name__ == "__main__":
    main()
