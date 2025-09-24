#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
WAN模型注意力可视化测试脚本
用于测试和展示cross attention权重的可视化功能
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from wan.text2video import WanT2V
from wan.attention_visualizer import AttentionVisualizer, create_attention_visualization_dir
from transformers import T5Tokenizer


class AttentionAwareWanT2V(WanT2V):
    """支持注意力可视化的WAN T2V模型"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights_history = []
        self.current_step = 0
        
    def generate_with_attention_visualization(self, 
                                            prompt: str,
                                            num_frames: int = 16,
                                            height: int = 256,
                                            width: int = 256,
                                            num_inference_steps: int = 25,
                                            guidance_scale: float = 7.5,
                                            save_attention: bool = True,
                                            output_dir: str = "attention_outputs"):
        """生成视频并记录注意力权重"""
        
        # 创建输出目录
        if save_attention:
            output_dir = create_attention_visualization_dir(output_dir)
            print(f"注意力可视化将保存到: {output_dir}")
        
        # 重置注意力历史
        self.attention_weights_history = []
        self.current_step = 0
        
        # 获取tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        tokens = tokenizer.tokenize(prompt)
        
        print(f"输入提示: {prompt}")
        print(f"Token数量: {len(tokens)}")
        print(f"Token列表: {tokens}")
        
        # 修改模型以支持注意力权重返回
        self._enable_attention_capture()
        
        try:
            # 生成视频
            video, timing_info = self.generate(
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            if save_attention and self.attention_weights_history:
                # 创建可视化
                self._create_attention_visualizations(tokens, output_dir)
                
                # 创建分析报告
                self._create_analysis_report(tokens, output_dir)
            
            return video, timing_info
            
        finally:
            # 恢复原始模型
            self._disable_attention_capture()
    
    def _enable_attention_capture(self):
        """启用注意力权重捕获"""
        # 这里我们需要修改模型的forward方法来捕获注意力权重
        # 由于模型结构复杂，我们使用hook的方式
        self.attention_hooks = []
        
        def create_attention_hook(module_name):
            def hook_fn(module, input, output):
                if hasattr(module, 'cross_attn') and hasattr(module.cross_attn, 'forward'):
                    # 尝试获取注意力权重
                    try:
                        # 这里需要修改cross_attn的forward调用来返回权重
                        pass
                    except:
                        pass
            return hook_fn
        
        # 注册hook到所有attention block
        for name, module in self.unet.named_modules():
            if 'attention_block' in name.lower() or 'cross_attn' in name.lower():
                hook = module.register_forward_hook(create_attention_hook(name))
                self.attention_hooks.append(hook)
    
    def _disable_attention_capture(self):
        """禁用注意力权重捕获"""
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks = []
    
    def _create_attention_visualizations(self, tokens, output_dir):
        """创建注意力可视化"""
        if not self.attention_weights_history:
            print("没有捕获到注意力权重数据")
            return
        
        print(f"创建注意力可视化，共 {len(self.attention_weights_history)} 步...")
        
        # 创建可视化器
        visualizer = AttentionVisualizer(None, None, self.device)
        
        # 创建单步可视化
        for i, attention_weights in enumerate(self.attention_weights_history):
            save_path = os.path.join(output_dir, f"attention_step_{i:03d}.png")
            visualizer.visualize_attention_step(
                attention_weights, tokens, i, save_path
            )
        
        # 创建蒙太奇
        montage_path = os.path.join(output_dir, "attention_montage.png")
        visualizer.create_attention_montage(
            self.attention_weights_history, tokens, montage_path
        )
        
        # 创建动画
        animation_path = os.path.join(output_dir, "attention_animation.gif")
        visualizer.create_attention_animation(
            self.attention_weights_history, tokens, animation_path
        )
        
        print(f"注意力可视化已保存到: {output_dir}")
    
    def _create_analysis_report(self, tokens, output_dir):
        """创建分析报告"""
        if not self.attention_weights_history:
            return
        
        visualizer = AttentionVisualizer(None, None, self.device)
        
        # 分析注意力模式
        analysis = visualizer.analyze_attention_patterns(
            self.attention_weights_history, tokens
        )
        
        # 保存报告
        report_path = os.path.join(output_dir, "attention_analysis_report.md")
        visualizer.save_analysis_report(analysis, report_path)
        
        print(f"注意力分析报告已保存到: {report_path}")


def test_attention_visualization():
    """测试注意力可视化功能"""
    
    # 测试参数
    test_cases = [
        {
            "prompt": "A beautiful sunset over the ocean with waves crashing on the shore",
            "num_frames": 8,
            "height": 256,
            "width": 256,
            "num_inference_steps": 10,
            "guidance_scale": 7.5
        },
        {
            "prompt": "A cat playing with a ball of yarn in slow motion",
            "num_frames": 8,
            "height": 256,
            "width": 256,
            "num_inference_steps": 10,
            "guidance_scale": 7.5
        }
    ]
    
    print("开始测试WAN模型注意力可视化功能...")
    
    # 初始化模型
    try:
        model = AttentionAwareWanT2V.from_pretrained(
            "pkucaoyuan/WAN2.2_T2V_A14B",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 运行测试用例
    for i, test_case in enumerate(test_cases):
        print(f"\n=== 测试用例 {i+1}/{len(test_cases)} ===")
        print(f"提示: {test_case['prompt']}")
        
        try:
            video, timing_info = model.generate_with_attention_visualization(
                **test_case,
                save_attention=True,
                output_dir=f"attention_test_case_{i+1}"
            )
            
            print(f"视频生成成功: {video.shape}")
            print(f"生成时间: {timing_info}")
            
        except Exception as e:
            print(f"测试用例 {i+1} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n注意力可视化测试完成!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="WAN模型注意力可视化测试")
    parser.add_argument("--prompt", type=str, 
                       default="A beautiful sunset over the ocean with waves crashing on the shore",
                       help="输入提示文本")
    parser.add_argument("--num_frames", type=int, default=8, help="视频帧数")
    parser.add_argument("--height", type=int, default=256, help="视频高度")
    parser.add_argument("--width", type=int, default=256, help="视频宽度")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="引导尺度")
    parser.add_argument("--output_dir", type=str, default="attention_outputs", help="输出目录")
    parser.add_argument("--test_mode", action="store_true", help="运行测试模式")
    
    args = parser.parse_args()
    
    if args.test_mode:
        test_attention_visualization()
    else:
        # 单次生成模式
        print("初始化WAN模型...")
        try:
            model = AttentionAwareWanT2V.from_pretrained(
                "pkucaoyuan/WAN2.2_T2V_A14B",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            print("开始生成视频并可视化注意力...")
            video, timing_info = model.generate(
                input_prompt=args.prompt,
                frame_num=args.num_frames,
                size=(args.width, args.height),
                sampling_steps=args.num_inference_steps,
                guide_scale=args.guidance_scale,
                enable_attention_visualization=True,
                attention_output_dir=args.output_dir
            )
            
            print(f"视频生成完成: {video.shape}")
            print(f"生成时间: {timing_info}")
            print(f"注意力可视化已保存到: {args.output_dir}")
            
        except Exception as e:
            print(f"生成失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
