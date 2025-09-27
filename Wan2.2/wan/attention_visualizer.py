# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
注意力可视化模块
用于捕获和可视化WAN模型中的cross attention权重
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional, Tuple
import os
from PIL import Image
import cv2


class AttentionHook:
    """注意力权重捕获Hook"""
    
    def __init__(self, module: nn.Module, layer_name: str):
        self.module = module
        self.layer_name = layer_name
        self.attention_weights = []
        self.hook_handle = None
        
    def register_hook(self):
        """注册hook"""
        if hasattr(self.module, 'cross_attn'):
            self.hook_handle = self.module.cross_attn.register_forward_hook(self._hook_fn)
        else:
            print(f"Warning: Module {self.layer_name} has no cross_attn attribute")
    
    def unregister_hook(self):
        """取消注册hook"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def _hook_fn(self, module, input, output):
        """Hook函数，捕获attention权重"""
        # 这里我们需要修改flash_attention来返回attention权重
        # 暂时先记录输入输出信息
        self.attention_weights.append({
            'input_shape': [x.shape for x in input],
            'output_shape': output.shape,
            'step': len(self.attention_weights)
        })


class CustomFlashAttention(nn.Module):
    """自定义Flash Attention，支持返回attention权重"""
    
    def __init__(self, original_attention):
        super().__init__()
        self.original_attention = original_attention
        self.attention_weights = None
        
    def forward(self, q, k, v, k_lens=None, return_weights=False):
        """前向传播，可选择返回attention权重"""
        if return_weights:
            # 使用标准attention计算权重
            b, n, d = q.size(0), q.size(2), q.size(3)
            scale = 1.0 / np.sqrt(d)
            
            # 计算attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attention_weights = torch.softmax(scores, dim=-1)
            
            # 计算输出
            output = torch.matmul(attention_weights, v)
            
            # 存储权重
            self.attention_weights = attention_weights.detach().cpu()
            
            return output, attention_weights
        else:
            return self.original_attention(q, k, v, k_lens=k_lens)


class AttentionVisualizer:
    """注意力可视化器"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.attention_data = {}
        self.hooks = []
        
    def register_attention_hooks(self, layer_names: List[str]):
        """注册注意力hook"""
        for name, module in self.model.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                hook = AttentionHook(module, name)
                hook.register_hook()
                self.hooks.append(hook)
                print(f"Registered hook for layer: {name}")
    
    def unregister_hooks(self):
        """取消所有hook"""
        for hook in self.hooks:
            hook.unregister_hook()
        self.hooks = []
    
    def extract_tokens(self, prompt: str) -> List[str]:
        """提取token"""
        if hasattr(self.tokenizer, 'tokenize'):
            tokens = self.tokenizer.tokenize(prompt)
        else:
            # 简单的tokenization
            tokens = prompt.split()
        return tokens
    
    def visualize_attention_step(self, 
                                attention_weights: torch.Tensor,
                                tokens: List[str],
                                step: int,
                                save_path: str = None,
                                title: str = None) -> np.ndarray:
        """可视化单步注意力权重"""
        
        # attention_weights shape: [batch, num_heads, seq_len_q, seq_len_k]
        if attention_weights.dim() == 4:
            # 平均所有头
            attention_weights = attention_weights.mean(dim=1)  # [batch, seq_len_q, seq_len_k]
        
        if attention_weights.dim() == 3:
            # 取第一个batch
            attention_weights = attention_weights[0]  # [seq_len_q, seq_len_k]
        
        # 转换为numpy
        weights = attention_weights.detach().cpu().numpy()
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 优化可视化效果
        # 使用更好的颜色映射和对比度
        im = ax.imshow(weights, cmap='viridis', aspect='auto', interpolation='nearest')
        
        # 设置坐标轴
        ax.set_xlabel('Key Tokens (Context)', fontsize=12)
        ax.set_ylabel('Query Tokens (Image)', fontsize=12)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Cross Attention Weights - Step {step}', fontsize=14, fontweight='bold')
        
        # 添加统计信息到标题
        min_val, max_val = weights.min(), weights.max()
        mean_val = weights.mean()
        ax.set_title(f'{title}\nRange: {min_val:.4f} - {max_val:.4f}, Mean: {mean_val:.4f}', 
                    fontsize=12, fontweight='bold')
        
        # 设置tick labels
        # 注意：tokens数量可能与attention权重的context维度不匹配
        context_len = weights.shape[1]  # attention权重的context维度
        
        if len(tokens) == context_len:
            # 如果tokens数量匹配，直接使用
            if len(tokens) <= 50:
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right')
            else:
                ax.set_xticks(range(0, len(tokens), max(1, len(tokens)//20)))
                ax.set_xticklabels([tokens[i] for i in range(0, len(tokens), max(1, len(tokens)//20))], 
                                  rotation=45, ha='right')
        else:
            # 如果tokens数量不匹配，使用位置索引
            print(f"⚠️ Tokens数量({len(tokens)})与attention权重context维度({context_len})不匹配，使用位置索引")
            if context_len <= 50:
                ax.set_xticks(range(context_len))
                ax.set_xticklabels([f"T{i}" for i in range(context_len)], rotation=45, ha='right')
            else:
                step = max(1, context_len // 20)
                ax.set_xticks(range(0, context_len, step))
                ax.set_xticklabels([f"T{i}" for i in range(0, context_len, step)], rotation=45, ha='right')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attention Weight', fontsize=12, fontweight='bold')
        
        # 设置颜色条刻度
        cbar.ax.tick_params(labelsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention visualization saved to: {save_path}")
        
        # 转换为numpy数组用于返回
        fig.canvas.draw()
        # 修复matplotlib版本兼容性问题
        try:
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        except AttributeError:
            # 新版本matplotlib使用buffer_rgba()
            img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img_array = img_array[:, :, :3]  # 只取RGB通道
        else:
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img_array
    
    def create_attention_montage(self, 
                                attention_weights_list: List[torch.Tensor],
                                tokens: List[str],
                                save_path: str = None) -> np.ndarray:
        """创建注意力权重的蒙太奇图像"""
        
        num_steps = len(attention_weights_list)
        cols = min(4, num_steps)
        rows = (num_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, attention_weights in enumerate(attention_weights_list):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # 处理attention权重
            if attention_weights.dim() == 4:
                attention_weights = attention_weights.mean(dim=1)
            if attention_weights.dim() == 3:
                attention_weights = attention_weights[0]
            
            weights = attention_weights.detach().cpu().numpy()
            
            # 绘制热力图
            im = ax.imshow(weights, cmap='Blues', aspect='auto')
            ax.set_title(f'Step {i}', fontsize=10)
            ax.set_xlabel('Context Tokens' if row == rows - 1 else '')
            ax.set_ylabel('Image Tokens' if col == 0 else '')
            
            # 设置tick labels（简化）
            if len(tokens) <= 20:
                ax.set_xticks(range(0, len(tokens), max(1, len(tokens)//10)))
                ax.set_xticklabels([tokens[j] for j in range(0, len(tokens), max(1, len(tokens)//10))], 
                                  rotation=45, ha='right', fontsize=8)
        
        # 隐藏多余的子图
        for i in range(num_steps, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention montage saved to: {save_path}")
        
        # 转换为numpy数组
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img_array
    
    def create_attention_sequence(self, 
                                 attention_weights_list: List[torch.Tensor],
                                 tokens: List[str],
                                 save_dir: str = None) -> List[np.ndarray]:
        """创建注意力权重的序列图像"""
        
        frames = []
        for i, attention_weights in enumerate(attention_weights_list):
            frame = self.visualize_attention_step(attention_weights, tokens, i)
            frames.append(frame)
            
            if save_dir:
                # 保存单步图像
                step_path = os.path.join(save_dir, f"attention_step_{i:03d}.png")
                Image.fromarray(frame).save(step_path)
        
        if save_dir:
            print(f"Attention sequence images saved to: {save_dir}")
        
        return frames
    
    def analyze_attention_patterns(self, 
                                 attention_weights_list: List[torch.Tensor],
                                 tokens: List[str]) -> Dict:
        """分析注意力模式"""
        
        analysis = {
            'token_importance': {},
            'step_evolution': [],
            'attention_entropy': [],
            'max_attention_tokens': []
        }
        
        for step, attention_weights in enumerate(attention_weights_list):
            # 处理权重
            if attention_weights.dim() == 4:
                attention_weights = attention_weights.mean(dim=1)
            if attention_weights.dim() == 3:
                attention_weights = attention_weights[0]
            
            weights = attention_weights.detach().cpu().numpy()
            
            # 计算每个token的平均注意力
            token_attention = weights.mean(axis=0)  # 平均所有query位置
            
            # 更新token重要性
            for i, token in enumerate(tokens):
                if token not in analysis['token_importance']:
                    analysis['token_importance'][token] = []
                analysis['token_importance'][token].append(token_attention[i])
            
            # 计算注意力熵
            entropy = -np.sum(weights * np.log(weights + 1e-8), axis=-1).mean()
            analysis['attention_entropy'].append(entropy)
            
            # 找到最大注意力的token
            max_token_idx = np.argmax(token_attention)
            analysis['max_attention_tokens'].append(tokens[max_token_idx])
            
            # 记录步骤演化
            analysis['step_evolution'].append({
                'step': step,
                'max_attention': float(token_attention.max()),
                'mean_attention': float(token_attention.mean()),
                'attention_std': float(token_attention.std())
            })
        
        return analysis
    
    def save_analysis_report(self, analysis: Dict, save_path: str):
        """保存分析报告"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("# WAN Cross Attention Analysis Report\n\n")
            
            # Token重要性分析
            f.write("## Token Importance Analysis\n\n")
            f.write("| Token | Mean Attention | Std Attention | Max Attention |\n")
            f.write("|-------|----------------|---------------|---------------|\n")
            
            for token, attentions in analysis['token_importance'].items():
                mean_att = np.mean(attentions)
                std_att = np.std(attentions)
                max_att = np.max(attentions)
                f.write(f"| {token} | {mean_att:.4f} | {std_att:.4f} | {max_att:.4f} |\n")
            
            # 步骤演化分析
            f.write("\n## Step Evolution Analysis\n\n")
            f.write("| Step | Max Attention | Mean Attention | Std Attention | Entropy |\n")
            f.write("|------|---------------|----------------|---------------|----------|\n")
            
            for i, step_data in enumerate(analysis['step_evolution']):
                entropy = analysis['attention_entropy'][i]
                f.write(f"| {step_data['step']} | {step_data['max_attention']:.4f} | "
                       f"{step_data['mean_attention']:.4f} | {step_data['attention_std']:.4f} | "
                       f"{entropy:.4f} |\n")
            
            # 最大注意力token
            f.write("\n## Max Attention Tokens by Step\n\n")
            for i, token in enumerate(analysis['max_attention_tokens']):
                f.write(f"Step {i}: {token}\n")
        
        print(f"Analysis report saved to: {save_path}")


def create_attention_visualization_dir(base_dir: str = "attention_visualizations") -> str:
    """创建注意力可视化目录"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir
