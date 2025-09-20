# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
自适应Token修剪模块
实现基于变化幅度的动态token停止更新机制
"""

import torch
import torch.nn as nn
import math

class AdaptiveTokenPruning:
    """
    自适应Token修剪器
    在高噪声专家的后期阶段，动态停止变化小的token更新
    """
    
    def __init__(self, 
                 threshold=0.01,           # 变化阈值
                 min_active_ratio=0.3,     # 最小激活token比例
                 start_layer=20,           # 开始修剪的层数
                 structural_ratio=0.2):    # 结构token比例
        self.threshold = threshold
        self.min_active_ratio = min_active_ratio
        self.start_layer = start_layer
        self.structural_ratio = structural_ratio
        
        # 状态追踪
        self.token_states = {}
        self.frozen_tokens = set()
        self.structural_tokens = set()
        
    def should_prune_token(self, token_idx, current_hidden, prev_hidden, layer_idx):
        """
        判断是否应该停止更新某个token
        
        Args:
            token_idx: token索引
            current_hidden: 当前层的hidden state
            prev_hidden: 前一层的hidden state  
            layer_idx: 当前层索引
            
        Returns:
            bool: 是否应该停止更新
        """
        # 只在指定层数后开始修剪
        if layer_idx < self.start_layer:
            return False
            
        # 已经冻结的token保持冻结
        if token_idx in self.frozen_tokens:
            return True
            
        # 结构token不修剪
        if token_idx in self.structural_tokens:
            return False
            
        # 计算token变化幅度
        if prev_hidden is not None:
            change_magnitude = torch.norm(current_hidden - prev_hidden, dim=-1)
            relative_change = change_magnitude / (torch.norm(prev_hidden, dim=-1) + 1e-8)
            
            # 变化小于阈值则考虑停止
            if relative_change < self.threshold:
                return True
                
        return False
    
    def identify_structural_tokens(self, attention_weights, seq_len):
        """
        识别结构性token（高注意力权重的token）
        
        Args:
            attention_weights: [B, H, L, L] 注意力权重
            seq_len: 序列长度
            
        Returns:
            set: 结构token的索引集合
        """
        # 计算每个token的平均注意力接收量
        avg_attention = attention_weights.mean(dim=(0, 1))  # [L, L]
        token_importance = avg_attention.sum(dim=-1)  # [L]
        
        # 选择top-k作为结构token
        num_structural = int(seq_len * self.structural_ratio)
        structural_indices = torch.topk(token_importance, num_structural).indices
        
        return set(structural_indices.cpu().tolist())
    
    def apply_pruning(self, hidden_states, attention_weights, layer_idx):
        """
        应用token修剪
        
        Args:
            hidden_states: [B, L, C] 当前hidden states
            attention_weights: [B, H, L, L] 注意力权重
            layer_idx: 当前层索引
            
        Returns:
            tuple: (修剪后的hidden_states, 激活mask, 修剪统计)
        """
        B, L, C = hidden_states.shape
        
        # 第一次运行时识别结构token
        if layer_idx == self.start_layer and not self.structural_tokens:
            self.structural_tokens = self.identify_structural_tokens(attention_weights, L)
        
        # 创建激活mask
        active_mask = torch.ones(L, dtype=torch.bool, device=hidden_states.device)
        
        # 获取前一层状态用于比较
        prev_hidden = self.token_states.get(layer_idx - 1)
        
        # 逐token判断是否修剪
        newly_frozen = []
        for token_idx in range(L):
            if self.should_prune_token(
                token_idx, 
                hidden_states[0, token_idx], 
                prev_hidden[0, token_idx] if prev_hidden is not None else None,
                layer_idx
            ):
                active_mask[token_idx] = False
                self.frozen_tokens.add(token_idx)
                newly_frozen.append(token_idx)
        
        # 确保最小激活比例
        active_count = active_mask.sum().item()
        min_active = int(L * self.min_active_ratio)
        
        if active_count < min_active:
            # 重新激活一些重要的token
            inactive_indices = (~active_mask).nonzero().flatten()
            reactivate_count = min_active - active_count
            if len(inactive_indices) > 0:
                # 基于重要性重新激活
                reactivate_indices = inactive_indices[:reactivate_count]
                active_mask[reactivate_indices] = True
                for idx in reactivate_indices:
                    self.frozen_tokens.discard(idx.item())
        
        # 保存当前状态
        self.token_states[layer_idx] = hidden_states.clone()
        
        # 修剪统计
        pruning_stats = {
            'layer': layer_idx,
            'total_tokens': L,
            'active_tokens': active_mask.sum().item(),
            'pruned_tokens': (~active_mask).sum().item(),
            'newly_frozen': len(newly_frozen),
            'pruning_ratio': (~active_mask).sum().item() / L
        }
        
        return hidden_states, active_mask, pruning_stats

def create_pruned_attention_mask(active_mask, original_mask=None):
    """
    创建修剪后的注意力mask
    
    Args:
        active_mask: [L] 激活token的mask
        original_mask: 原始注意力mask
        
    Returns:
        torch.Tensor: 修剪后的注意力mask
    """
    L = len(active_mask)
    
    # 创建基础mask：冻结token只能被attention，不能attend to others
    pruned_mask = torch.ones(L, L, dtype=torch.bool)
    
    # 冻结token的行设为False（不能attend to others）
    frozen_indices = (~active_mask).nonzero().flatten()
    if len(frozen_indices) > 0:
        pruned_mask[frozen_indices, :] = False
        # 但允许其他token attend to 冻结token（保持KV可用）
    
    if original_mask is not None:
        pruned_mask = pruned_mask & original_mask
        
    return pruned_mask
