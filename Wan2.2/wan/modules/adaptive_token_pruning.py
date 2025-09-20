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
    基于token变化、自注意力权重、跨模态注意力权重的综合评分进行修剪
    """
    
    def __init__(self, 
                 change_threshold=0.01,        # 变化阈值
                 min_active_ratio=0.3,         # 最小激活token比例
                 start_layer=20,               # 开始修剪的层数
                 change_weight=0.4,            # token变化权重
                 self_attn_weight=0.3,         # 自注意力权重
                 cross_attn_weight=0.3,        # 跨模态注意力权重
                 image_token_start=0,          # 图像token起始索引
                 text_token_start=None):       # 文本token起始索引
        self.change_threshold = change_threshold
        self.min_active_ratio = min_active_ratio
        self.start_layer = start_layer
        
        # 加权系数
        self.change_weight = change_weight
        self.self_attn_weight = self_attn_weight  
        self.cross_attn_weight = cross_attn_weight
        
        # Token分区
        self.image_token_start = image_token_start
        self.text_token_start = text_token_start
        
        # 状态追踪
        self.token_states = {}
        self.frozen_tokens = set()
        self.token_scores_history = {}
        
    def calculate_image_self_attention_score(self, self_attn_weights, token_idx, seq_len):
        """
        计算单个图像token在自注意力中的权重总和
        
        Args:
            self_attn_weights: [B, H, L, L] 自注意力权重矩阵
            token_idx: 目标token索引
            seq_len: 图像token序列长度
            
        Returns:
            float: 该token的自注意力权重总和
        """
        if self.text_token_start is None:
            # 假设所有token都是图像token
            image_end = seq_len
        else:
            image_end = self.text_token_start
            
        # 计算该token与所有图像token的注意力权重
        # token_idx作为query，与所有图像token作为key的注意力权重
        image_self_attn = self_attn_weights[0, :, token_idx, self.image_token_start:image_end]
        
        # 所有头的平均，然后求和
        score = image_self_attn.mean(dim=0).sum().item()
        return score
    
    def calculate_image_cross_attention_score(self, cross_attn_weights, token_idx):
        """
        计算单个图像token与文本token的跨模态注意力权重总和
        
        Args:
            cross_attn_weights: [B, H, L, text_len] 跨模态注意力权重矩阵
            token_idx: 目标图像token索引
            
        Returns:
            float: 该token与文本token的注意力权重总和
        """
        if cross_attn_weights is None:
            return 0.0
            
        # 计算该图像token对所有文本token的注意力权重
        cross_attn = cross_attn_weights[0, :, token_idx, :]  # [H, text_len]
        
        # 所有头的平均，然后求和
        score = cross_attn.mean(dim=0).sum().item()
        return score
    
    def calculate_composite_score(self, token_idx, current_hidden, prev_hidden, 
                                self_attn_weights, cross_attn_weights, seq_len):
        """
        计算综合评分：变化值 + 自注意力权重 + 跨模态注意力权重
        
        Args:
            token_idx: token索引
            current_hidden: 当前隐藏状态
            prev_hidden: 前一层隐藏状态
            self_attn_weights: 自注意力权重
            cross_attn_weights: 跨模态注意力权重
            seq_len: 序列长度
            
        Returns:
            float: 综合重要性评分（越高越重要）
        """
        scores = {}
        
        # 1. Token变化分数（变化越大越重要）
        if prev_hidden is not None:
            change_magnitude = torch.norm(current_hidden - prev_hidden, dim=-1).item()
            relative_change = change_magnitude / (torch.norm(prev_hidden, dim=-1).item() + 1e-8)
            scores['change'] = relative_change
        else:
            scores['change'] = 1.0  # 第一层默认高重要性
        
        # 2. 自注意力分数（图像token间的重要性）
        scores['self_attn'] = self.calculate_image_self_attention_score(
            self_attn_weights, token_idx, seq_len)
        
        # 3. 跨模态注意力分数（与文本的关联性）
        scores['cross_attn'] = self.calculate_image_cross_attention_score(
            cross_attn_weights, token_idx)
        
        # 4. 加权综合评分
        composite_score = (
            self.change_weight * scores['change'] +
            self.self_attn_weight * scores['self_attn'] + 
            self.cross_attn_weight * scores['cross_attn']
        )
        
        return composite_score, scores

    def should_prune_token(self, token_idx, current_hidden, prev_hidden, layer_idx,
                          self_attn_weights=None, cross_attn_weights=None, seq_len=None):
        """
        基于综合评分判断是否应该停止更新某个token
        
        Args:
            token_idx: token索引
            current_hidden: 当前层的hidden state
            prev_hidden: 前一层的hidden state  
            layer_idx: 当前层索引
            self_attn_weights: 自注意力权重矩阵
            cross_attn_weights: 跨模态注意力权重矩阵
            seq_len: 序列长度
            
        Returns:
            bool: 是否应该停止更新
        """
        # 只在指定层数后开始修剪
        if layer_idx < self.start_layer:
            return False
            
        # 已经冻结的token保持冻结
        if token_idx in self.frozen_tokens:
            return True
        
        # 只对图像token进行修剪判断
        if self.text_token_start is not None and token_idx >= self.text_token_start:
            return False  # 文本token不修剪
            
        # 计算综合重要性评分
        if self_attn_weights is not None and seq_len is not None:
            composite_score, detailed_scores = self.calculate_composite_score(
                token_idx, current_hidden, prev_hidden,
                self_attn_weights, cross_attn_weights, seq_len
            )
            
            # 记录评分历史
            if token_idx not in self.token_scores_history:
                self.token_scores_history[token_idx] = []
            self.token_scores_history[token_idx].append({
                'layer': layer_idx,
                'composite_score': composite_score,
                'change_score': detailed_scores['change'],
                'self_attn_score': detailed_scores['self_attn'],
                'cross_attn_score': detailed_scores['cross_attn']
            })
            
            # 综合评分低于阈值则修剪
            return composite_score < self.change_threshold
        else:
            # 回退到简单的变化检测
            if prev_hidden is not None:
                change_magnitude = torch.norm(current_hidden - prev_hidden, dim=-1)
                relative_change = change_magnitude / (torch.norm(prev_hidden, dim=-1) + 1e-8)
                return relative_change < self.change_threshold
                
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
    
    def apply_pruning(self, hidden_states, self_attn_weights, cross_attn_weights, layer_idx):
        """
        应用基于综合评分的token修剪
        
        Args:
            hidden_states: [B, L, C] 当前hidden states
            self_attn_weights: [B, H, L, L] 自注意力权重
            cross_attn_weights: [B, H, L, text_len] 跨模态注意力权重
            layer_idx: 当前层索引
            
        Returns:
            tuple: (修剪后的hidden_states, 激活mask, 修剪统计)
        """
        B, L, C = hidden_states.shape
        
        # 创建激活mask
        active_mask = torch.ones(L, dtype=torch.bool, device=hidden_states.device)
        
        # 获取前一层状态用于比较
        prev_hidden = self.token_states.get(layer_idx - 1)
        
        # 计算所有图像token的综合评分
        token_scores = []
        detailed_scores_list = []
        
        image_token_end = self.text_token_start if self.text_token_start is not None else L
        
        for token_idx in range(self.image_token_start, image_token_end):
            composite_score, detailed_scores = self.calculate_composite_score(
                token_idx, 
                hidden_states[0, token_idx],
                prev_hidden[0, token_idx] if prev_hidden is not None else None,
                self_attn_weights, cross_attn_weights, L
            )
            token_scores.append((token_idx, composite_score))
            detailed_scores_list.append(detailed_scores)
        
        # 根据综合评分排序，保留重要的token
        token_scores.sort(key=lambda x: x[1], reverse=True)  # 按评分降序
        
        # 计算需要保持激活的token数量
        total_image_tokens = image_token_end - self.image_token_start
        min_active_image_tokens = max(
            int(total_image_tokens * self.min_active_ratio),
            1  # 至少保持1个图像token激活
        )
        
        # 修剪评分低的token
        newly_frozen = []
        for i, (token_idx, score) in enumerate(token_scores):
            # 已经冻结的保持冻结
            if token_idx in self.frozen_tokens:
                active_mask[token_idx] = False
                continue
                
            # 保留top-k高评分token，修剪其余
            if i >= min_active_image_tokens and score < self.change_threshold:
                active_mask[token_idx] = False
                self.frozen_tokens.add(token_idx)
                newly_frozen.append(token_idx)
        
        # 记录评分历史
        for i, (token_idx, score) in enumerate(token_scores):
            if token_idx not in self.token_scores_history:
                self.token_scores_history[token_idx] = []
            self.token_scores_history[token_idx].append({
                'layer': layer_idx,
                'composite_score': score,
                'rank': i + 1,
                'is_active': active_mask[token_idx].item(),
                **detailed_scores_list[i]
            })
        
        # 保存当前状态
        self.token_states[layer_idx] = hidden_states.clone()
        
        # 修剪统计
        pruning_stats = {
            'layer': layer_idx,
            'total_tokens': L,
            'image_tokens': total_image_tokens,
            'active_image_tokens': active_mask[self.image_token_start:image_token_end].sum().item(),
            'pruned_image_tokens': total_image_tokens - active_mask[self.image_token_start:image_token_end].sum().item(),
            'newly_frozen': len(newly_frozen),
            'image_pruning_ratio': (total_image_tokens - active_mask[self.image_token_start:image_token_end].sum().item()) / total_image_tokens,
            'avg_change_score': sum(scores['change'] for scores in detailed_scores_list) / len(detailed_scores_list),
            'avg_self_attn_score': sum(scores['self_attn'] for scores in detailed_scores_list) / len(detailed_scores_list),
            'avg_cross_attn_score': sum(scores['cross_attn'] for scores in detailed_scores_list) / len(detailed_scores_list)
        }
        
        return hidden_states, active_mask, pruning_stats
    
    def get_pruning_summary(self):
        """获取修剪过程的详细总结"""
        summary = {
            'total_frozen_tokens': len(self.frozen_tokens),
            'frozen_token_list': list(self.frozen_tokens),
            'scores_history': self.token_scores_history
        }
        return summary

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
