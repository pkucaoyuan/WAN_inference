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
                 baseline_steps=5,             # 前几步完全推理，用于建立基准
                 percentile_threshold=20,      # 第5步最低x%作为阈值 (可调参数)
                 start_layer=6,                # 开始修剪的层数（第6步开始）
                 end_layer=35,                 # 结束修剪的层数（高噪声专家结束前）
                 change_weight=0.4,            # token变化权重
                 self_attn_weight=0.3,         # 自注意力权重
                 cross_attn_weight=0.3,        # 跨模态注意力权重
                 image_token_start=0,          # 图像token起始索引
                 text_token_start=None,        # 文本token起始索引
                 expert_name="high_noise"):    # 专家名称，只在高噪声专家使用
        self.baseline_steps = baseline_steps
        self.percentile_threshold = percentile_threshold
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.expert_name = expert_name
        
        # 动态阈值（将在第baseline_steps步确定）
        self.dynamic_threshold = None
        
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
        
        # 变化分数统计（用于attention分数量级对齐）
        self.change_score_stats = {
            'min': float('inf'), 
            'max': 0.0, 
            'sum': 0.0, 
            'count': 0,
            'values': []  # 存储所有变化分数用于计算百分位数
        }
        
        # 第baseline_steps步的所有token综合评分（用于确定动态阈值）
        self.baseline_scores = []
        
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
    
    def update_change_score_statistics(self, change_score):
        """更新变化分数统计信息"""
        stats = self.change_score_stats
        stats['min'] = min(stats['min'], change_score)
        stats['max'] = max(stats['max'], change_score)
        stats['sum'] += change_score
        stats['count'] += 1
        stats['values'].append(change_score)
    
    def scale_attention_scores_to_change_magnitude(self, self_attn_score, cross_attn_score):
        """将attention分数缩放到变化分数的量级"""
        if self.change_score_stats['count'] == 0:
            return self_attn_score, cross_attn_score
            
        # 使用变化分数的平均值作为目标量级
        change_avg = self.change_score_stats['sum'] / self.change_score_stats['count']
        change_range = self.change_score_stats['max'] - self.change_score_stats['min']
        
        if change_range == 0:
            return self_attn_score, cross_attn_score
        
        # 将attention分数缩放到变化分数的量级
        # 假设attention分数的合理范围，然后线性缩放到change分数范围
        scaled_self_attn = self_attn_score * (change_range / max(self_attn_score, 1e-8))
        scaled_cross_attn = cross_attn_score * (change_range / max(cross_attn_score, 1e-8))
        
        return scaled_self_attn, scaled_cross_attn

    def calculate_composite_score(self, token_idx, current_hidden, prev_hidden, 
                                self_attn_weights, cross_attn_weights, seq_len, layer_idx):
        """
        计算基于量级对齐的综合评分：变化值 + 缩放后的attention权重
        
        Args:
            token_idx: token索引
            current_hidden: 当前隐藏状态
            prev_hidden: 前一层隐藏状态
            self_attn_weights: 自注意力权重
            cross_attn_weights: 跨模态注意力权重
            seq_len: 序列长度
            layer_idx: 当前层索引
            
        Returns:
            tuple: (综合评分, 原始分数字典, 缩放后分数字典)
        """
        raw_scores = {}
        
        # 1. Token变化分数（基准量级）
        if prev_hidden is not None:
            change_magnitude = torch.norm(current_hidden - prev_hidden, dim=-1).item()
            relative_change = change_magnitude / (torch.norm(prev_hidden, dim=-1).item() + 1e-8)
            raw_scores['change'] = relative_change
        else:
            raw_scores['change'] = 1.0  # 第一层默认高重要性
        
        # 2. 自注意力分数（图像token间的重要性）
        raw_scores['self_attn'] = self.calculate_image_self_attention_score(
            self_attn_weights, token_idx, seq_len)
        
        # 3. 跨模态注意力分数（与文本的关联性）
        raw_scores['cross_attn'] = self.calculate_image_cross_attention_score(
            cross_attn_weights, token_idx)
        
        # 4. 更新变化分数统计信息（前baseline_steps步收集统计）
        if layer_idx <= self.baseline_steps:
            self.update_change_score_statistics(raw_scores['change'])
        
        # 5. 将attention分数缩放到变化分数的量级
        scaled_self_attn, scaled_cross_attn = self.scale_attention_scores_to_change_magnitude(
            raw_scores['self_attn'], raw_scores['cross_attn'])
        
        scaled_scores = {
            'change': raw_scores['change'],
            'self_attn': scaled_self_attn,
            'cross_attn': scaled_cross_attn
        }
        
        # 6. 加权综合评分（量级对齐后的分数）
        composite_score = (
            self.change_weight * scaled_scores['change'] +
            self.self_attn_weight * scaled_scores['self_attn'] + 
            self.cross_attn_weight * scaled_scores['cross_attn']
        )
        
        return composite_score, raw_scores, scaled_scores

    def calculate_dynamic_threshold(self):
        """根据第baseline_steps步的分数分布计算动态阈值"""
        if len(self.baseline_scores) == 0:
            return None
            
        # 计算第percentile_threshold百分位数作为阈值
        import numpy as np
        threshold = np.percentile(self.baseline_scores, self.percentile_threshold)
        return threshold

    def should_apply_pruning(self, layer_idx, expert_name):
        """
        判断当前层是否应该应用token修剪
        
        Args:
            layer_idx: 当前层索引
            expert_name: 专家名称 ("high_noise" 或 "low_noise")
            
        Returns:
            bool: 是否应该应用修剪
        """
        # 1. 只在高噪声专家中应用
        if expert_name != "high_noise":
            return False
            
        # 2. 前baseline_steps步完全推理，不修剪
        if layer_idx <= self.baseline_steps:
            return False
            
        # 3. 只在指定层数范围内应用（渐进式修剪）
        if layer_idx > self.end_layer:
            return False
            
        return True

    def should_prune_token(self, token_idx, composite_score, layer_idx):
        """
        基于动态阈值判断是否应该修剪某个token
        
        Args:
            token_idx: token索引
            composite_score: 综合评分
            layer_idx: 当前层索引
            
        Returns:
            bool: 是否应该修剪该token
        """
        # 已经冻结的token保持冻结
        if token_idx in self.frozen_tokens:
            return True
        
        # 只对图像token进行修剪判断
        if self.text_token_start is not None and token_idx >= self.text_token_start:
            return False  # 文本token不修剪
            
        # 如果还没有确定动态阈值，不修剪
        if self.dynamic_threshold is None:
            return False
            
        # 动态阈值修剪：评分低于第baseline_steps步最低x%的阈值就修剪
        return composite_score < self.dynamic_threshold
    
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
    
    def apply_progressive_pruning(self, hidden_states, self_attn_weights, cross_attn_weights, 
                                layer_idx, expert_name):
        """
        应用渐进式token修剪（只在高噪声专家中使用）
        
        Args:
            hidden_states: [B, L, C] 当前hidden states
            self_attn_weights: [B, H, L, L] 自注意力权重
            cross_attn_weights: [B, H, L, text_len] 跨模态注意力权重
            layer_idx: 当前层索引
            expert_name: 专家名称
            
        Returns:
            tuple: (修剪后的hidden_states, 激活mask, 修剪统计)
        """
        B, L, C = hidden_states.shape
        
        # 检查是否应该应用修剪
        if not self.should_apply_pruning(layer_idx, expert_name):
            # 不修剪，返回全激活mask
            active_mask = torch.ones(L, dtype=torch.bool, device=hidden_states.device)
            pruning_stats = {
                'layer': layer_idx,
                'expert': expert_name,
                'pruning_applied': False,
                'reason': 'outside_pruning_range' if expert_name == "high_noise" else 'low_noise_expert'
            }
            return hidden_states, active_mask, pruning_stats
        
        # 创建激活mask
        active_mask = torch.ones(L, dtype=torch.bool, device=hidden_states.device)
        
        # 获取前一层状态用于比较
        prev_hidden = self.token_states.get(layer_idx - 1)
        
        # 计算所有图像token的综合评分
        token_scores = []
        raw_scores_list = []
        normalized_scores_list = []
        
        image_token_end = self.text_token_start if self.text_token_start is not None else L
        
        for token_idx in range(self.image_token_start, image_token_end):
            composite_score, raw_scores, normalized_scores = self.calculate_composite_score(
                token_idx, 
                hidden_states[0, token_idx],
                prev_hidden[0, token_idx] if prev_hidden is not None else None,
                self_attn_weights, cross_attn_weights, L, layer_idx
            )
            token_scores.append((token_idx, composite_score))
            raw_scores_list.append(raw_scores)
            normalized_scores_list.append(normalized_scores)
        
        # 在第baseline_steps步收集所有分数用于确定动态阈值
        if layer_idx == self.baseline_steps:
            self.baseline_scores = [score for _, score in token_scores]
            self.dynamic_threshold = self.calculate_dynamic_threshold()
            print(f"🎯 动态阈值已确定: {self.dynamic_threshold:.4f} (第{self.percentile_threshold}百分位数)")
        
        # 渐进式修剪：基于动态阈值，逐步增加修剪的token
        newly_frozen = []
        for i, (token_idx, score) in enumerate(token_scores):
            if self.should_prune_token(token_idx, score, layer_idx):
                active_mask[token_idx] = False
                if token_idx not in self.frozen_tokens:
                    self.frozen_tokens.add(token_idx)
                    newly_frozen.append(token_idx)
        
        # 记录详细的评分历史
        for i, (token_idx, score) in enumerate(token_scores):
            if token_idx not in self.token_scores_history:
                self.token_scores_history[token_idx] = []
            self.token_scores_history[token_idx].append({
                'layer': layer_idx,
                'expert': expert_name,
                'composite_score': score,
                'is_active': active_mask[token_idx].item(),
                'raw_scores': raw_scores_list[i],
                'normalized_scores': normalized_scores_list[i]
            })
        
        # 保存当前状态
        self.token_states[layer_idx] = hidden_states.clone()
        
        # 修剪统计
        total_image_tokens = image_token_end - self.image_token_start
        active_image_tokens = active_mask[self.image_token_start:image_token_end].sum().item()
        
        pruning_stats = {
            'layer': layer_idx,
            'expert': expert_name,
            'pruning_applied': True,
            'total_tokens': L,
            'image_tokens': total_image_tokens,
            'active_image_tokens': active_image_tokens,
            'pruned_image_tokens': total_image_tokens - active_image_tokens,
            'newly_frozen': len(newly_frozen),
            'cumulative_frozen': len(self.frozen_tokens),
            'image_pruning_ratio': (total_image_tokens - active_image_tokens) / total_image_tokens,
            'dynamic_threshold': self.dynamic_threshold,
            'percentile_threshold': self.percentile_threshold,
            'avg_composite_score': sum(score for _, score in token_scores) / len(token_scores),
            'change_score_stats': dict(self.change_score_stats)  # 当前变化分数统计
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
