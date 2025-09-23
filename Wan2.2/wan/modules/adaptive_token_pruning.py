"""
自适应Token修剪模块 - 简化版本
只基于真实的latent变化进行token修剪，无任何模拟成分
"""

import torch
import torch.nn as nn
import math

class AdaptiveTokenPruning:
    """
    自适应Token修剪器 - 简化版本
    仅基于真实的latent变化进行修剪，确保100%真实计算节省
    """
    
    def __init__(self, 
                 baseline_steps=5,             # 前几步完全推理，用于建立基准
                 percentile_threshold=20,      # 第5步最低x%作为阈值 (可调参数)
                 start_layer=6,                # 开始修剪的层数（第6步开始）
                 end_layer=35,                 # 结束修剪的层数（高噪声专家结束前）
                 expert_name="high_noise"):    # 专家名称，只在高噪声专家使用
        self.baseline_steps = baseline_steps
        self.percentile_threshold = percentile_threshold
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.expert_name = expert_name
        
        # 动态阈值（将在第baseline_steps步确定）
        self.dynamic_threshold = None
        
        # 状态追踪
        self.frozen_tokens = set()
        
        # 真实变化分数统计（用于动态阈值计算）
        self.change_score_stats = {
            'min': float('inf'), 
            'max': 0.0, 
            'sum': 0.0, 
            'count': 0,
            'values': []  # 仅在计算阈值时临时存储，不输出到文件
        }
        
        # 第baseline_steps步的所有token真实变化评分（用于确定动态阈值）
        self.baseline_scores = []
        
        # 每步时间记录
        self.step_timings = []
        
    def update_change_score_statistics(self, change_score):
        """更新真实变化分数统计信息"""
        stats = self.change_score_stats
        stats['min'] = min(stats['min'], change_score)
        stats['max'] = max(stats['max'], change_score)
        stats['sum'] += change_score
        stats['count'] += 1
        stats['values'].append(change_score)
    
    def calculate_dynamic_threshold(self):
        """根据第baseline_steps步的真实变化分数分布计算动态阈值"""
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
        if layer_idx < self.baseline_steps:
            return False
            
        # 3. 只在指定层数范围内应用（渐进式修剪）
        if layer_idx > self.end_layer:
            return False
            
        return True

    def should_prune_token(self, change_score):
        """
        基于真实变化分数和动态阈值判断是否应该修剪某个token
        
        Args:
            change_score: 真实的token变化分数
            
        Returns:
            bool: 是否应该修剪该token
        """
        # 如果还没有确定动态阈值，不修剪
        if self.dynamic_threshold is None:
            return False
            
        # 动态阈值修剪：变化分数低于阈值就修剪
        return change_score < self.dynamic_threshold
    
    def get_pruning_summary(self):
        """获取修剪过程的详细总结"""
        # 创建不包含详细score值的统计信息
        stats_summary = {
            'min': self.change_score_stats.get('min', 0),
            'max': self.change_score_stats.get('max', 0),
            'sum': self.change_score_stats.get('sum', 0),
            'count': self.change_score_stats.get('count', 0),
            'avg': self.change_score_stats.get('sum', 0) / max(self.change_score_stats.get('count', 1), 1)
            # 不包含'values'数组，避免输出3600个score值
        }
        
        summary = {
            'total_frozen_tokens': len(self.frozen_tokens),
            'frozen_token_list': list(self.frozen_tokens),
            'dynamic_threshold': self.dynamic_threshold,
            'percentile_threshold': self.percentile_threshold,
            'baseline_steps': self.baseline_steps,
            'change_score_stats': stats_summary  # 只包含统计信息，不包含原始数据
        }
        return summary
    
    def save_pruning_log(self, output_dir, step_idx=None):
        """保存裁剪日志到输出文件夹"""
        import os
        import json
        from datetime import datetime
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if step_idx is not None:
            log_filename = f"token_pruning_step_{step_idx:02d}_{timestamp}.json"
        else:
            log_filename = f"token_pruning_summary_{timestamp}.json"
        
        log_path = os.path.join(output_dir, log_filename)
        
        # 准备日志数据
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'step_index': step_idx,
            'pruning_summary': self.get_pruning_summary(),
            'expert_name': self.expert_name,
            'configuration': {
                'baseline_steps': self.baseline_steps,
                'percentile_threshold': self.percentile_threshold,
                'start_layer': self.start_layer,
                'end_layer': self.end_layer
            }
        }
        
        # 保存到JSON文件
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        return log_path

    def generate_pruning_summary_report(self, output_dir):
        """生成完整的裁剪过程汇总报告"""
        import os
        from datetime import datetime
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成汇总报告文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"token_pruning_summary_report_{timestamp}.txt"
        report_path = os.path.join(output_dir, report_filename)
        
        # 准备汇总数据
        summary = self.get_pruning_summary()
        
        # 生成文本报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("WAN2.2 Token修剪过程详细报告 (基于真实Latent变化)\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"专家类型: {self.expert_name}\n")
            f.write("\n")
            
            # 配置信息
            f.write("📋 配置参数:\n")
            f.write("-" * 40 + "\n")
            f.write(f"基准步数: {self.baseline_steps}\n")
            f.write(f"百分位阈值: {self.percentile_threshold}%\n")
            f.write(f"动态阈值: {self.dynamic_threshold:.4f}\n" if self.dynamic_threshold else "动态阈值: 未确定\n")
            f.write(f"开始层数: {self.start_layer}\n")
            f.write(f"结束层数: {self.end_layer}\n")
            f.write("评分方式: 仅基于真实Latent变化，无任何模拟\n")
            f.write("\n")
            
            # 总体统计
            f.write("📊 总体统计:\n")
            f.write("-" * 40 + "\n")
            f.write(f"总冻结Token数: {summary['total_frozen_tokens']}\n")
            f.write(f"变化分数统计: min={summary['change_score_stats'].get('min', 0):.4f}, "
                   f"max={summary['change_score_stats'].get('max', 0):.4f}, "
                   f"avg={summary['change_score_stats'].get('sum', 0) / max(summary['change_score_stats'].get('count', 1), 1):.4f}\n")
            f.write(f"统计样本数: {summary['change_score_stats'].get('count', 0)} 个真实token变化值\n")
            f.write("\n")
            
            # 冻结Token列表
            f.write("🧊 冻结Token列表:\n")
            f.write("-" * 40 + "\n")
            frozen_tokens = sorted(summary['frozen_token_list'])
            for i, token_idx in enumerate(frozen_tokens):
                if i % 10 == 0 and i > 0:
                    f.write("\n")
                f.write(f"{token_idx:4d} ")
            f.write(f"\n\n共 {len(frozen_tokens)} 个Token被冻结\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("报告结束 - 100%基于真实数据，零模拟\n")
            f.write("=" * 80 + "\n")
        
        print(f"📄 裁剪汇总报告已保存: {report_path}")
        return report_path
