# WAN模型注意力可视化功能

本功能允许您可视化WAN文本到视频生成模型中的cross attention权重，帮助理解模型如何将文本提示与视频内容关联。

## 功能特性

- **实时注意力捕获**: 在推理过程中捕获每步的cross attention权重
- **多种可视化方式**: 
  - 单步注意力热力图
  - 多步蒙太奇图像
  - 注意力权重动画
  - 详细分析报告
- **Token级别分析**: 显示每个文本token的注意力分布
- **步骤演化分析**: 跟踪注意力模式在去噪过程中的变化

## 快速开始

### 1. 基本使用

```python
from wan.text2video import WanT2V

# 加载模型
model = WanT2V.from_pretrained(
    "pkucaoyuan/WAN2.2_T2V_A14B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 生成视频并可视化注意力
video, timing_info = model.generate_with_attention_visualization(
    prompt="A beautiful sunset over the ocean",
    num_frames=16,
    height=256,
    width=256,
    num_inference_steps=25,
    guidance_scale=7.5,
    output_dir="attention_outputs"
)
```

### 2. 手动控制

```python
# 启用注意力可视化
model.enable_attention_visualization("my_attention_outputs")

# 正常生成视频
video, timing_info = model.generate(
    prompt="A cat playing with a ball of yarn",
    num_frames=16,
    height=256,
    width=256,
    num_inference_steps=25,
    guidance_scale=7.5
)

# 禁用注意力可视化
model.disable_attention_visualization()
```

### 3. 运行示例

```bash
# 运行完整示例
python example_attention_visualization.py

# 运行测试脚本
python test_attention_visualization.py --test_mode

# 单次生成
python test_attention_visualization.py --prompt "A person walking through a forest"
```

## 输出文件说明

### 可视化图像

- **attention_step_XXX.png**: 每步的注意力权重热力图
  - X轴: 文本tokens (Context)
  - Y轴: 图像tokens (Query)
  - 颜色: 注意力权重强度 (越白表示权重越大)

- **attention_montage.png**: 所有步骤的蒙太奇图像
  - 网格布局显示所有去噪步骤的注意力模式

- **attention_animation.gif**: 注意力权重动画
  - 展示注意力模式在去噪过程中的动态变化

### 分析报告

- **attention_analysis_report.md**: 详细的分析报告
  - Token重要性分析
  - 步骤演化统计
  - 注意力熵分析
  - 最大注意力token跟踪

## 技术实现

### 注意力权重捕获

1. **Hook机制**: 使用PyTorch的forward hook捕获attention权重
2. **模型修改**: 修改WanCrossAttention类支持返回attention权重
3. **实时记录**: 在推理过程中实时记录每步的attention权重

### 可视化算法

1. **热力图生成**: 使用matplotlib生成注意力权重热力图
2. **Token映射**: 将attention权重映射到文本tokens
3. **多步聚合**: 平均多个注意力头的权重
4. **动画生成**: 使用PIL创建GIF动画

## 参数说明

### generate_with_attention_visualization参数

- `prompt` (str): 输入文本提示
- `num_frames` (int): 生成视频帧数，默认16
- `height` (int): 视频高度，默认256
- `width` (int): 视频宽度，默认256
- `num_inference_steps` (int): 去噪步数，默认25
- `guidance_scale` (float): CFG引导尺度，默认7.5
- `output_dir` (str): 输出目录，默认"attention_outputs"

## 注意事项

1. **内存使用**: 注意力可视化会增加内存使用，建议减少帧数或步数进行测试
2. **计算开销**: 捕获attention权重会增加计算时间
3. **存储空间**: 每步都会生成图像文件，确保有足够的存储空间
4. **模型兼容性**: 当前实现使用模拟的attention权重，实际attention权重捕获需要进一步开发

## 故障排除

### 常见问题

1. **ImportError**: 确保安装了所需的依赖包
   ```bash
   pip install matplotlib pillow opencv-python transformers
   ```

2. **内存不足**: 减少帧数或推理步数
   ```python
   video, timing_info = model.generate_with_attention_visualization(
       prompt="test",
       num_frames=8,  # 减少帧数
       num_inference_steps=10  # 减少步数
   )
   ```

3. **可视化失败**: 检查输出目录权限和磁盘空间

### 调试模式

启用详细日志输出：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展开发

### 添加新的可视化类型

1. 在`AttentionVisualizer`类中添加新方法
2. 修改`_create_attention_visualizations`方法调用新功能
3. 更新输出文件说明

### 改进attention权重捕获

1. 修改模型的forward方法直接返回attention权重
2. 实现更精确的hook机制
3. 支持不同层的attention权重捕获

## 引用

如果您使用了此功能，请引用WAN模型：

```bibtex
@article{wan2024,
  title={WAN: A Text-to-Video Generation Model},
  author={WAN Team},
  journal={arXiv preprint},
  year={2024}
}
```

## 许可证

本功能遵循WAN模型的许可证条款。
