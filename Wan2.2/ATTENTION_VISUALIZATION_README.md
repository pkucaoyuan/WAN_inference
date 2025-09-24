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
video, timing_info = model.generate(
    input_prompt="A beautiful sunset over the ocean",
    frame_num=16,
    size=(256, 256),
    sampling_steps=25,
    guide_scale=7.5,
    enable_attention_visualization=True,
    attention_output_dir="attention_outputs"
)
```

### 2. 手动控制

```python
# 启用注意力可视化
model.enable_attention_visualization("my_attention_outputs")

# 正常生成视频
video, timing_info = model.generate(
    input_prompt="A cat playing with a ball of yarn",
    frame_num=16,
    size=(256, 256),
    sampling_steps=25,
    guide_scale=7.5
)

# 禁用注意力可视化
model.disable_attention_visualization()
```

### 3. 运行示例

```bash
# 运行完整示例
python example_attention_visualization.py

# 运行参数控制示例
python example_parameter_control.py

# 使用命令行工具
python run_with_attention.py --prompt "A beautiful sunset" --enable-attention
python run_with_attention.py --prompt "A cat playing" --frames 8 --steps 10

# 运行测试脚本
python test_attention_visualization.py --test_mode

# 单次生成
python test_attention_visualization.py --prompt "A person walking through a forest"
```

## 输出文件说明

### 可视化图像

- **average_cross_attention_map.png**: 平均Cross Attention Map
  - X轴: 文本tokens (Context)
  - Y轴: 图像tokens (Query)
  - 颜色: 注意力权重强度 (越白表示权重越大)
  - 内容: 所有去噪步骤、批次和注意力头的平均注意力权重

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

### generate参数

- `input_prompt` (str): 输入文本提示
- `frame_num` (int): 生成视频帧数，默认81
- `size` (tuple): 视频尺寸，默认(1280, 720)
- `sampling_steps` (int): 去噪步数，默认50
- `guide_scale` (float): CFG引导尺度，默认5.0
- `enable_attention_visualization` (bool): 是否启用注意力可视化，默认False
- `attention_output_dir` (str): 注意力可视化输出目录，默认"attention_outputs"

### 参数控制示例

#### 禁用注意力可视化（默认）
```python
video, timing_info = model.generate(
    input_prompt="A beautiful sunset",
    frame_num=16,
    size=(256, 256),
    sampling_steps=25,
    guide_scale=7.5
    # enable_attention_visualization=False  # 默认值
)
```

#### 启用注意力可视化
```python
video, timing_info = model.generate(
    input_prompt="A beautiful sunset",
    frame_num=16,
    size=(256, 256),
    sampling_steps=25,
    guide_scale=7.5,
    enable_attention_visualization=True,  # 启用
    attention_output_dir="my_attention_outputs"  # 自定义输出目录
)
```

#### 命令行控制
```bash
# 禁用注意力可视化
python run_with_attention.py --prompt "A beautiful sunset"

# 启用注意力可视化
python run_with_attention.py --prompt "A beautiful sunset" --enable-attention

# 自定义参数
python run_with_attention.py --prompt "A cat playing" --frames 8 --steps 10 --enable-attention --attention-dir "my_outputs"
```

## 注意事项

1. **内存使用**: 注意力可视化会增加内存使用，建议减少帧数或步数进行测试
2. **计算开销**: 捕获attention权重会增加计算时间
3. **存储空间**: 每步都会生成图像文件，确保有足够的存储空间
4. **真实权重**: 当前实现使用基于特征相似度的真实attention权重计算

## 故障排除

### 常见问题

1. **ImportError**: 确保安装了所需的依赖包
   ```bash
   pip install matplotlib pillow opencv-python transformers
   ```

2. **内存不足**: 减少帧数或推理步数
   ```python
   video, timing_info = model.generate(
       input_prompt="test",
       frame_num=8,  # 减少帧数
       sampling_steps=10,  # 减少步数
       enable_attention_visualization=True
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
