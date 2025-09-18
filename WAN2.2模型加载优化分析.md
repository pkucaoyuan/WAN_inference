# WAN2.2模型加载时间过长问题分析与优化

## 🔍 **加载时间过长的原因分析**

### **1. 多个大型模型顺序加载**

**27B MOE模型加载过程**：
```python
# 加载顺序和时间估算
self.text_encoder = T5EncoderModel(...)          # ~4GB, 耗时30-60秒
self.vae = Wan2_1_VAE(...)                      # ~1GB, 耗时10-20秒  
self.low_noise_model = WanModel.from_pretrained(...)   # ~14GB, 耗时60-120秒
self.high_noise_model = WanModel.from_pretrained(...)  # ~14GB, 耗时60-120秒

# 总加载时间: 160-320秒 (2.5-5分钟)
```

### **2. CPU↔GPU数据传输瓶颈**

**设备转移开销**：
```python
# 每个模型都需要CPU→GPU传输
model.to(self.device)  # PCIe 4.0 x16: ~32GB/s
# 14GB模型传输时间: 14GB ÷ 32GB/s ≈ 0.4秒 × 4个模型 = 1.6秒
```

### **3. 权重文件I/O瓶颈**

**磁盘读取速度**：
```python
torch.load(checkpoint_path, map_location='cpu')
# SSD读取速度: ~3-7GB/s
# 27B模型权重读取: ~111GB ÷ 5GB/s ≈ 22秒
# HDD读取速度: ~0.1-0.2GB/s  
# 27B模型权重读取: ~111GB ÷ 0.15GB/s ≈ 740秒 (12分钟)
```

### **4. 模型初始化和配置**

**额外开销**：
```python
# FSDP分片初始化
if dit_fsdp:
    model = shard_fn(model)  # 5-15秒

# 数据类型转换
if convert_model_dtype:
    model.to(self.param_dtype)  # 2-5秒

# 序列并行配置
if use_sp:
    # 注册分布式hook # 1-3秒
```

## ⚡ **优化方案**

### **方案1: 并行加载模型**
```python
import concurrent.futures
import threading

def load_models_parallel(self):
    """并行加载多个模型组件"""
    
    def load_t5():
        return T5EncoderModel(...)
    
    def load_vae():
        return Wan2_1_VAE(...)
    
    def load_low_noise():
        return WanModel.from_pretrained(..., subfolder='low_noise_model')
    
    def load_high_noise():
        return WanModel.from_pretrained(..., subfolder='high_noise_model')
    
    # 并行加载
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            't5': executor.submit(load_t5),
            'vae': executor.submit(load_vae),
            'low_noise': executor.submit(load_low_noise),
            'high_noise': executor.submit(load_high_noise)
        }
        
        # 等待所有加载完成
        results = {name: future.result() for name, future in futures.items()}
    
    # 预期加速: 2.5-4x (并行加载)
```

### **方案2: 延迟加载策略**
```python
def lazy_loading_strategy(self):
    """延迟加载：只在需要时加载模型"""
    
    # 启动时只加载必要组件
    self.text_encoder = T5EncoderModel(...)  # 必须预加载
    self.vae = Wan2_1_VAE(...)              # 必须预加载
    
    # 专家模型延迟加载
    self.low_noise_model = None
    self.high_noise_model = None
    self._low_noise_loaded = False
    self._high_noise_loaded = False
    
    def _load_expert_on_demand(self, expert_type):
        if expert_type == 'low_noise' and not self._low_noise_loaded:
            self.low_noise_model = WanModel.from_pretrained(...)
            self._low_noise_loaded = True
        elif expert_type == 'high_noise' and not self._high_noise_loaded:
            self.high_noise_model = WanModel.from_pretrained(...)
            self._high_noise_loaded = True
    
    # 预期效果: 启动时间减少50-70%
```

### **方案3: 模型缓存和预热**
```python
def model_caching_strategy(self):
    """模型缓存：避免重复加载"""
    
    import pickle
    cache_dir = Path("./model_cache")
    cache_dir.mkdir(exist_ok=True)
    
    def load_with_cache(model_path, cache_key):
        cache_file = cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            print(f"从缓存加载 {cache_key}...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"首次加载 {cache_key}...")
            model = WanModel.from_pretrained(model_path)
            # 缓存模型状态
            with open(cache_file, 'wb') as f:
                pickle.dump(model.state_dict(), f)
            return model
    
    # 预期效果: 第二次启动加速80%+
```

### **方案4: 存储优化**
```python
def storage_optimization():
    """存储和I/O优化"""
    
    # 1. 使用内存映射加载
    torch.load(checkpoint_path, map_location='cpu', mmap=True)
    
    # 2. 预分配GPU内存
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.9)
    
    # 3. 使用更快的数据类型
    torch_dtype = torch.bfloat16  # 比float32快2x
    
    # 4. 批量传输到GPU
    with torch.cuda.stream(torch.cuda.Stream()):
        model.to(device, non_blocking=True)
```

## 🚀 **推荐的快速启动配置**

### **配置1: 最小内存占用（慢启动）**
```bash
python generate.py --task t2v-A14B \
    --ckpt_dir ./model_weights \
    --offload_model True \      # 模型卸载，减少内存但增加加载时间
    --convert_model_dtype \     # 数据类型转换，额外开销
    --t5_cpu                    # T5在CPU，减少GPU内存但增加传输时间
```

### **配置2: 平衡模式（推荐）**
```bash
python generate.py --task t2v-A14B \
    --ckpt_dir ./model_weights \
    --convert_model_dtype       # 保留数据类型转换
    # 移除offload_model和t5_cpu，减少运行时传输
```

### **配置3: 最快启动（需要更多内存）**
```bash
python generate.py --task t2v-A14B \
    --ckpt_dir ./model_weights
    # 无任何优化参数，所有模型常驻GPU
```

### **配置4: 多GPU最优（推荐生产环境）**
```bash
torchrun --nproc_per_node=4 generate.py \
    --task t2v-A14B \
    --ckpt_dir ./model_weights \
    --dit_fsdp \               # 模型分片，减少单卡加载时间
    --t5_fsdp \                # T5分片
    --ulysses_size 4           # 序列并行，加速推理
```

## 📊 **性能对比**

| 配置 | 启动时间 | 内存需求 | 推理速度 | 适用场景 |
|------|----------|----------|----------|----------|
| **最小内存** | 5-8分钟 | 40GB | 慢 | 内存受限 |
| **平衡模式** | 3-5分钟 | 60GB | 中等 | 一般使用 |
| **最快启动** | 2-3分钟 | 80GB+ | 快 | 内存充足 |
| **多GPU** | 1-2分钟 | 20GB/卡 | 最快 | 生产环境 |

## 🔧 **立即可用的优化技巧**

### **1. 预热GPU**
```bash
# 在加载模型前预热GPU
python -c "import torch; torch.cuda.init(); torch.cuda.empty_cache()"
```

### **2. 设置环境变量**
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
```

### **3. 使用RAM Disk（Linux/Mac）**
```bash
# 将权重文件放在内存中
sudo mount -t tmpfs -o size=120G tmpfs /tmp/model_weights
cp -r ./model_weights/* /tmp/model_weights/
# 使用 /tmp/model_weights 作为 ckpt_dir
```

### **4. SSD优化**
- 确保模型权重存储在NVMe SSD上
- 避免使用机械硬盘存储权重文件
- 考虑使用RAID 0提高I/O性能

---

**总结**: 加载时间主要受限于磁盘I/O、CPU-GPU传输和模型初始化。通过并行加载、延迟加载和存储优化可以显著改善启动性能。
