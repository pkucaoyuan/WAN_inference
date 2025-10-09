"""
T2I评估脚本
计算FID、IS、CLIP Score等统计量
"""
import os
import argparse
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import pandas as pd


def calculate_fid(generated_dir: str, real_dir: str, batch_size: int = 50, device: str = "cuda"):
    """
    计算FID (Fréchet Inception Distance)
    
    Args:
        generated_dir: 生成图片目录
        real_dir: 真实图片目录
        batch_size: 批次大小
        device: 设备
    
    Returns:
        fid_score: FID分数
    """
    try:
        from pytorch_fid import fid_score
        
        print(f"📊 计算FID...")
        print(f"   生成图片: {generated_dir}")
        print(f"   真实图片: {real_dir}")
        
        fid_value = fid_score.calculate_fid_given_paths(
            [generated_dir, real_dir],
            batch_size=batch_size,
            device=device,
            dims=2048
        )
        
        print(f"✅ FID: {fid_value:.4f}")
        return fid_value
    
    except ImportError:
        print(f"❌ 未安装pytorch-fid，跳过FID计算")
        print(f"   安装命令: pip install pytorch-fid")
        return None
    except Exception as e:
        print(f"❌ FID计算失败: {e}")
        return None


def calculate_inception_score(generated_dir: str, batch_size: int = 32, splits: int = 10, device: str = "cuda"):
    """
    计算Inception Score
    
    Args:
        generated_dir: 生成图片目录
        batch_size: 批次大小
        splits: 分割数
        device: 设备
    
    Returns:
        is_mean: IS均值
        is_std: IS标准差
    """
    try:
        from torchmetrics.image.inception import InceptionScore
        from torchvision import transforms
        
        print(f"📊 计算Inception Score...")
        
        # 初始化IS计算器
        inception = InceptionScore(normalize=True).to(device)
        
        # 加载图片
        image_files = sorted([f for f in os.listdir(generated_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])
        
        all_images = []
        for img_file in tqdm(image_files, desc="加载图片"):
            img_path = os.path.join(generated_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            all_images.append(img_tensor)
        
        # 批处理计算
        for i in tqdm(range(0, len(all_images), batch_size), desc="计算IS"):
            batch = torch.stack(all_images[i:i+batch_size]).to(device)
            inception.update(batch)
        
        is_mean, is_std = inception.compute()
        
        print(f"✅ Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        return is_mean.item(), is_std.item()
    
    except ImportError:
        print(f"❌ 未安装torchmetrics，跳过IS计算")
        print(f"   安装命令: pip install torchmetrics")
        return None, None
    except Exception as e:
        print(f"❌ IS计算失败: {e}")
        return None, None


def calculate_clip_score(generated_dir: str, prompts_csv: str, batch_size: int = 32, device: str = "cuda"):
    """
    计算CLIP Score
    
    Args:
        generated_dir: 生成图片目录
        prompts_csv: prompts CSV文件
        batch_size: 批次大小
        device: 设备
    
    Returns:
        clip_score_mean: CLIP Score均值
        clip_score_std: CLIP Score标准差
    """
    try:
        import clip
        from PIL import Image
        
        print(f"📊 计算CLIP Score...")
        
        # 加载CLIP模型
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        # 读取prompts
        df = pd.read_csv(prompts_csv)
        prompt_column = 'prompt' if 'prompt' in df.columns else 'caption'
        
        # 获取图片文件
        image_files = sorted([f for f in os.listdir(generated_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # 匹配图片和prompt
        scores = []
        
        for img_file in tqdm(image_files, desc="计算CLIP Score"):
            # 从文件名提取image_id
            # 假设文件名格式: {image_id}_seed{seed}.png
            image_id = img_file.split('_')[0]
            
            # 查找对应的prompt
            try:
                image_id_int = int(image_id)
                prompt_row = df[df['image_id'] == image_id_int]
                if len(prompt_row) == 0:
                    continue
                prompt = prompt_row.iloc[0][prompt_column]
            except:
                continue
            
            # 加载图片
            img_path = os.path.join(generated_dir, img_file)
            image = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
            
            # 编码文本
            text = clip.tokenize([prompt]).to(device)
            
            # 计算相似度
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                
                # 归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 计算余弦相似度
                similarity = (image_features @ text_features.T).item()
                scores.append(similarity)
        
        clip_score_mean = np.mean(scores)
        clip_score_std = np.std(scores)
        
        print(f"✅ CLIP Score: {clip_score_mean:.4f} ± {clip_score_std:.4f}")
        return clip_score_mean, clip_score_std
    
    except ImportError:
        print(f"❌ 未安装CLIP，跳过CLIP Score计算")
        print(f"   安装命令: pip install git+https://github.com/openai/CLIP.git")
        return None, None
    except Exception as e:
        print(f"❌ CLIP Score计算失败: {e}")
        return None, None


def calculate_lpips(generated_dir: str, real_dir: str, num_samples: int = 1000, device: str = "cuda"):
    """
    计算LPIPS (Learned Perceptual Image Patch Similarity)
    
    Args:
        generated_dir: 生成图片目录
        real_dir: 真实图片目录
        num_samples: 采样数量
        device: 设备
    
    Returns:
        lpips_mean: LPIPS均值
        lpips_std: LPIPS标准差
    """
    try:
        import lpips
        from torchvision import transforms
        
        print(f"📊 计算LPIPS...")
        
        # 初始化LPIPS模型
        loss_fn = lpips.LPIPS(net='alex').to(device)
        
        # 加载图片
        gen_files = sorted([f for f in os.listdir(generated_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        real_files = sorted([f for f in os.listdir(real_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # 采样
        import random
        random.seed(42)
        gen_files = random.sample(gen_files, min(num_samples, len(gen_files)))
        real_files = random.sample(real_files, min(num_samples, len(real_files)))
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        scores = []
        for gen_file, real_file in tqdm(zip(gen_files, real_files), total=len(gen_files), desc="计算LPIPS"):
            gen_img = Image.open(os.path.join(generated_dir, gen_file)).convert('RGB')
            real_img = Image.open(os.path.join(real_dir, real_file)).convert('RGB')
            
            gen_tensor = transform(gen_img).unsqueeze(0).to(device)
            real_tensor = transform(real_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                dist = loss_fn(gen_tensor, real_tensor)
                scores.append(dist.item())
        
        lpips_mean = np.mean(scores)
        lpips_std = np.std(scores)
        
        print(f"✅ LPIPS: {lpips_mean:.4f} ± {lpips_std:.4f}")
        return lpips_mean, lpips_std
    
    except ImportError:
        print(f"❌ 未安装lpips，跳过LPIPS计算")
        print(f"   安装命令: pip install lpips")
        return None, None
    except Exception as e:
        print(f"❌ LPIPS计算失败: {e}")
        return None, None


def calculate_basic_stats(generated_dir: str):
    """
    计算基本统计信息
    
    Args:
        generated_dir: 生成图片目录
    
    Returns:
        stats: 统计信息字典
    """
    print(f"📊 计算基本统计信息...")
    
    image_files = [f for f in os.listdir(generated_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 统计图片尺寸
    sizes = []
    for img_file in tqdm(image_files[:100], desc="采样图片尺寸"):  # 只采样100张
        img_path = os.path.join(generated_dir, img_file)
        img = Image.open(img_path)
        sizes.append(img.size)
    
    stats = {
        'num_images': len(image_files),
        'image_sizes': sizes[:10],  # 只记录前10个
        'common_size': max(set(sizes), key=sizes.count) if sizes else None
    }
    
    print(f"✅ 基本统计:")
    print(f"   图片数量: {stats['num_images']}")
    print(f"   常见尺寸: {stats['common_size']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="T2I评估脚本")
    
    # 输入路径
    parser.add_argument("--generated_dir", type=str, required=True,
                        help="生成图片目录")
    parser.add_argument("--real_dir", type=str, default=None,
                        help="真实图片目录（用于FID和LPIPS）")
    parser.add_argument("--prompts_csv", type=str, default=None,
                        help="Prompts CSV文件（用于CLIP Score）")
    
    # 评估选项
    parser.add_argument("--metrics", type=str, nargs='+', 
                        default=['fid', 'is', 'clip', 'lpips'],
                        choices=['fid', 'is', 'clip', 'lpips', 'all'],
                        help="要计算的指标")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    
    # 输出
    parser.add_argument("--output_json", type=str, default=None,
                        help="输出JSON文件路径")
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"📊 T2I评估工具")
    print(f"{'='*60}\n")
    
    # 检查输入
    if not os.path.exists(args.generated_dir):
        print(f"❌ 生成图片目录不存在: {args.generated_dir}")
        return
    
    # 处理metrics
    if 'all' in args.metrics:
        args.metrics = ['fid', 'is', 'clip', 'lpips']
    
    # 计算基本统计
    basic_stats = calculate_basic_stats(args.generated_dir)
    
    # 初始化结果
    results = {
        'generated_dir': args.generated_dir,
        'num_images': basic_stats['num_images'],
        'metrics': {}
    }
    
    # 计算FID
    if 'fid' in args.metrics:
        if args.real_dir is None:
            print(f"⚠️  跳过FID: 未提供真实图片目录")
        else:
            print(f"\n{'='*60}")
            fid_value = calculate_fid(args.generated_dir, args.real_dir, args.batch_size, args.device)
            if fid_value is not None:
                results['metrics']['fid'] = fid_value
    
    # 计算Inception Score
    if 'is' in args.metrics:
        print(f"\n{'='*60}")
        is_mean, is_std = calculate_inception_score(args.generated_dir, args.batch_size, device=args.device)
        if is_mean is not None:
            results['metrics']['inception_score'] = {
                'mean': is_mean,
                'std': is_std
            }
    
    # 计算CLIP Score
    if 'clip' in args.metrics:
        if args.prompts_csv is None:
            print(f"⚠️  跳过CLIP Score: 未提供prompts CSV")
        else:
            print(f"\n{'='*60}")
            clip_mean, clip_std = calculate_clip_score(args.generated_dir, args.prompts_csv, args.batch_size, args.device)
            if clip_mean is not None:
                results['metrics']['clip_score'] = {
                    'mean': clip_mean,
                    'std': clip_std
                }
    
    # 计算LPIPS
    if 'lpips' in args.metrics:
        if args.real_dir is None:
            print(f"⚠️  跳过LPIPS: 未提供真实图片目录")
        else:
            print(f"\n{'='*60}")
            lpips_mean, lpips_std = calculate_lpips(args.generated_dir, args.real_dir, device=args.device)
            if lpips_mean is not None:
                results['metrics']['lpips'] = {
                    'mean': lpips_mean,
                    'std': lpips_std
                }
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"✅ 评估完成!")
    print(f"{'='*60}")
    print(f"\n📊 评估结果:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # 保存结果
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果已保存到: {args.output_json}")
    else:
        # 默认保存到生成目录
        default_output = os.path.join(args.generated_dir, "evaluation_results.json")
        with open(default_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果已保存到: {default_output}")


if __name__ == "__main__":
    main()

