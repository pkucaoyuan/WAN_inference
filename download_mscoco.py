"""
下载MS-COCO数据集和prompts
用于T2I评估
"""
import os
import argparse
import requests
import zipfile
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from urllib.request import urlretrieve


def download_file(url: str, output_path: str, description: str = "Downloading"):
    """下载文件并显示进度条"""
    print(f"📥 {description}...")
    print(f"   URL: {url}")
    print(f"   保存到: {output_path}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✅ 下载完成: {output_path}")


def extract_zip(zip_path: str, extract_to: str):
    """解压ZIP文件"""
    print(f"📦 解压文件: {zip_path}")
    print(f"   解压到: {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"✅ 解压完成")


def download_mscoco_val2014(output_dir: str):
    """
    下载MS-COCO 2014验证集图片
    
    Args:
        output_dir: 输出目录
    """
    # MS-COCO 2014验证集URL
    val2014_url = "http://images.cocodataset.org/zips/val2014.zip"
    
    # 下载路径
    zip_path = os.path.join(output_dir, "val2014.zip")
    
    # 下载
    if not os.path.exists(zip_path):
        download_file(val2014_url, zip_path, "下载MS-COCO val2014图片")
    else:
        print(f"⏭️  文件已存在，跳过下载: {zip_path}")
    
    # 解压
    extract_dir = os.path.join(output_dir, "images")
    if not os.path.exists(os.path.join(extract_dir, "val2014")):
        extract_zip(zip_path, extract_dir)
    else:
        print(f"⏭️  目录已存在，跳过解压: {extract_dir}/val2014")
    
    return os.path.join(extract_dir, "val2014")


def download_mscoco_annotations(output_dir: str):
    """
    下载MS-COCO 2014标注文件
    
    Args:
        output_dir: 输出目录
    """
    # MS-COCO 2014标注URL
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    
    # 下载路径
    zip_path = os.path.join(output_dir, "annotations_trainval2014.zip")
    
    # 下载
    if not os.path.exists(zip_path):
        download_file(annotations_url, zip_path, "下载MS-COCO标注文件")
    else:
        print(f"⏭️  文件已存在，跳过下载: {zip_path}")
    
    # 解压
    extract_dir = output_dir
    annotations_path = os.path.join(extract_dir, "annotations", "captions_val2014.json")
    if not os.path.exists(annotations_path):
        extract_zip(zip_path, extract_dir)
    else:
        print(f"⏭️  文件已存在，跳过解压: {annotations_path}")
    
    return annotations_path


def create_prompts_csv(annotations_path: str, output_csv_path: str, num_samples: int = None):
    """
    从MS-COCO标注文件创建prompts CSV
    
    Args:
        annotations_path: 标注JSON文件路径
        output_csv_path: 输出CSV路径
        num_samples: 采样数量（None表示全部）
    """
    print(f"📝 创建prompts CSV...")
    print(f"   读取标注: {annotations_path}")
    
    # 读取标注
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 提取captions
    # 每张图片有多个caption，这里只取第一个
    image_to_caption = {}
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_caption:
            image_to_caption[image_id] = ann['caption']
    
    # 创建DataFrame
    data = []
    for image_id, caption in image_to_caption.items():
        data.append({
            'image_id': image_id,
            'prompt': caption
        })
    
    df = pd.DataFrame(data)
    
    # 采样
    if num_samples is not None:
        df = df.sample(n=min(num_samples, len(df)), random_state=42)
        df = df.reset_index(drop=True)
    
    # 保存CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"✅ Prompts CSV已创建: {output_csv_path}")
    print(f"   共 {len(df)} 条prompts")
    print(f"\n示例prompts:")
    for i in range(min(3, len(df))):
        print(f"   [{i+1}] {df.iloc[i]['prompt']}")


def download_fid_stats(output_dir: str):
    """
    下载预计算的FID统计文件（如果可用）
    
    Args:
        output_dir: 输出目录
    """
    # 常用的FID统计文件URL（来自pytorch-fid或其他来源）
    # 这里提供一些常见的预计算统计文件
    
    stats_urls = {
        "mscoco_val2014_inception_v3.npz": 
            "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"
    }
    
    print(f"📊 下载FID统计文件...")
    print(f"   注意: 如果没有预计算的统计文件，评估脚本会自动计算")
    
    stats_dir = os.path.join(output_dir, "fid_stats")
    os.makedirs(stats_dir, exist_ok=True)
    
    # 这里只下载Inception模型权重，实际的统计文件需要从真实数据计算
    inception_weights_path = os.path.join(stats_dir, "pt_inception-2015-12-05-6726825d.pth")
    
    if not os.path.exists(inception_weights_path):
        print(f"ℹ️  FID统计文件将在评估时自动计算")
    else:
        print(f"✅ FID统计文件已存在")
    
    return stats_dir


def main():
    parser = argparse.ArgumentParser(description="下载MS-COCO数据集和prompts")
    
    parser.add_argument("--output_dir", type=str, default="./mscoco_data",
                        help="输出目录")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="采样prompts数量（默认全部，约40k）")
    parser.add_argument("--skip_images", action="store_true",
                        help="跳过下载图片（只下载标注和创建prompts CSV）")
    parser.add_argument("--skip_fid_stats", action="store_true",
                        help="跳过下载FID统计文件")
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"📦 MS-COCO数据下载工具")
    print(f"{'='*60}\n")
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 下载标注文件
    print(f"\n{'='*60}")
    print(f"步骤 1/4: 下载标注文件")
    print(f"{'='*60}")
    annotations_path = download_mscoco_annotations(output_dir)
    
    # 2. 创建prompts CSV
    print(f"\n{'='*60}")
    print(f"步骤 2/4: 创建prompts CSV")
    print(f"{'='*60}")
    prompts_csv_path = os.path.join(output_dir, "prompts.csv")
    create_prompts_csv(annotations_path, prompts_csv_path, args.num_samples)
    
    # 3. 下载验证集图片（可选）
    if not args.skip_images:
        print(f"\n{'='*60}")
        print(f"步骤 3/4: 下载验证集图片")
        print(f"{'='*60}")
        val2014_dir = download_mscoco_val2014(output_dir)
    else:
        print(f"\n⏭️  跳过下载图片")
        val2014_dir = None
    
    # 4. 下载FID统计文件（可选）
    if not args.skip_fid_stats:
        print(f"\n{'='*60}")
        print(f"步骤 4/4: 准备FID统计")
        print(f"{'='*60}")
        stats_dir = download_fid_stats(output_dir)
    else:
        print(f"\n⏭️  跳过FID统计文件")
        stats_dir = None
    
    # 输出总结
    print(f"\n{'='*60}")
    print(f"✅ 下载完成!")
    print(f"{'='*60}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📝 Prompts CSV: {prompts_csv_path}")
    if val2014_dir:
        print(f"🖼️  验证集图片: {val2014_dir}")
    if stats_dir:
        print(f"📊 FID统计目录: {stats_dir}")
    
    print(f"\n🚀 下一步:")
    print(f"   1. 使用 batch_generate_t2i.py 生成图片:")
    print(f"      python batch_generate_t2i.py \\")
    print(f"          --prompts_csv {prompts_csv_path} \\")
    print(f"          --output_dir ./generated_images \\")
    print(f"          --num_samples 1000")
    print(f"\n   2. 使用 evaluate_t2i.py 计算评估指标:")
    print(f"      python evaluate_t2i.py \\")
    print(f"          --generated_dir ./generated_images \\")
    print(f"          --real_dir {val2014_dir if val2014_dir else './mscoco_data/images/val2014'} \\")
    print(f"          --prompts_csv {prompts_csv_path}")


if __name__ == "__main__":
    main()

