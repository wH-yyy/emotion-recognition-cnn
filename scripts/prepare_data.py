"""
FER2013 数据集转换脚本
将 CSV 格式的 FER2013 数据集转换为图片文件夹格式
"""

import os
import csv
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

# 情绪标签映射
EMOTION_MAP = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}


def create_directories(output_dir):
    """创建输出目录结构"""
    splits = ['train', 'val', 'test']
    for split in splits:
        for emotion_id in range(7):
            emotion_name = EMOTION_MAP[emotion_id]
            dir_path = os.path.join(output_dir, split, f'label_{emotion_id}')
            os.makedirs(dir_path, exist_ok=True)


def process_fer2013(input_csv, output_dir):
    """
    处理 FER2013 CSV 文件并转换为图片格式
    
    Args:
        input_csv: FER2013 CSV 文件路径
        output_dir: 输出目录路径
    """
    # 创建输出目录结构
    create_directories(output_dir)
    
    # 统计计数
    stats = {split: {i: 0 for i in range(7)} for split in ['train', 'val', 'test']}
    
    # 映射 Usage 列到 split 目录
    usage_to_split = {
        'Training': 'train',
        'PublicTest': 'val',
        'PrivateTest': 'test'
    }
    
    print("开始处理 FER2013 数据集...")
    
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        total_rows = sum(1 for _ in open(input_csv)) - 1  # 排除 header
        
        # 重新打开文件以进行处理
        f.seek(0)
        reader = csv.DictReader(f)
        
        for row_idx, row in enumerate(tqdm(reader, total=total_rows, desc="转换中")):
            try:
                emotion_id = int(row['emotion'])
                pixels_str = row['pixels']
                usage = row['Usage']
                
                # 获取 split 类型
                split = usage_to_split.get(usage, 'train')
                
                # 将像素字符串转换为 numpy 数组
                pixels = np.array([int(p) for p in pixels_str.split(' ')], dtype=np.uint8)
                
                # 重塑为 48x48 灰度图
                image_array = pixels.reshape((48, 48))
                
                # 转换为 PIL Image
                image = Image.fromarray(image_array, mode='L')
                
                # 构造输出路径
                emotion_name = EMOTION_MAP[emotion_id]
                output_path = os.path.join(
                    output_dir,
                    split,
                    f'label_{emotion_id}',
                    f'{row_idx}.png'
                )
                
                # 保存图片
                image.save(output_path)
                stats[split][emotion_id] += 1
                
            except Exception as e:
                print(f"处理第 {row_idx} 行时出错: {e}")
                continue
    
    # 打印统计信息
    print("\n转换完成！统计信息：")
    for split in ['train', 'val', 'test']:
        total = sum(stats[split].values())
        print(f"\n{split.upper()}:")
        for emotion_id in range(7):
            emotion_name = EMOTION_MAP[emotion_id]
            count = stats[split][emotion_id]
            print(f"  {emotion_name}: {count}")
        print(f"  总计: {total}")


def main():
    parser = argparse.ArgumentParser(description='FER2013 数据集转换脚本')
    parser.add_argument(
        '--input',
        default='data/raw/fer2013.csv',
        help='FER2013 CSV 文件路径'
    )
    parser.add_argument(
        '--output',
        default='data/processed',
        help='输出目录路径'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在")
        return
    
    process_fer2013(args.input, args.output)


if __name__ == '__main__':
    main()
