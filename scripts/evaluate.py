import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from tqdm import tqdm

from utils.dataset import FER2013Dataset
from utils.transforms import get_val_transforms
from models import get_resnet_model, get_mobilenet_model

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


def load_checkpoint(checkpoint_path, device):
    """加载训练好的模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 构建模型
    model_name = config['model']
    num_classes = config.get('num_classes', 7)
    
    if model_name == 'resnet':
        model = get_resnet_model(num_classes=num_classes, pretrained=False)
    elif model_name == 'mobilenet':
        model = get_mobilenet_model(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 加载权重
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, config


def evaluate(model, data_loader, device):
    """在数据集上进行评估"""
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)


def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # 在单元格中显示数值
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(
                j, i, cm[i, j],
                ha="center", va="center", color="black", fontsize=10
            )
    
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    # 创建输出目录
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='评估情绪识别模型')
    parser.add_argument(
        '--model',
        type=str,
        choices=['resnet', 'mobilenet'],
        help='模型名称（用于自动查找检查点）'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='模型检查点路径（优先）'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='data/processed',
        help='数据集根目录'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='批次大小'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='数据加载线程数'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/tables',
        help='结果输出目录'
    )
    
    args = parser.parse_args()
    
    # 自动根据模型选择默认 checkpoint 路径
    if args.checkpoint is None:
        if args.model is None:
            print("错误: 请指定 --checkpoint 或 --model")
            return
        args.checkpoint = f'checkpoints/{args.model}/best_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型文件 {args.checkpoint} 不存在")
        return
    
    # 加载模型
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_checkpoint(args.checkpoint, device)
    print(f"Model: {config['model']}")
    
    # 构建验证集
    print("Loading validation dataset...")
    val_dataset = FER2013Dataset(
        root=args.data_root,
        mode='val',
        transform=get_val_transforms()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 进行评估
    print("Evaluating...")
    predictions, labels = evaluate(model, val_loader, device)
    
    # 计算指标
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, predictions)
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*60)
    
    # 按类别的详细指标
    print("\n按类别的详细指标:")
    print("="*60)
    class_names = [EMOTION_MAP[i] for i in range(7)]
    print(classification_report(labels, predictions, target_names=class_names, zero_division=0))
    
    # 绘制混淆矩阵
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # 保存详细结果到文本文件
    results_text_path = output_dir / 'evaluation_results.txt'
    with open(results_text_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("情绪识别模型评估结果\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {config['model']}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"\nOverall Metrics:\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("Per-Class Metrics:\n")
        f.write("="*60 + "\n")
        f.write(classification_report(labels, predictions, target_names=class_names, zero_division=0))
    
    print(f"\n结果已保存到 {results_text_path}")
    print("\n评估完成！")


if __name__ == '__main__':
    main()
