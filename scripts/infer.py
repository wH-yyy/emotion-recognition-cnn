import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

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
    
    return model


def load_and_preprocess_image(image_path, transform):
    """加载并预处理图片"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 打开图片
    image = Image.open(image_path).convert('RGB')
    
    # 应用变换
    image_tensor = transform(image)
    
    # 添加 batch 维度
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def infer(model, image_tensor, device):
    """进行推理"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        # 获取概率
        probabilities = F.softmax(outputs, dim=1)
        # 获取预测类别
        predicted_class = torch.argmax(outputs, dim=1).item()
        predicted_prob = probabilities[0, predicted_class].item()
    
    return predicted_class, probabilities[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='人脸情绪识别推理')
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='输入图片路径'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='模型检查点路径'
    )
    parser.add_argument(
        '--show_all_probs',
        action='store_true',
        help='是否显示所有类别的概率'
    )
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 检查文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型文件 {args.checkpoint} 不存在")
        return
    
    if not os.path.exists(args.image):
        print(f"错误: 图片文件 {args.image} 不存在")
        return
    
    # 加载模型
    print(f"Loading model from {args.checkpoint}...")
    model = load_checkpoint(args.checkpoint, device)
    model.eval()
    
    # 获取预处理变换
    transform = get_val_transforms()
    
    # 加载并预处理图片
    print(f"Loading image from {args.image}...")
    image_tensor = load_and_preprocess_image(args.image, transform)
    
    # 推理
    print("Inferring...")
    predicted_class, probabilities = infer(model, image_tensor, device)
    
    predicted_emotion = EMOTION_MAP[predicted_class]
    predicted_prob = probabilities[predicted_class]
    
    # 打印结果
    print("\n" + "="*60)
    print("推理结果")
    print("="*60)
    print(f"图片路径: {args.image}")
    print(f"预测情绪: {predicted_emotion} (类别: {predicted_class})")
    print(f"置信度:   {predicted_prob:.4f} ({predicted_prob*100:.2f}%)")
    
    # 显示所有类别的概率
    if args.show_all_probs:
        print("\n各情绪概率:")
        print("-"*60)
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices:
            emotion_name = EMOTION_MAP[idx]
            prob = probabilities[idx]
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"{emotion_name:10s} | {bar} | {prob:.4f}")
    
    print("="*60)


if __name__ == '__main__':
    main()
