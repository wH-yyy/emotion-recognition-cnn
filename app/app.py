import os
import sys
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

# 全局变量：存储模型和设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
transform = None


def load_model(checkpoint_path='checkpoints/best_model.pth'):
    """加载模型"""
    global model
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
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


def predict_emotion(image):
    """
    预测图片中的情绪
    
    Args:
        image: PIL Image 对象
    
    Returns:
        output_dict: 包含预测结果和概率的字典
    """
    if model is None:
        return {"error": "模型未加载"}
    
    if image is None:
        return {"error": "请上传一张图片"}
    
    # 确保图片是 RGB 格式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 预处理图片
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    # 获取预测结果
    predicted_class = torch.argmax(outputs, dim=1).item()
    predicted_emotion = EMOTION_MAP[predicted_class]
    predicted_prob = probabilities[0, predicted_class].item()
    
    # 构建输出字典
    output_dict = {
        predicted_emotion: float(predicted_prob)
    }
    
    # 添加其他类别的概率
    for class_id in range(7):
        emotion_name = EMOTION_MAP[class_id]
        if emotion_name not in output_dict:
            output_dict[emotion_name] = float(probabilities[0, class_id].item())
    
    return output_dict


def create_interface():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="人脸情绪识别系统", theme=gr.themes.Soft()) as demo:
        # 标题
        gr.Markdown("# 🎭 人脸情绪识别系统")
        gr.Markdown("上传一张人脸图片，模型将预测其情绪类别")
        
        with gr.Row():
            # 左侧：输入
            with gr.Column(scale=1):
                gr.Markdown("### 上传图片")
                image_input = gr.Image(
                    type="pil",
                    label="上传人脸图片",
                    scale=1
                )
            
            # 右侧：输出
            with gr.Column(scale=1):
                gr.Markdown("### 预测结果")
                emotion_output = gr.Label(
                    label="情绪概率分布",
                    num_top_classes=7
                )
        
        # 预测按钮
        predict_btn = gr.Button("预测情绪", variant="primary", size="lg")
        
        # 绑定事件
        predict_btn.click(
            fn=predict_emotion,
            inputs=image_input,
            outputs=emotion_output
        )
        
        # 示例图片
        gr.Markdown("### 示例")
        example_images = []
        demo_dir = Path(project_root) / "app" / "demo_images"
        if demo_dir.exists():
            for img_path in sorted(demo_dir.glob("*.jpg")) + sorted(demo_dir.glob("*.png")):
                example_images.append(str(img_path))
        
        if example_images:
            gr.Examples(
                examples=[[img] for img in example_images[:6]],
                inputs=image_input,
                outputs=emotion_output,
                fn=predict_emotion,
                cache_examples=False,
                label="点击示例进行推理"
            )
        
        # 页脚
        gr.Markdown("""
        ---
        **技术栈**: PyTorch + ResNet/MobileNet + Gradio
        
        **支持的情绪类别**: angry(生气) | disgust(厌恶) | fear(恐惧) | happy(开心) | neutral(中立) | sad(悲伤) | surprise(惊讶)
        """)
    
    return demo


def main():
    # 加载模型
    checkpoint_path = str(Path(project_root) / 'checkpoints' / 'best_model.pth')
    
    print("加载模型中...")
    try:
        load_model(checkpoint_path)
        global transform
        transform = get_val_transforms()
        print(f"✓ 模型加载成功 (设备: {device})")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print(f"  请确保在 {checkpoint_path} 处有训练好的模型")
        return
    
    # 创建并启动界面
    demo = create_interface()
    
    print("\n" + "="*60)
    print("🚀 Gradio Web 应用启动中...")
    print("="*60)
    
    demo.launch()


if __name__ == "__main__":
    main()
