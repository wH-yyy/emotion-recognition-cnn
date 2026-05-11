import argparse
import os
import sys
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.transforms import get_val_transforms
from models import get_resnet_model, get_mobilenet_model

EMOTION_MAP = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise',
}

EMOTION_LABELS_ZH = {
    'angry': '生气',
    'disgust': '厌恶',
    'fear': '恐惧',
    'happy': '开心',
    'neutral': '中立',
    'sad': '悲伤',
    'surprise': '惊讶',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}
transform = None

CUSTOM_CSS = """
:root {
    --color-primary: #2563eb;
    --color-primary-hover: #1d4ed8;
    --color-neutral-50: #f9fafb;
    --color-neutral-100: #f3f4f6;
    --color-neutral-200: #e5e7eb;
    --color-neutral-300: #d1d5db;
    --color-neutral-500: #6b7280;
    --color-neutral-700: #374151;
    --color-neutral-900: #111827;
    --text-xs: 12px;
    --text-sm: 14px;
    --text-base: 16px;
    --text-lg: 20px;
    --text-xl: 24px;
    --text-2xl: 32px;
    --space-2: 8px;
    --space-3: 12px;
    --space-4: 16px;
    --space-6: 24px;
    --space-8: 32px;
    --radius-sm: 6px;
    --radius-md: 8px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
}

body {
    background: var(--color-neutral-50);
    color: var(--color-neutral-900);
    font-size: var(--text-sm);
    line-height: 1.5;
}

h1 {
    font-size: var(--text-2xl);
    font-weight: 600;
    line-height: 1.25;
    color: var(--color-neutral-900);
    margin-bottom: var(--space-2);
}

h3 {
    font-size: var(--text-lg);
    font-weight: 600;
    line-height: 1.25;
    color: var(--color-neutral-700);
    margin-bottom: var(--space-3);
}

button.primary {
    background: var(--color-primary) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-size: var(--text-sm) !important;
    font-weight: 500 !important;
    padding: var(--space-2) var(--space-6) !important;
    color: #ffffff !important;
    cursor: pointer;
    transition: background 0.15s ease;
}

button.primary:hover {
    background: var(--color-primary-hover) !important;
}

.gr-box, .gr-form {
    background: #ffffff;
    border: 1px solid var(--color-neutral-200);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-sm);
}

footer {
    font-size: var(--text-xs);
    color: var(--color-neutral-500);
    border-top: 1px solid var(--color-neutral-200);
    padding-top: var(--space-4);
    margin-top: var(--space-8);
}
"""

MODEL_BUILDERS = {
    'resnet': lambda nc: get_resnet_model(num_classes=nc, pretrained=False),
    'mobilenet': lambda nc: get_mobilenet_model(num_classes=nc, pretrained=False),
}


def _load_one(name: str, checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        print(f"  Skipping {name}: checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    num_classes = config.get('num_classes', 7)

    builder = MODEL_BUILDERS.get(name)
    if builder is None:
        print(f"  Skipping {name}: unknown model type")
        return

    m = builder(num_classes)
    m.load_state_dict(checkpoint['model'])
    m.to(device)
    m.eval()
    models[name] = m
    print(f"  {name} loaded")


def load_all_models():
    checkpoints_dir = Path(project_root) / 'checkpoints'
    for name in ('resnet', 'mobilenet'):
        path = checkpoints_dir / name / 'best_model.pth'
        _load_one(name, str(path))

    if not models:
        raise RuntimeError(
            "No models loaded. Place checkpoints at "
            "checkpoints/resnet/best_model.pth or checkpoints/mobilenet/best_model.pth"
        )


def predict_emotion(image, model_name: str):
    m = models.get(model_name)
    if m is None:
        return {"error": f"Model '{model_name}' not loaded"}

    if image is None:
        return {"error": "Please upload an image"}

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = m(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

    predicted_class = torch.argmax(outputs, dim=1).item()
    predicted_emotion = EMOTION_MAP[predicted_class]
    predicted_prob = probabilities[0, predicted_class].item()

    output_dict = {predicted_emotion: float(predicted_prob)}
    for class_id in range(7):
        emotion_name = EMOTION_MAP[class_id]
        if emotion_name not in output_dict:
            output_dict[emotion_name] = float(probabilities[0, class_id].item())

    return output_dict


def create_interface(available_models: list):
    theme = gr.themes.Monochrome(
        primary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(
        title="Facial Emotion Recognition",
        theme=theme,
        css=CUSTOM_CSS,
    ) as demo:
        gr.Markdown("# Facial Emotion Recognition")
        gr.Markdown(
            "Upload a facial image and the model will classify it into one of seven emotion categories."
        )

        model_selector = gr.Dropdown(
            choices=available_models,
            value=available_models[0],
            label="Model",
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Input Image", sources=["upload"])

            with gr.Column(scale=1):
                emotion_output = gr.Label(label="Prediction", num_top_classes=7)

        predict_btn = gr.Button("Classify", variant="primary", size="sm")
        predict_btn.click(
            fn=predict_emotion,
            inputs=[image_input, model_selector],
            outputs=emotion_output,
        )

        example_images = []
        demo_dir = Path(project_root) / "app" / "demo_images"
        if demo_dir.exists():
            for ext in ("*.jpg", "*.png"):
                example_images.extend(str(p) for p in sorted(demo_dir.glob(ext)))

        if example_images:
            gr.Examples(
                examples=[[img] for img in example_images[:6]],
                inputs=image_input,
                outputs=emotion_output,
                fn=predict_emotion,
                cache_examples=False,
            )

        labels_desc = " | ".join(
            f"{en} ({zh})" for en, zh in EMOTION_LABELS_ZH.items()
        )
        gr.Markdown(
            f"**Categories**: {labels_desc}\n\n"
            f"**Tech stack**: PyTorch + ResNet / MobileNet + Gradio"
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Single checkpoint path (legacy mode, skips auto-discovery)",
    )
    args = parser.parse_args()

    global transform
    transform = get_val_transforms()

    if args.checkpoint:
        print("Loading model from single checkpoint...")
        _load_one('custom', args.checkpoint)
    else:
        print("Loading models...")
        load_all_models()

    if not models:
        print("No models available. Exiting.")
        return

    print(f"Available models: {list(models.keys())}")
    print(f"Device: {device}")

    demo = create_interface(list(models.keys()))
    print("\nStarting Gradio server...")
    demo.launch()


if __name__ == "__main__":
    main()