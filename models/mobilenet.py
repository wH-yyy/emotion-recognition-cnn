import torch
import torch.nn as nn
from torchvision import models


def get_mobilenet_model(num_classes=7, pretrained=True):
    """
    创建基于 MobileNetV2 的人脸情绪识别模型。

    Args:
        num_classes (int): 分类类别数，默认 7 类情绪
        pretrained (bool): 是否使用预训练权重，默认 True

    Returns:
        nn.Module: 修改后的 MobileNetV2 模型
    """
    # 加载 MobileNetV2 模型
    model = models.mobilenet_v2(pretrained=pretrained)

    # 修改最后的分类器
    # MobileNetV2 的 classifier 是 Sequential，最后一层是 Linear(1280, 1000)
    model.classifier[-1] = nn.Linear(1280, num_classes)

    return model


if __name__ == '__main__':
    # 测试模型创建
    model = get_mobilenet_model(num_classes=7, pretrained=False)
    print(f"Model: {model.__class__.__name__}")
    print(f"Output classes: {model.classifier[-1].out_features}")

    # 测试前向传播
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")