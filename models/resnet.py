import torch
import torch.nn as nn
from torchvision import models


def get_resnet_model(num_classes=7, pretrained=True):
    """
    创建基于 ResNet18 的人脸情绪识别模型。

    Args:
        num_classes (int): 分类类别数，默认 7 类情绪
        pretrained (bool): 是否使用预训练权重，默认 True

    Returns:
        nn.Module: 修改后的 ResNet18 模型
    """
    # 加载 ResNet18 模型
    model = models.resnet18(pretrained=pretrained)

    # 修改最后的全连接层
    # ResNet18 的 fc 层输入特征数为 512
    model.fc = nn.Linear(512, num_classes)

    return model


if __name__ == '__main__':
    # 测试模型创建
    model = get_resnet_model(num_classes=7, pretrained=False)
    print(f"Model: {model.__class__.__name__}")
    print(f"Output classes: {model.fc.out_features}")

    # 测试前向传播
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")