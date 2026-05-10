import torch
import torch.nn as nn
from torchvision import models


def get_resnet_model(num_classes=7, pretrained=True, dropout=0.5):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(512, num_classes)
    )
    return model


if __name__ == '__main__':
    model = get_resnet_model(num_classes=7, pretrained=False)
    print(f"Model: {model.__class__.__name__}")
    print(f"Output classes: {model.fc[-1].out_features}")
    print(f"Dropout p: {model.fc[0].p}")

    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")