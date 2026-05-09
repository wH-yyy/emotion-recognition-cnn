import os
from pathlib import Path
from typing import Callable, Optional, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_transform(mode: str = 'train') -> transforms.Compose:
    """返回图像预处理变换，train/val 共用 resize + normalize。"""
    if mode not in ['train', 'val']:
        raise ValueError("mode must be 'train' or 'val'")

    transform_list = [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    if mode == 'train':
        transform_list = [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    return transforms.Compose(transform_list)


class FER2013Dataset(Dataset):
    """自定义 Dataset，用于加载 FER2013 处理后的人脸情绪图像。"""

    def __init__(
        self,
        root: str,
        mode: str = 'train',
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            root: 数据集根目录，例如 data/processed
            mode: 数据模式，'train' 或 'val'
            transform: 可选的 torchvision.transforms 组合
        """
        if mode not in ['train', 'val']:
            raise ValueError("mode must be 'train' or 'val'")

        self.root = Path(root)
        self.mode = mode
        self.transform = transform if transform is not None else get_transform(mode)
        self.images: List[Path] = []
        self.labels: List[int] = []

        self._load_samples()

    def _load_samples(self) -> None:
        """扫描目录并准备样本列表。"""
        data_dir = self.root / self.mode
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        for label_dir in sorted(data_dir.iterdir()):
            if not label_dir.is_dir():
                continue

            label_name = label_dir.name
            if not label_name.startswith('label_'):
                continue

            try:
                label = int(label_name.split('_')[-1])
            except ValueError:
                continue

            for image_path in sorted(label_dir.glob('*.png')):
                self.images.append(image_path)
                self.labels.append(label)

        if len(self.images) == 0:
            raise RuntimeError(f"未发现任何图片: {data_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


def build_dataset(root: str, mode: str = 'train') -> FER2013Dataset:
    """构建并返回 FER2013 数据集对象。"""
    return FER2013Dataset(root=root, mode=mode)
