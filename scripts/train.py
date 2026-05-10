import os
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.dataset import FER2013Dataset
from utils.transforms import get_train_transforms, get_val_transforms
from models import get_resnet_model, get_mobilenet_model


class Trainer:
    """情绪识别模型训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 创建输出目录
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 构建数据集
        self.train_dataset = FER2013Dataset(
            root=config['data_root'],
            mode='train',
            transform=get_train_transforms()
        )
        self.val_dataset = FER2013Dataset(
            root=config['data_root'],
            mode='val',
            transform=get_val_transforms()
        )

        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )

        # 加载模型
        self.model = self._build_model()
        self.model.to(self.device)

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # 学习率调度器（可选）
        self.scheduler = None
        if config.get('use_scheduler', False):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs']
            )

        # 记录历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        self.best_val_acc = 0.0

    def _build_model(self):
        """构建模型"""
        model_name = self.config['model']
        num_classes = self.config.get('num_classes', 7)

        if model_name == 'resnet':
            model = get_resnet_model(num_classes=num_classes, pretrained=self.config.get('pretrained', True))
        elif model_name == 'mobilenet':
            model = get_mobilenet_model(num_classes=num_classes, pretrained=self.config.get('pretrained', True))
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return model

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct / total

        return avg_loss, avg_acc

    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Val", leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = correct / total

        return avg_loss, avg_acc

    def save_checkpoint(self, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
        }

        if is_best:
            model_name = self.config['model']
            save_path = self.checkpoint_dir / model_name / 'best_model.pth'
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            save_path = self.checkpoint_dir / 'latest_model.pth'

        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.history = checkpoint['history']
        print(f"Model loaded from {checkpoint_path}")

    def plot_curves(self):
        """绘制训练曲线"""
        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss 曲线
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Train')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)

        # Accuracy 曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], 'b-', label='Train')
        plt.plot(epochs, self.history['val_acc'], 'r-', label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        save_path = self.checkpoint_dir.parent / 'results' / 'curves' / 'training_curves.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Curves saved to {save_path}")
        plt.close()

    def train(self):
        """完整训练过程"""
        num_epochs = self.config['epochs']

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")

            # 训练
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # 验证
            val_loss, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 打印统计信息
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            # 学习率调整
            if self.scheduler is not None:
                self.scheduler.step()

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(is_best=True)
                print(f"Best model updated! Val Acc: {val_acc:.4f}")

            # 保存最新模型
            self.save_checkpoint(is_best=False)

        # 绘制曲线
        self.plot_curves()

        # 保存训练历史
        model_name = self.config['model']
        history_path = self.checkpoint_dir / model_name / 'history.json'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"History saved to {history_path}")


def main():
    parser = argparse.ArgumentParser(description='训练情绪识别模型')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'mobilenet'],
                        help='模型选择')
    parser.add_argument('--data_root', type=str, default='data/processed',
                        help='数据集根目录')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='是否使用学习率调度器')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='是否使用预训练权重')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='分类类别数')

    args = parser.parse_args()
    config = vars(args)

    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
