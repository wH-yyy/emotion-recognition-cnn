import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_history(history_path):
    """加载训练历史数据"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history


def plot_loss_curve(history, save_path):
    """绘制 loss 曲线"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3)
    plt.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss', marker='s', markersize=3)
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Loss curve saved to {save_path}")
    plt.close()


def plot_accuracy_curve(history, save_path):
    """绘制 accuracy 曲线"""
    epochs = range(1, len(history['train_acc']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc', marker='o', markersize=3)
    plt.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc', marker='s', markersize=3)
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Training Accuracy Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Accuracy curve saved to {save_path}")
    plt.close()


def plot_combined_curve(history, save_path):
    """绘制组合曲线（loss 和 accuracy）"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3)
    axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[0].set_title('Training Loss Curve', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc', marker='o', markersize=3)
    axes[1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[1].set_title('Training Accuracy Curve', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Combined curve saved to {save_path}")
    plt.close()


def print_statistics(history):
    """打印训练统计信息"""
    print("\n" + "="*60)
    print("训练统计信息")
    print("="*60)
    
    num_epochs = len(history['train_loss'])
    print(f"总训练轮数: {num_epochs}")
    
    # Loss 统计
    min_train_loss = min(history['train_loss'])
    min_val_loss = min(history['val_loss'])
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    print(f"\nLoss:")
    print(f"  Train Loss - 最小值: {min_train_loss:.4f}, 最终值: {final_train_loss:.4f}")
    print(f"  Val Loss   - 最小值: {min_val_loss:.4f}, 最终值: {final_val_loss:.4f}")
    
    # Accuracy 统计
    max_train_acc = max(history['train_acc'])
    max_val_acc = max(history['val_acc'])
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    
    print(f"\nAccuracy:")
    print(f"  Train Acc - 最高值: {max_train_acc:.4f}, 最终值: {final_train_acc:.4f}")
    print(f"  Val Acc   - 最高值: {max_val_acc:.4f}, 最终值: {final_val_acc:.4f}")
    
    # 找最佳模型
    best_val_acc_idx = history['val_acc'].index(max_val_acc) + 1
    print(f"\n最佳模型: 第 {best_val_acc_idx} 个 epoch (Val Acc: {max_val_acc:.4f})")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='绘制训练曲线')
    parser.add_argument(
        '--history',
        type=str,
        default='checkpoints/history.json',
        help='训练历史 JSON 文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/curves',
        help='输出目录'
    )
    parser.add_argument(
        '--combined',
        action='store_true',
        help='是否绘制组合曲线'
    )
    
    args = parser.parse_args()
    
    history_path = Path(args.history)
    output_dir = Path(args.output_dir)
    
    # 检查历史文件是否存在
    if not history_path.exists():
        print(f"错误: 历史文件不存在 {history_path}")
        return
    
    # 加载历史数据
    print(f"加载训练历史从 {history_path}...")
    history = load_history(history_path)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制曲线
    print("正在生成曲线图...")
    
    if args.combined:
        # 绘制组合曲线
        combined_save_path = output_dir / 'training_curves_combined.png'
        plot_combined_curve(history, combined_save_path)
    else:
        # 分别绘制
        loss_save_path = output_dir / 'loss_curve.png'
        plot_loss_curve(history, loss_save_path)
        
        acc_save_path = output_dir / 'accuracy_curve.png'
        plot_accuracy_curve(history, acc_save_path)
    
    # 打印统计信息
    print_statistics(history)
    
    print("\n✓ 曲线绘制完成!")


if __name__ == '__main__':
    main()
