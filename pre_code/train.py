"""
Modified WS-DAN with Swin-Transformer-Large Backbone
"""
import os
import logging

import pandas as pd
import torch
import torch.nn.functional as func

from pre_code.net import SwinPMG
from pre_code.lr_scheduler import WarmUpCosineLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.amp import autocast, GradScaler


# 创建 logs 目录（如果没有）
os.makedirs("logs", exist_ok=True)

# 设置日志格式与文件名
logging.basicConfig(
    filename='logs/swinv2_l_192_uniatt_transform_train.log',
    filemode='a',
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)

EPSILON = 1e-12
NUM_EPOCHS = 50

def gen_cost_matrix(file_path):
    cost_df = pd.read_csv(file_path, index_col=0)  # 替换为实际路径
    cost_matrix_np = cost_df.values.astype(float)  # 确保数值为浮点型
    cost_matrix = torch.from_numpy(cost_matrix_np).float()
    return cost_matrix


def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SwinPMG(num_classes=79).to(device)
    cost_matrix_device = gen_cost_matrix("/root/FathomNet25/cost_metrix.csv").to(device)

    # 参数分组
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not "backbone" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    # 学习率和权重衰减分别设置
    optimizer = torch.optim.SGD([
        {"params": backbone_params, "lr": 2e-3, "weight_decay": 5e-4},  # backbone
        {"params": head_params, "lr": 2e-2, "weight_decay": 1e-4},  # head
    ], momentum=0.9)

    scheduler = WarmUpCosineLR(
        optimizer,
        {
            'T_0': NUM_EPOCHS, 'T_MULTI': 1, 'DELAY': 0,
            'LAST_EPOCH': -1, 'VERBOSE': True,
        },
        {
            'ENABLED': False, 'MODE': 'linear', 'SIZE': 5,
            'FINISH_LR': 0.0, 'VALUE': 2.0e-4, 'MIN_VALUE': 2.0e-4,
        }
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.1)  # 随机擦除
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448, ),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=0.1, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=79)

    train_dataset = ImageFolder('/root/autodl-tmp/dataset/train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    valid_dataset = ImageFolder('/root/autodl-tmp/dataset/val', transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=False)

    loss_func = SoftTargetCrossEntropy().to(device)
    save_dir = "/root/autodl-tmp/saved_models"
    os.makedirs(save_dir, exist_ok=True)

    scaler = GradScaler()  # 梯度缩放器

    best_val_loss = 1000.0
    # 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_cost = 0.0
        # 训练阶段带进度条
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', leave=False)
        for images, labels in train_progress:
            images, pre_labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                images, labels = mixup_fn(images, pre_labels)
                p1, p2, p3, pc = model(images)
                loss = loss_func(p1, labels) + loss_func(p2, labels) + loss_func(p3, labels)+ loss_func(pc, labels)
                loss = loss / 4.0

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                preds = (p1 + p2 + p3 + pc).argmax(dim=-1)
                batch_cost = cost_matrix_device[pre_labels, preds].mean()
                train_cost += batch_cost.item()

            train_loss += loss.item()
            train_progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cost': f"{batch_cost.item():.2f}"  # 显示原始代价
            })

        # 验证阶段
        val_progress = tqdm(valid_loader, desc='Validating', leave=False)
        with torch.inference_mode():
            model.eval()
            val_loss = 0.0
            val_cost = 0.0

            for images, labels in val_progress:
                images, labels = images.to(device), labels.to(device)
                p1, p2, p3, pc = model(images)
                logits = p1 + p2 + p3 + pc

                # 计算损失和代价
                loss = func.cross_entropy(logits, labels)
                preds = logits.argmax(dim=1)
                batch_cost = cost_matrix_device[labels, preds].mean()

                val_loss += loss.item()
                val_cost += batch_cost.item()
        scheduler.step(epoch)

        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_train_cost = train_cost / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        avg_val_cost = val_cost / len(valid_loader)

        if best_val_loss > val_loss:
            state_dict = model.state_dict()
            torch.save(
                state_dict,
                os.path.join(save_dir, 'SwinPMG_SD_E%d.pkl' % (epoch, ),)
            )

        # 更新学习率
        # 打印epoch结果
        current_lr = optimizer.param_groups[0]['lr']
        tqdm.write(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} | "
                   f"Train Loss: {avg_train_loss:.4f} | "
                   f"Train Cost: {avg_train_cost:.4f} | "
                   f"Val Loss: {avg_val_loss:.4f} | "
                   f"Val Cost: {avg_val_cost:.4f} | "
                   f"LR: {current_lr:.2e}")
        logging.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} | "
                     f"Train Loss: {avg_train_loss:.4f} | "
                     f"Train Cost: {avg_train_cost:.4f} | "
                     f"Val Loss: {avg_val_loss:.4f} | "
                     f"Val Cost: {avg_val_cost:.4f} | "
                     f"LR: {current_lr:.2e}")

if __name__ == '__main__':
    train()