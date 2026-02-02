# train_foundation_model.py
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 核心修改 1：导入自定义模块 ---
# 确保脚本能找到同级目录下的 utils 和 datasets
from common_utils import (
    BASE_DIR, 
    get_signal_stats, 
    adapt_vit_to_12_channels, 
    contrastive_loss
)
from datasets import ECGSignalImageDataset, collate_fn

# 导入外部依赖模块 (确保这些文件在 src 或 PYTHONPATH 中)
import clip_package as official_clip_module
from finetune_model import ft_12lead_ECGFounder

# --- 核心修改 2：基于 BASE_DIR 配置相对路径 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 数据路径
DATA_ROOT = os.path.join(BASE_DIR, "data", "心电数据集")
METADATA_FILE = os.path.join(DATA_ROOT, "mimic_metadata_patient_split", "metadata_train.csv")
VAL_METADATA_FILE = os.path.join(DATA_ROOT, "mimic_metadata_patient_split", "metadata_val.csv")
STATS_FILE = os.path.join(DATA_ROOT, "signal_stats_vit.npz")

# 权重路径
MODELS_DIR = os.path.join(BASE_DIR, "models")
ECG_SIGNAL_MODEL_PATH = os.path.join(MODELS_DIR, "12_lead_ECGFounder.pth")
CLIP_MODEL_PATH = os.path.join(MODELS_DIR, "ViT-B-32.pt")

# 保存路径
SAVE_MODEL_DIR = os.path.join(BASE_DIR, "results", "save_model")
BEST_MODEL_SAVE_PATH = os.path.join(SAVE_MODEL_DIR, "ECG-Image-FM.pth")
LATEST_CHECKPOINT_PATH = os.path.join(SAVE_MODEL_DIR, "latest_checkpoint.pth")

# --- 3. 配置参数 ---
N_CLASSES_FOR_NET1D_INIT = 5
TARGET_SIGNAL_LENGTH = 1250
NUM_SIGNAL_CHANNELS = 12
SIGNAL_FEATURE_DIM = 1024
CLIP_IMAGE_FEATURE_DIM = 512
PROJECTION_DIM = 1024

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-5
PATIENCE = 5
WEIGHT_DECAY = 0.05
TEMPERATURE = nn.Parameter(torch.tensor(np.log(1 / 0.07), device=DEVICE))

def main():
    # 确保保存目录存在
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

    # 1. 获取信号统计量 (用于归一化)
    signal_stats = get_signal_stats(STATS_FILE, METADATA_FILE)

    # 2. 初始化信号编码器 (Frozen)
    ecg_signal_model = ft_12lead_ECGFounder(DEVICE, ECG_SIGNAL_MODEL_PATH, n_classes=N_CLASSES_FOR_NET1D_INIT)
    ecg_signal_model.eval()
    for param in ecg_signal_model.parameters():
        param.requires_grad = False
    print("ECG signal model loaded and frozen.")
    
    # 3. 初始化图像编码器 (Trainable)
    clip_model, _ = official_clip_module.load(model_path=CLIP_MODEL_PATH, device="cpu", jit=False)
    clip_model = adapt_vit_to_12_channels(clip_model)
    clip_model.float().to(DEVICE)
    # 仅视觉部分参与训练
    for name, param in clip_model.named_parameters():
        param.requires_grad = name.startswith("visual.")
    print("CLIP model loaded, adapted for 12-channels, and visual encoder is trainable.")
    
    # 4. 定义预处理 (Grayscale版)
    grayscale_mean = (0.48145466 + 0.4578275 + 0.40821073) / 3
    grayscale_std = (0.26862954 + 0.26130258 + 0.27577711) / 3

    clip_image_preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(grayscale_mean,), std=(grayscale_std,))
    ])
    
    # 5. 投影层与优化器
    image_projector = nn.Linear(CLIP_IMAGE_FEATURE_DIM, PROJECTION_DIM).to(DEVICE)
    trainable_params = list(clip_model.visual.parameters()) + list(image_projector.parameters()) + [TEMPERATURE]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 6. 数据迭代器
    train_dataset = ECGSignalImageDataset(METADATA_FILE, clip_image_preprocess, TARGET_SIGNAL_LENGTH, signal_stats)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    val_dataset = ECGSignalImageDataset(VAL_METADATA_FILE, clip_image_preprocess, TARGET_SIGNAL_LENGTH, signal_stats)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 7. 断点续传初始化
    start_epoch = 0
    best_val_loss = sys.float_info.max
    epochs_without_improvement = 0
    train_loss_history, val_loss_history = [], []

    if os.path.exists(LATEST_CHECKPOINT_PATH):
        print(f"--- 发现检查点！正在从 '{LATEST_CHECKPOINT_PATH}' 恢复训练 ---")
        checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=DEVICE)
        clip_model.visual.load_state_dict(checkpoint['visual_encoder_state_dict'])
        image_projector.load_state_dict(checkpoint['image_projector_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'temperature' in checkpoint:
            TEMPERATURE.data.copy_(checkpoint['temperature'].data)
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', sys.float_info.max)
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        train_loss_history = checkpoint.get('train_loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        print(f"--- 恢复成功！从 Epoch {start_epoch + 1} 开始 ---")
    else:
        print("--- 未发现检查点，从头开始训练 ---")

    # 8. 训练循环
    clip_dtype = clip_model.dtype

    for epoch in range(start_epoch, EPOCHS):
        # --- Train ---
        clip_model.visual.train()
        image_projector.train()
        total_train_loss, num_train_batches = 0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} Train")
        
        for batch_data in pbar:
            ecg_signals, ecg_images = batch_data
            if ecg_signals is None: continue
            ecg_signals, ecg_images = ecg_signals.to(DEVICE), ecg_images.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.no_grad():
                _, signal_features = ecg_signal_model(ecg_signals)
            
            raw_image_features = clip_model.encode_image(ecg_images.to(clip_dtype)).float()
            image_features = image_projector(raw_image_features)
            
            loss = contrastive_loss(signal_features, image_features, TEMPERATURE)
            if torch.isnan(loss): continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
            pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        train_loss_history.append(avg_train_loss)

        # --- Validation ---
        clip_model.visual.eval()
        image_projector.eval()
        total_val_loss, num_val_batches = 0, 0
        with torch.no_grad():
            for batch_data in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} Val"):
                ecg_signals, ecg_images = batch_data
                if ecg_signals is None: continue
                ecg_signals, ecg_images = ecg_signals.to(DEVICE), ecg_images.to(DEVICE)
                
                _, signal_features = ecg_signal_model(ecg_signals)
                raw_image_features = clip_model.encode_image(ecg_images.to(clip_dtype)).float()
                image_features = image_projector(raw_image_features)
                
                v_loss = contrastive_loss(signal_features, image_features, TEMPERATURE)
                if not torch.isnan(v_loss):
                    total_val_loss += v_loss.item()
                    num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        val_loss_history.append(avg_val_loss)
        print(f"--- Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Temp: {TEMPERATURE.exp().item():.4f} ---")

        # 早停判断
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            print(f"!!! New best model! Saving to {BEST_MODEL_SAVE_PATH} !!!")
            torch.save({
                'visual_encoder_state_dict': clip_model.visual.state_dict(),
                'image_projector_state_dict': image_projector.state_dict(),
                'temperature': TEMPERATURE,
                'epoch': epoch + 1,
                'val_loss': best_val_loss
            }, BEST_MODEL_SAVE_PATH)
        else:
            epochs_without_improvement += 1

        # 保存最新的检查点
        torch.save({
            'epoch': epoch + 1,
            'visual_encoder_state_dict': clip_model.visual.state_dict(),
            'image_projector_state_dict': image_projector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'temperature': TEMPERATURE, 
            'best_val_loss': best_val_loss,
            'epochs_without_improvement': epochs_without_improvement,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }, LATEST_CHECKPOINT_PATH)
        
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered after {PATIENCE} epochs.")
            break

    # 9. 绘图逻辑 (无删减)
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    if val_loss_history:
        best_epoch = np.argmin(val_loss_history)
        plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch+1})')
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_MODEL_DIR, 'loss_curve.png'))
    plt.close()
    print("Training finished. Loss curve saved.")

if __name__ == '__main__':
    main()