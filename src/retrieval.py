# eval_retrieval.py
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# --- 导入自定义模块 ---
from common_utils import (
    BASE_DIR, 
    adapt_vit_to_12_channels
)
from datasets import ECGSignalImageDataset, collate_fn

# 导入外部依赖模块
import clip_package as official_clip_module
from finetune_model import ft_12lead_ECGFounder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 数据路径
DATA_ROOT = os.path.join(BASE_DIR, "data", "心电数据集")
TEST_METADATA_FILE = os.path.join(DATA_ROOT, "mimic_metadata_patient_split", "metadata_test.csv")
STATS_FILE = os.path.join(DATA_ROOT, "signal_stats_vit.npz")

# 权重路径
MODELS_DIR = os.path.join(BASE_DIR, "models")
ECG_SIGNAL_MODEL_PATH = os.path.join(MODELS_DIR, "12_lead_ECGFounder.pth")
CLIP_MODEL_PATH = os.path.join(MODELS_DIR, "ViT-B-32.pt")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "results", "save_model", "ECG-Image-FM.pth") 

# 模型超参数 (需与训练时保持一致)
N_CLASSES_FOR_NET1D_INIT = 5
TARGET_SIGNAL_LENGTH = 1250
NUM_SIGNAL_CHANNELS = 12
SIGNAL_FEATURE_DIM = 1024
CLIP_IMAGE_FEATURE_DIM = 512
PROJECTION_DIM = 1024
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128 # 用于评估相似度矩阵时的批大小

def main():
    # 2. 加载信号统计值
    if not os.path.exists(STATS_FILE):
        raise FileNotFoundError(f"统计值文件 {STATS_FILE} 未找到。请先运行训练脚本生成它。")
    signal_stats = np.load(STATS_FILE)

    # 3. 初始化模型结构
    print("Initializing model structures for evaluation...")
    ecg_signal_model = ft_12lead_ECGFounder(DEVICE, ECG_SIGNAL_MODEL_PATH, n_classes=N_CLASSES_FOR_NET1D_INIT)
    
    clip_model, _ = official_clip_module.load(model_path=CLIP_MODEL_PATH, device="cpu", jit=False)
    clip_model = adapt_vit_to_12_channels(clip_model)
    clip_model.float().to(DEVICE)
    
    image_projector = nn.Linear(CLIP_IMAGE_FEATURE_DIM, PROJECTION_DIM).to(DEVICE)

    # 4. 加载最佳训练权重
    print(f"Loading best model weights from {BEST_MODEL_PATH}...")
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"未找到训练好的模型权重：{BEST_MODEL_PATH}")
        
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    clip_model.visual.load_state_dict(checkpoint['visual_encoder_state_dict'])
    image_projector.load_state_dict(checkpoint['image_projector_state_dict'])
    print(f"Model loaded from epoch {checkpoint['epoch']} (Val Loss: {checkpoint['val_loss']:.4f})")

    # 5. 定义评估用的预处理 (无增强版)
    grayscale_mean = (0.48145466 + 0.4578275 + 0.40821073) / 3
    grayscale_std = (0.26862954 + 0.26130258 + 0.27577711) / 3
    clip_image_preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(grayscale_mean,), std=(grayscale_std,))
    ])

    # 6. 准备测试数据
    test_dataset = ECGSignalImageDataset(TEST_METADATA_FILE, clip_image_preprocess, TARGET_SIGNAL_LENGTH, signal_stats)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
    print(f"Test dataset loaded with {len(test_dataset)} samples.")

    # 7. 特征提取
    print("Extracting features from test set...")
    ecg_signal_model.eval()
    clip_model.visual.eval()
    image_projector.eval()
    clip_dtype = clip_model.dtype

    all_signal_features, all_image_features = [], []
    with torch.no_grad():
        for batch_data in tqdm(test_dataloader, desc="Extracting"):
            if batch_data is None or batch_data[0] is None:
                continue
            ecg_signals, ecg_images = batch_data
            ecg_signals, ecg_images = ecg_signals.to(DEVICE), ecg_images.to(DEVICE)
            
            # 提取信号特征 [Batch, 1024]
            _, signal_features = ecg_signal_model(ecg_signals)
            
            # 提取图像特征并投影 [Batch, 1024]
            raw_image_features = clip_model.encode_image(ecg_images.to(clip_dtype)).float()
            image_features = image_projector(raw_image_features)
            
            # 归一化
            signal_features = F.normalize(signal_features, dim=-1)
            image_features = F.normalize(image_features, dim=-1)
            
            all_signal_features.append(signal_features)
            all_image_features.append(image_features)

    all_signal_features = torch.cat(all_signal_features)
    all_image_features = torch.cat(all_image_features)
    num_test_samples = len(all_signal_features)
    print(f"Successfully extracted features for {num_test_samples} test samples.")

    # 8. 跨模态检索评估 (分批计算防止显存溢出)
    print("\n--- Final Evaluation Metrics (Retrieval) ---")

    # --- Signal-to-Image (S2I) Retrieval ---
    s2i_top1, s2i_top5, s2i_top10 = 0, 0, 0
    with torch.no_grad():
        for i in tqdm(range(0, num_test_samples, EVAL_BATCH_SIZE), desc="S2I Eval"):
            signal_batch = all_signal_features[i : i + EVAL_BATCH_SIZE]
            similarity = signal_batch @ all_image_features.T # [Batch, N]
            
            labels = torch.arange(i, i + len(signal_batch)).to(DEVICE)
            
            # Top-1
            preds = similarity.argmax(dim=1)
            s2i_top1 += (preds == labels).sum().item()
            
            # Top-5
            _, top5 = torch.topk(similarity, k=5, dim=1)
            s2i_top5 += (top5 == labels.view(-1, 1)).any(dim=1).sum().item()
            
            # Top-10
            _, top10 = torch.topk(similarity, k=10, dim=1)
            s2i_top10 += (top10 == labels.view(-1, 1)).any(dim=1).sum().item()

    print(f"Signal-to-Image:")
    print(f"  - Top-1 Accuracy: {s2i_top1/num_test_samples*100:.2f}%")
    print(f"  - Top-5 Accuracy: {s2i_top5/num_test_samples*100:.2f}%")
    print(f"  - Top-10 Accuracy: {s2i_top10/num_test_samples*100:.2f}%")

    # --- Image-to-Signal (I2S) Retrieval ---
    i2s_top1, i2s_top5, i2s_top10 = 0, 0, 0
    with torch.no_grad():
        for i in tqdm(range(0, num_test_samples, EVAL_BATCH_SIZE), desc="I2S Eval"):
            image_batch = all_image_features[i : i + EVAL_BATCH_SIZE]
            similarity = image_batch @ all_signal_features.T # [Batch, N]
            
            labels = torch.arange(i, i + len(image_batch)).to(DEVICE)
            
            # Top-1
            preds = similarity.argmax(dim=1)
            i2s_top1 += (preds == labels).sum().item()
            
            # Top-5
            _, top5 = torch.topk(similarity, k=5, dim=1)
            i2s_top5 += (top5 == labels.view(-1, 1)).any(dim=1).sum().item()
            
            # Top-10
            _, top10 = torch.topk(similarity, k=10, dim=1)
            i2s_top10 += (top10 == labels.view(-1, 1)).any(dim=1).sum().item()

    print(f"Image-to-Signal:")
    print(f"  - Top-1 Accuracy: {i2s_top1/num_test_samples*100:.2f}%")
    print(f"  - Top-5 Accuracy: {i2s_top5/num_test_samples*100:.2f}%")
    print(f"  - Top-10 Accuracy: {i2s_top10/num_test_samples*100:.2f}%")

if __name__ == '__main__':
    main()