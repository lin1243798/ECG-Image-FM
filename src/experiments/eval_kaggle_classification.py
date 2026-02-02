# eval_custom_classification.py
import os
import sys

# --- 核心修改：确保能导入父目录的 src 模块 ---
CUR_DIR = os.path.dirname(os.path.abspath(__file__)) # src/experiments
SRC_DIR = os.path.dirname(CUR_DIR)                  # src
sys.path.append(SRC_DIR)

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt

# 导入自定义模块
from common_utils import BASE_DIR, adapt_vit_to_12_channels
from datasets import ECGImageDataset, collate_fn
import clip_package as official_clip_module

# --- 1. 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 路径配置 (相对路径化)
CUSTOM_CSV_PATH = os.path.join(BASE_DIR, 'data', 'ecg_dataset.csv')
FEATURE_EXTRACTOR_PATH = os.path.join(BASE_DIR, "results", "save_model", "ECG-Image-FM.pth")
CLIP_MODEL_PATH = os.path.join(BASE_DIR, "models", "ViT-B-32.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "custom_classification_auc_only")

# 模型参数
CLIP_IMAGE_FEATURE_DIM = 512
PROJECTION_DIM = 1024
CLASSES = ['Abnormal', 'History_MI', 'MI', 'Normal']
NUM_CLASSES = len(CLASSES) 

# --- 核心修改：固定训练参数 (移除网格搜索) ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-2
EPOCHS = 50 
EARLY_STOPPING_PATIENCE = 5 
TEST_SPLIT_SIZE = 0.20
VALIDATION_SPLIT_SIZE = 0.125

# --- 2. 分类器定义 ---

class ECGImageClassifier(nn.Module):
    def __init__(self, image_encoder, image_projector, num_classes):
        super().__init__()
        self.image_encoder = image_encoder
        self.image_projector = image_projector
        # 冻结特征提取器 (Linear Probe)
        for param in self.image_encoder.parameters(): param.requires_grad = False
        for param in self.image_projector.parameters(): param.requires_grad = False
        self.classification_head = nn.Linear(PROJECTION_DIM, num_classes)
        
    def forward(self, x):
        self.image_encoder.eval()
        self.image_projector.eval()
        with torch.no_grad():
            raw_features = self.image_encoder(x)
            projected_features = self.image_projector(raw_features.float())
        logits = self.classification_head(projected_features)
        return logits

# --- 3. 绘图函数---

def plot_combined_roc_curves(roc_data, save_path):
    plt.figure(figsize=(10, 8))
    for c, d in roc_data.items():
        plt.plot(d['fpr'], d['tpr'], lw=2, label=f"{c} (AUC = {d['auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Custom Dataset Classification')
    plt.legend(loc="lower right"); plt.grid(True)
    plt.savefig(save_path); plt.close()

# --- 4. 主执行逻辑 ---

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4.1 数据加载与划分
    df = pd.read_csv(CUSTOM_CSV_PATH)
    if 'image_path' in df.columns and 'filepath' not in df.columns:
        df.rename(columns={'image_path': 'filepath'}, inplace=True)
    class_to_idx = {name: i for i, name in enumerate(CLASSES)}
    
    print(f"Dividing dataset (Stratified) - Train/Val/Test...")
    train_val_df, test_df = train_test_split(df, test_size=TEST_SPLIT_SIZE, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_val_df, test_size=VALIDATION_SPLIT_SIZE, random_state=42, stratify=train_val_df['label'])
    
    # 4.2 模型加载
    print("Loading foundation model...")
    clip_model, _ = official_clip_module.load(model_path=CLIP_MODEL_PATH, device="cpu", jit=False)
    clip_model = adapt_vit_to_12_channels(clip_model)
    image_projector = nn.Linear(CLIP_IMAGE_FEATURE_DIM, PROJECTION_DIM)
    
    checkpoint = torch.load(FEATURE_EXTRACTOR_PATH, map_location=DEVICE)
    clip_model.visual.load_state_dict(checkpoint['visual_encoder_state_dict'])
    image_projector.load_state_dict(checkpoint['image_projector_state_dict'])
    
    model = ECGImageClassifier(clip_model.visual, image_projector, NUM_CLASSES).to(DEVICE)
    clip_dtype = clip_model.dtype

    # 4.3 预处理与加载器
    preprocess = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466,), std=(0.26862954,))
    ])

    train_loader = DataLoader(ECGImageDataset(train_df, preprocess, class_to_idx), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(ECGImageDataset(val_df, preprocess, class_to_idx), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(ECGImageDataset(test_df, preprocess, class_to_idx), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # 4.4 训练 (Linear Probe)
    print(f"\nStarting training [LR={LEARNING_RATE}]...")
    optimizer = optim.Adam(model.classification_head.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_loss, epochs_no_improve, best_state = float('inf'), 0, None

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
            if images is None: continue
            images, labels = images.to(DEVICE).to(clip_dtype), labels.to(DEVICE)
            optimizer.zero_grad(); loss = criterion(model(images), labels); loss.backward(); optimizer.step()
        
        model.eval(); val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if images is None: continue
                val_loss += criterion(model(images.to(DEVICE).to(clip_dtype)), labels.to(DEVICE)).item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss, epochs_no_improve, best_state = avg_val_loss, 0, model.state_dict()
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE: break
    
    if best_state: model.load_state_dict(best_state)

    # 4.5 测试集评估
    print("\nEvaluating AUC on test set...")
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final Testing"):
            if images is None: continue
            images = images.to(DEVICE).to(clip_dtype)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    
    # 二值化标签用于计算 OVR AUC
    y_true_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))

    # 4.6 保存结果与绘图
    report_path = os.path.join(OUTPUT_DIR, 'auc_report.txt')
    roc_plot_data = {}
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Custom Dataset Classification AUC Report\n" + "="*40 + "\n")
        for i, class_name in enumerate(CLASSES):
            # 计算 One-vs-Rest AUC
            auc_val = roc_auc_score(y_true_bin[:, i], all_probs[:, i])
            f.write(f"Class {class_name:<12}: AUC = {auc_val:.4f}\n")
            
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
            roc_plot_data[class_name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_val}
        
        macro_auc = roc_auc_score(y_true_bin, all_probs, average='macro', multi_class='ovr')
        f.write("-" * 40 + f"\nMacro Average AUC: {macro_auc:.4f}\n")
        print(f"\nFinal Macro AUC: {macro_auc:.4f}")

    plot_combined_roc_curves(roc_plot_data, os.path.join(OUTPUT_DIR, 'roc_curves_final.png'))
    print(f"评估完成。AUC 报告保存至: {report_path}")

if __name__ == '__main__':
    main()