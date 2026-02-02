# eval_ptbxl_classification.py
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt

# 导入自定义模块
from common_utils import BASE_DIR, adapt_vit_to_12_channels
from datasets import MultiLabelECGImageDataset, collate_fn_multilabel
import clip_package as official_clip_module

# --- 2. 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 路径配置
LABELED_METADATA_PATH = os.path.join(BASE_DIR, 'data', 'ptb-xl', 'ptbxl_full_manifest.csv')
BEST_MODEL_PATH = os.path.join(BASE_DIR, "results", "save_model", "ECG-Image-FM.pth")
CLIP_MODEL_PATH = os.path.join(BASE_DIR, "models", "ViT-B-32.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "ptbxl_classification_auc")

# 模型参数
CLIP_IMAGE_FEATURE_DIM = 512
PROJECTION_DIM = 1024
NUM_CLASSES = 5 
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# 固定训练参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5 

# --- 3. 模型定义 ---

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

# --- 4. 绘图函数  ---

def plot_combined_roc_curves(roc_data, save_path):
    plt.figure(figsize=(10, 8))
    for class_name, data in roc_data.items():
        plt.plot(data['fpr'], data['tpr'], lw=2,
                 label=f"{class_name} (AUC = {data['auc']:.4f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - PTB-XL Classification')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    print(f"ROC曲线已保存至: {save_path}")
    plt.close()

# --- 5. 主执行逻辑 ---

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5.1 加载模型
    print("Loading foundation model...")
    clip_model, _ = official_clip_module.load(model_path=CLIP_MODEL_PATH, device="cpu", jit=False)
    clip_model = adapt_vit_to_12_channels(clip_model)
    image_projector = nn.Linear(CLIP_IMAGE_FEATURE_DIM, PROJECTION_DIM)
    
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    clip_model.visual.load_state_dict(checkpoint['visual_encoder_state_dict'])
    image_projector.load_state_dict(checkpoint['image_projector_state_dict'])
    
    model = ECGImageClassifier(clip_model.visual, image_projector, NUM_CLASSES).to(DEVICE)
    clip_dtype = clip_model.dtype

    # 5.2 数据准备
    df = pd.read_csv(LABELED_METADATA_PATH)
    for col in CLASSES: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df['label_sum'] = df[CLASSES].sum(axis=1)
    df_filtered = df[df['label_sum'] > 0].copy().drop(columns=['label_sum'])

    # 划分 (70/10/20)
    splitter_test = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_val_idx, test_idx = next(splitter_test.split(df_filtered, groups=df_filtered['patient_id']))
    train_val_data, test_df = df_filtered.iloc[train_val_idx], df_filtered.iloc[test_idx]

    splitter_val = GroupShuffleSplit(test_size=0.125, n_splits=1, random_state=42)
    train_idx, val_idx = next(splitter_val.split(train_val_data, groups=train_val_data['patient_id']))
    train_df, val_df = train_val_data.iloc[train_idx], train_val_data.iloc[val_idx]

    # 预处理
    grayscale_mean = (0.48145466 + 0.4578275 + 0.40821073) / 3
    grayscale_std = (0.26862954 + 0.26130258 + 0.27577711) / 3
    preprocess = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
        transforms.Normalize(mean=(grayscale_mean,), std=(grayscale_std,))
    ])

    train_loader = DataLoader(MultiLabelECGImageDataset(train_df, preprocess, CLASSES), 
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_multilabel, num_workers=4)
    val_loader = DataLoader(MultiLabelECGImageDataset(val_df, preprocess, CLASSES), 
                            batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_multilabel, num_workers=4)
    test_loader = DataLoader(MultiLabelECGImageDataset(test_df, preprocess, CLASSES), 
                             batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_multilabel, num_workers=4)

    # 5.3 训练
    print(f"\nStarting training [Linear Probe] LR={LEARNING_RATE}...")
    pos_counts = train_df[CLASSES].sum()
    pos_weight = torch.tensor((len(train_df) - pos_counts) / (pos_counts + 1e-6)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.classification_head.parameters(), lr=LEARNING_RATE)

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

    # 5.4 最终评估 
    print("\nEvaluating AUC on test set...")
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            if images is None: continue
            images = images.to(DEVICE).to(clip_dtype)
            all_labels.extend(labels.numpy())
            all_probs.extend(torch.sigmoid(model(images)).cpu().numpy())
    
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)

    # 保存 AUC 报告
    report_path = os.path.join(OUTPUT_DIR, 'auc_report.txt')
    roc_plot_data = {}
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("PTB-XL Classification AUC Report\n" + "="*40 + "\n")
        
        # 计算每一类的 AUC
        for i, name in enumerate(CLASSES):
            y_true, y_prob = all_labels[:, i], all_probs[:, i]
            auc_val = roc_auc_score(y_true, y_prob)
            f.write(f"Class {name:<10}: AUC = {auc_val:.4f}\n")
            
            # 准备绘图数据
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_plot_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_val}
        
        # 计算 Macro AUC
        macro_auc = roc_auc_score(all_labels, all_probs, average='macro')
        f.write("-" * 40 + f"\nMacro Average AUC: {macro_auc:.4f}\n")
        print(f"\nFinal Macro AUC: {macro_auc:.4f}")

    # 5.5 绘制 ROC 曲线
    plot_combined_roc_curves(roc_plot_data, os.path.join(OUTPUT_DIR, 'roc_curves_final.png'))
    print(f"评估完成。AUC 报告已保存至: {report_path}")

if __name__ == '__main__':
    main()