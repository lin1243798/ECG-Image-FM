# eval_efficiency_our.py
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
from sklearn.metrics import roc_auc_score
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
# 加载你之前训练好的基础模型权重
BEST_MODEL_PATH = os.path.join(BASE_DIR, "results", "save_model", "ECG-Image-FM.pth")
CLIP_MODEL_PATH = os.path.join(BASE_DIR, "models", "ViT-B-32.pt")
RESULTS_SAVE_DIR = os.path.join(BASE_DIR, "results", "ptbxl_efficiency_our_method")

# 模型参数
CLIP_IMAGE_FEATURE_DIM = 512
PROJECTION_DIM = 1024
NUM_CLASSES = 5 
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# 实验参数 (Linear Probe 通常使用较大的学习率)
BATCH_SIZE = 32
EPOCHS_PER_ROUND = 30
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5 

# 数据比例增量设置
TRAIN_FRACTIONS = np.arange(0.05, 1.05, 0.05) 

# --- 3. 辅助类定义 ---

class ECGImageClassifier(nn.Module):
    def __init__(self, image_encoder, image_projector, num_classes):
        super().__init__()
        self.image_encoder = image_encoder
        self.image_projector = image_projector
        
        # --- 关键逻辑：冻结特征提取器 (Linear Probe) ---
        for param in self.image_encoder.parameters(): param.requires_grad = False
        for param in self.image_projector.parameters(): param.requires_grad = False
        
        self.classification_head = nn.Linear(PROJECTION_DIM, num_classes)
        
    def forward(self, x):
        # 强制特征提取部分在 eval 模式运行 (不计算梯度/不更新 BN 或 Dropout)
        self.image_encoder.eval()
        self.image_projector.eval()
        
        with torch.no_grad():
            raw_features = self.image_encoder(x)
            projected_features = self.image_projector(raw_features.float())
            
        logits = self.classification_head(projected_features)
        return logits

    def reset_head(self):
        """每轮实验重置分类头"""
        self.classification_head = nn.Linear(PROJECTION_DIM, NUM_CLASSES).to(DEVICE)

def plot_single_metric(fractions, values, metric_name, color, save_path):
    plt.figure(figsize=(10, 8))
    plt.plot(fractions, values, marker='o', label=metric_name, linewidth=2, color=color)
    plt.title(f'{metric_name} vs Training Data Size (Our FM)', fontsize=14)
    plt.xlabel('Fraction of Training Data Used', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    min_val, max_val = min(values), max(values)
    margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
    plt.ylim([max(0, min_val - margin), min(1.0, max_val + margin)])
    
    for x, y in zip(fractions, values):
        if x == fractions[0] or x == fractions[-1] or (int(x*100) % 10 == 0):
            plt.text(x, y + 0.002, f'{y:.4f}', ha='center', va='bottom', fontsize=9)

    plt.savefig(save_path)
    plt.close()

# --- 4. 主执行逻辑 ---
def main():
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

    # 1. 加载并准备基础模型结构
    print("Loading pretrained foundation model components...")
    clip_model, _ = official_clip_module.load(model_path=CLIP_MODEL_PATH, device="cpu", jit=False)
    clip_model = adapt_vit_to_12_channels(clip_model)
    clip_dtype = clip_model.dtype
    image_projector = nn.Linear(CLIP_IMAGE_FEATURE_DIM, PROJECTION_DIM)
    
    # 加载你的对比学习训练后的权重
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"未找到预训练模型：{BEST_MODEL_PATH}")
        
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    clip_model.visual.load_state_dict(checkpoint['visual_encoder_state_dict'])
    image_projector.load_state_dict(checkpoint['image_projector_state_dict'])
    
    model = ECGImageClassifier(clip_model.visual, image_projector, NUM_CLASSES).to(DEVICE)
    
    # 2. 预处理设置
    grayscale_mean = (0.48145466 + 0.4578275 + 0.40821073) / 3
    grayscale_std = (0.26862954 + 0.26130258 + 0.27577711) / 3
    image_preprocess = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
        transforms.Normalize(mean=(grayscale_mean,), std=(grayscale_std,))
    ])

    # 3. 数据准备与切分
    df = pd.read_csv(LABELED_METADATA_PATH)
    for col in CLASSES: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df['label_sum'] = df[CLASSES].sum(axis=1)
    df_filtered = df[df['label_sum'] > 0].copy().drop(columns=['label_sum'])

    splitter_test = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
    dev_idx, test_idx = next(splitter_test.split(df_filtered, groups=df_filtered['patient_id']))
    df_dev, df_test = df_filtered.iloc[dev_idx], df_filtered.iloc[test_idx]

    splitter_val = GroupShuffleSplit(test_size=0.125, n_splits=1, random_state=42)
    train_pool_idx, val_idx = next(splitter_val.split(df_dev, groups=df_dev['patient_id']))
    df_train_pool, df_val = df_dev.iloc[train_pool_idx], df_dev.iloc[val_idx]

    val_loader = DataLoader(MultiLabelECGImageDataset(df_val, image_preprocess, CLASSES), 
                            batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_multilabel, num_workers=4)
    test_loader = DataLoader(MultiLabelECGImageDataset(df_test, image_preprocess, CLASSES), 
                             batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_multilabel, num_workers=4)

    results = {'fraction': [], 'num_samples': [], 'macro_auc': [], 'weighted_auc': []}
    pool_patient_ids = df_train_pool['patient_id'].unique()
    np.random.seed(42)
    np.random.shuffle(pool_patient_ids)

    # 4. 增量实验
    print("\n" + "="*50)
    print(f"开始基础模型数据效率实验 (Linear Probe)")
    print("="*50)

    for frac in TRAIN_FRACTIONS:
        n_patients = int(len(pool_patient_ids) * frac)
        if n_patients == 0: continue 

        df_curr_train = df_train_pool[df_train_pool['patient_id'].isin(pool_patient_ids[:n_patients])]
        print(f"\n>>> 比例: {frac*100:.1f}% | 样本: {len(df_curr_train)}")

        train_loader = DataLoader(MultiLabelECGImageDataset(df_curr_train, image_preprocess, CLASSES), 
                                  batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_multilabel, num_workers=4)

        # 重置分类头并设置优化器
        model.reset_head()
        optimizer = optim.Adam(model.classification_head.parameters(), lr=LEARNING_RATE)
        pos_counts = df_curr_train[CLASSES].sum()
        pos_weight = torch.tensor((len(df_curr_train) - pos_counts) / (pos_counts + 1e-6)).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss, epochs_no_improve, best_head_state = float('inf'), 0, None

        for epoch in range(EPOCHS_PER_ROUND):
            model.train()
            for images, labels in train_loader:
                if images is None: continue
                images, labels = images.to(DEVICE).to(clip_dtype), labels.to(DEVICE)
                optimizer.zero_grad(); loss = criterion(model(images), labels); loss.backward(); optimizer.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    if images is None: continue
                    val_loss += criterion(model(images.to(DEVICE).to(clip_dtype)), labels.to(DEVICE)).item()
            
            avg_val_loss = val_loss / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss, epochs_no_improve = avg_val_loss, 0
                best_head_state = model.classification_head.state_dict()
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE: break
        
        if best_head_state: model.classification_head.load_state_dict(best_head_state)

        # 评估
        model.eval(); all_labels, all_probs = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                if images is None: continue
                probs = torch.sigmoid(model(images.to(DEVICE).to(clip_dtype)))
                all_labels.extend(labels.numpy()); all_probs.extend(probs.cpu().numpy())
        
        all_labels, all_probs = np.array(all_labels), np.array(all_probs)
        macro_auc = roc_auc_score(all_labels, all_probs, average='macro')
        weighted_auc = roc_auc_score(all_labels, all_probs, average='weighted')

        results['fraction'].append(frac); results['num_samples'].append(len(df_curr_train))
        results['macro_auc'].append(macro_auc); results['weighted_auc'].append(weighted_auc)
        print(f"Result: Macro AUC={macro_auc:.4f}")

    # 5. 保存结果与绘图
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_SAVE_DIR, 'our_results.csv'), index=False)
    plot_single_metric(results['fraction'], results['macro_auc'], "Macro AUC", "blue", os.path.join(RESULTS_SAVE_DIR, 'our_macro_auc.png'))
    print(f"实验报告已生成至: {RESULTS_SAVE_DIR}")

if __name__ == '__main__':
    main()
