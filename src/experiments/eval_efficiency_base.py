# eval_efficiency_base.py
import os
import sys

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
import gc

# 导入自定义模块
from common_utils import BASE_DIR
from datasets import MultiLabelECGImageDataset, collate_fn_multilabel
import clip_package as official_clip_module

# --- 2. 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 路径配置 
LABELED_METADATA_PATH = os.path.join(BASE_DIR, 'data', 'ptb-xl', 'ptbxl_full_manifest.csv')
CLIP_MODEL_PATH = os.path.join(BASE_DIR, "models", "ViT-B-32.pt") 
RESULTS_SAVE_DIR = os.path.join(BASE_DIR, "results", "ptbxl_efficiency_base_vit32b")

# 模型与训练参数
CLIP_IMAGE_FEATURE_DIM = 512
PROJECTION_DIM = 1024
NUM_CLASSES = 5 
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

BATCH_SIZE = 32 
EPOCHS_PER_ROUND = 30  
LEARNING_RATE = 1e-5   
EARLY_STOPPING_PATIENCE = 5 
WEIGHT_DECAY = 0.05

# 数据比例增量设置
TRAIN_FRACTIONS = np.arange(0.05, 1.05, 0.05) 

# --- 3. 辅助函数与分类器定义 ---

def get_randomly_initialized_model():
    """创建一个全新的、随机初始化的12通道ViT模型"""
    # 1. 加载结构 (不加载权重)
    model, _ = official_clip_module.load(model_path=CLIP_MODEL_PATH, device="cpu", jit=False)
    
    # 2. 随机初始化所有参数
    model.visual.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    
    # 3. 适配 12 通道 (也是随机初始化)
    original_conv = model.visual.conv1
    new_conv = nn.Conv2d(
        in_channels=12, 
        out_channels=original_conv.out_channels, 
        kernel_size=original_conv.kernel_size, 
        stride=original_conv.stride, 
        padding=original_conv.padding,
        bias=(original_conv.bias is not None)
    )
    model.visual.conv1 = new_conv
    return model

class ECGImageClassifier(nn.Module):
    def __init__(self, image_encoder, image_projector, num_classes):
        super().__init__()
        self.image_encoder = image_encoder
        self.image_projector = image_projector
        
        # 全参数训练
        for param in self.image_encoder.parameters(): param.requires_grad = True
        for param in self.image_projector.parameters(): param.requires_grad = True
            
        self.classification_head = nn.Linear(PROJECTION_DIM, num_classes)
        
    def forward(self, x):
        raw_features = self.image_encoder(x)
        projected_features = self.image_projector(raw_features.float())
        logits = self.classification_head(projected_features)
        return logits

def plot_single_metric(fractions, values, metric_name, color, save_path):
    plt.figure(figsize=(10, 8))
    plt.plot(fractions, values, marker='o', label=f'{metric_name} (Base)', linewidth=2, color=color)
    plt.title(f'{metric_name} vs Data Size (Scratch)', fontsize=14)
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

    # 预处理
    grayscale_mean = (0.48145466 + 0.4578275 + 0.40821073) / 3
    grayscale_std = (0.26862954 + 0.26130258 + 0.27577711) / 3
    image_preprocess = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
        transforms.Normalize(mean=(grayscale_mean,), std=(grayscale_std,))
    ])

    # 加载数据
    df = pd.read_csv(LABELED_METADATA_PATH)
    for col in CLASSES: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df['label_sum'] = df[CLASSES].sum(axis=1)
    df_filtered = df[df['label_sum'] > 0].copy().drop(columns=['label_sum'])

    # 数据划分
    splitter_test = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
    dev_idx, test_idx = next(splitter_test.split(df_filtered, groups=df_filtered['patient_id']))
    df_dev, df_test = df_filtered.iloc[dev_idx], df_filtered.iloc[test_idx]

    splitter_val = GroupShuffleSplit(test_size=0.125, n_splits=1, random_state=42) # 0.1/0.8
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

    for frac in TRAIN_FRACTIONS:
        # 清理显存
        if 'model' in locals(): del model; torch.cuda.empty_cache(); gc.collect()

        print(f"\n[Fraction: {frac*100:.1f}%] Initializing FRESH model...")
        clip_model_raw = get_randomly_initialized_model()
        image_projector = nn.Linear(CLIP_IMAGE_FEATURE_DIM, PROJECTION_DIM)
        model = ECGImageClassifier(clip_model_raw.visual, image_projector, NUM_CLASSES).to(DEVICE)

        n_patients = int(len(pool_patient_ids) * frac)
        df_curr_train = df_train_pool[df_train_pool['patient_id'].isin(pool_patient_ids[:n_patients])]
        train_loader = DataLoader(MultiLabelECGImageDataset(df_curr_train, image_preprocess, CLASSES), 
                                  batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_multilabel, num_workers=4)

        # 优化器
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        pos_counts = df_curr_train[CLASSES].sum()
        pos_weight = torch.tensor((len(df_curr_train) - pos_counts) / (pos_counts + 1e-6)).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        best_val_loss, epochs_no_improve, best_model_state = float('inf'), 0, None

        # 训练循环
        for epoch in range(EPOCHS_PER_ROUND):
            model.train()
            for images, labels in train_loader:
                if images is None: continue
                images, labels = images.to(DEVICE).float(), labels.to(DEVICE)
                optimizer.zero_grad(); loss = criterion(model(images), labels); loss.backward(); optimizer.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    if images is None: continue
                    val_loss += criterion(model(images.to(DEVICE).float()), labels.to(DEVICE)).item()
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE: break
        
        if best_model_state: model.load_state_dict(best_model_state)

        # 评估
        model.eval(); all_labels, all_probs = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                if images is None: continue
                probs = torch.sigmoid(model(images.to(DEVICE).float()))
                all_labels.extend(labels.numpy()); all_probs.extend(probs.cpu().numpy())
        
        all_labels, all_probs = np.array(all_labels), np.array(all_probs)
        macro_auc = roc_auc_score(all_labels, all_probs, average='macro')
        weighted_auc = roc_auc_score(all_labels, all_probs, average='weighted')

        results['fraction'].append(frac); results['num_samples'].append(len(df_curr_train))
        results['macro_auc'].append(macro_auc); results['weighted_auc'].append(weighted_auc)
        print(f"Result: Macro AUC={macro_auc:.4f}")

    # 保存报告与图表
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_SAVE_DIR, 'results.csv'), index=False)
    plot_single_metric(results['fraction'], results['macro_auc'], "Macro AUC", "red", os.path.join(RESULTS_SAVE_DIR, 'macro_auc.png'))
    print("Supervised experiment complete.")

if __name__ == '__main__':
    main()