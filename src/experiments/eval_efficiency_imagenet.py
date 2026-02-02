# eval_efficiency_imagenet.py
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
import timm
import gc 

# 导入自定义模块
from common_utils import BASE_DIR
from datasets import MultiLabelECGImageDataset, collate_fn_multilabel

# --- 2. 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 路径配置 
LABELED_METADATA_PATH = os.path.join(BASE_DIR, 'data', 'ptb-xl', 'ptbxl_full_manifest.csv')
# 本地 ImageNet 权重路径 (需提前下载并放入 models 目录)
LOCAL_CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "pytorch_model.bin")
RESULTS_SAVE_DIR = os.path.join(BASE_DIR, "results", "ptbxl_efficiency_imagenet")

# 模型参数
VIT_FEATURE_DIM = 768 # ViT-Base 标准输出维度
PROJECTION_DIM = 1024
NUM_CLASSES = 5 
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# 训练参数
BATCH_SIZE = 32
EPOCHS_PER_ROUND = 30
LEARNING_RATE = 1e-5 
EARLY_STOPPING_PATIENCE = 5 
WEIGHT_DECAY = 0.01

# 数据比例增量设置
TRAIN_FRACTIONS = np.arange(0.05, 1.05, 0.05) 

# --- 3. 辅助函数与模型定义 ---

def get_imagenet_pretrained_model():
    """
    1. 创建 TIMM ViT 模型
    2. 加载 ImageNet 预训练权重
    3. 修改第一层以适配 12 通道 (取均值后扩展)
    """
    # A. 创建模型骨架 (ViT-Base)
    image_encoder = timm.create_model(
        'vit_base_patch16_224.augreg_in21k', 
        pretrained=False, 
        num_classes=0
    )

    # B. 加载权重
    try:
        if not os.path.exists(LOCAL_CHECKPOINT_PATH):
            print(f"警告: 未在 {LOCAL_CHECKPOINT_PATH} 找到权重文件！")
            return None
            
        state_dict = torch.load(LOCAL_CHECKPOINT_PATH, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
            
        image_encoder.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None

    # C. 适配 12 通道 (Channel Adaptation)
    # 取3通道权重的平均值，重复12次
    original_conv = image_encoder.patch_embed.proj
    new_conv = nn.Conv2d(
        in_channels=12, 
        out_channels=original_conv.out_channels, 
        kernel_size=original_conv.kernel_size, 
        stride=original_conv.stride, 
        padding=original_conv.padding
    )
    
    with torch.no_grad():
        avg_weights = original_conv.weight.data.mean(dim=1, keepdim=True).repeat(1, 12, 1, 1)
        new_conv.weight.data = avg_weights
        if original_conv.bias is not None:
            new_conv.bias.data = original_conv.bias.data

    image_encoder.patch_embed.proj = new_conv
    return image_encoder

class ECGImageClassifier(nn.Module):
    def __init__(self, image_encoder, image_projector, num_classes):
        super().__init__()
        self.image_encoder = image_encoder
        self.image_projector = image_projector
        
        # 全参数微调
        for param in self.image_encoder.parameters(): param.requires_grad = True
        for param in self.image_projector.parameters(): param.requires_grad = True
            
        self.classification_head = nn.Linear(PROJECTION_DIM, num_classes)
        
    def forward(self, x):
        raw_features = self.image_encoder(x)
        # timm 的 vit 输出通常是 [B, 768]
        projected_features = self.image_projector(raw_features.float())
        logits = self.classification_head(projected_features)
        return logits

def plot_single_metric(fractions, values, metric_name, color, save_path):
    plt.figure(figsize=(10, 8))
    plt.plot(fractions, values, marker='o', label=f'{metric_name} (ImageNet FT)', linewidth=2, color=color)
    plt.title(f'{metric_name} vs Data Size (Fine-tuning ImageNet)', fontsize=14)
    plt.xlabel('Fraction of Training Data Used', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
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

    for frac in TRAIN_FRACTIONS:
        # 清理显存
        if 'model' in locals(): del model; torch.cuda.empty_cache(); gc.collect()

        print(f"\n[Fraction: {frac*100:.1f}%] Loading ImageNet Pretrained Model...")
        vit_encoder = get_imagenet_pretrained_model()
        if vit_encoder is None: break
            
        model_dtype = next(vit_encoder.parameters()).dtype
        image_projector = nn.Linear(VIT_FEATURE_DIM, PROJECTION_DIM)
        model = ECGImageClassifier(vit_encoder, image_projector, NUM_CLASSES).to(DEVICE)

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
                images, labels = images.to(DEVICE).to(model_dtype), labels.to(DEVICE)
                optimizer.zero_grad(); loss = criterion(model(images), labels); loss.backward(); optimizer.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    if images is None: continue
                    val_loss += criterion(model(images.to(DEVICE).to(model_dtype)), labels.to(DEVICE)).item()
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve, best_model_state = 0, model.state_dict()
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE: break
        
        if best_model_state: model.load_state_dict(best_model_state)

        # 评估
        model.eval(); all_labels, all_probs = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                if images is None: continue
                probs = torch.sigmoid(model(images.to(DEVICE).to(model_dtype)))
                all_labels.extend(labels.numpy()); all_probs.extend(probs.cpu().numpy())
        
        all_labels, all_probs = np.array(all_labels), np.array(all_probs)
        macro_auc = roc_auc_score(all_labels, all_probs, average='macro')
        weighted_auc = roc_auc_score(all_labels, all_probs, average='weighted')

        results['fraction'].append(frac); results['num_samples'].append(len(df_curr_train))
        results['macro_auc'].append(macro_auc); results['weighted_auc'].append(weighted_auc)
        print(f"Result: Macro AUC={macro_auc:.4f}")

    # 保存报告
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_SAVE_DIR, 'imagenet_results.csv'), index=False)
    plot_single_metric(results['fraction'], results['macro_auc'], "Macro AUC", "purple", os.path.join(RESULTS_SAVE_DIR, 'imagenet_macro_auc.png'))
    print("ImageNet Transfer experiment complete.")

if __name__ == '__main__':
    main()