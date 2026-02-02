import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
# 从同级目录的 common_utils 中导入处理函数
from common_utils import process_ecg_signal

class ECGSignalImageDataset(Dataset):
    """
    用于基础模型 (Foundation Model) 对比学习训练的 Dataset。
    输出：(ECG信号张量, 12通道堆叠的ECG图像张量)
    """
    def __init__(self, metadata_file, clip_preprocess_fn, signal_target_length, signal_stats):
        self.metadata = pd.read_csv(metadata_file)
        self.clip_preprocess = clip_preprocess_fn
        self.signal_target_length = signal_target_length
        self.signal_stats = signal_stats
        print(f"Foundation Dataset initialized with {len(self.metadata)} samples.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 1. 处理信号 (调用公共工具函数)
        ecg_signal = process_ecg_signal(
            row['signal_path'], 
            stats=self.signal_stats, 
            target_length=self.signal_target_length
        )
        if ecg_signal is None:
            return None
            
        # 2. 处理图像：执行区域裁剪与 12 通道堆叠
        try:
            full_image = Image.open(row['image_path']).convert("RGB")
            num_rows, num_cols = 3, 4
            cell_width, cell_height = full_image.width // num_cols, full_image.height // num_rows
            
            lead_tensors = []
            # 遍历 3x4 的网格
            for col in range(num_cols):
                for r in range(num_rows):
                    left, upper = col * cell_width, r * cell_height
                    right, lower = left + cell_width, upper + cell_height
                    # 裁剪出单个导联区域
                    lead_crop = full_image.crop((left, upper, right, lower))
                    # 预处理 (Resize, Normalization 等)
                    lead_tensor = self.clip_preprocess(lead_crop)
                    lead_tensors.append(lead_tensor)
            
            # 在通道维度拼接，形成 [12, 224, 224] 的张量
            ecg_image_stack = torch.cat(lead_tensors, dim=0)
            return ecg_signal, ecg_image_stack
            
        except Exception as e:
            # 打印错误方便调试，实际运行时返回 None 会被 collate_fn 过滤
            # print(f"Error loading image {row['image_path']}: {e}")
            return None

class MultiLabelECGImageDataset(Dataset):
    """
    用于 PTB-XL 等下游分类任务实验的 Dataset。
    输出：(12通道堆叠的ECG图像张量, 多标签分类标签)
    """
    def __init__(self, dataframe, image_preprocess_fn, class_columns):
        self.df = dataframe
        self.image_preprocess = image_preprocess_fn
        self.class_columns = class_columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        
        # 获取多标签数值
        label_values = row[self.class_columns].values.astype(np.float32)
        labels = torch.from_numpy(label_values)
        
        # 处理图像：执行相同的区域裁剪与 12 通道堆叠逻辑
        try:
            full_image = Image.open(image_path).convert("RGB")
            num_rows, num_cols = 3, 4
            cell_width, cell_height = full_image.width // num_cols, full_image.height // num_rows
            
            lead_tensors = []
            for col in range(num_cols):
                for r in range(num_rows):
                    left, upper = col * cell_width, r * cell_height
                    right, lower = left + cell_width, upper + cell_height
                    lead_crop = full_image.crop((left, upper, right, lower))
                    lead_tensor = self.image_preprocess(lead_crop)
                    lead_tensors.append(lead_tensor)
            
            ecg_image_stack = torch.cat(lead_tensors, dim=0)
            return ecg_image_stack, labels
            
        except Exception:
            return None

def collate_fn(batch):
    """
    用于基础模型训练的整理函数，剔除读取失败(None)的样本
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    ecg_signals, ecg_images = zip(*batch)
    return torch.stack(ecg_signals), torch.stack(ecg_images)

def collate_fn_multilabel(batch):
    """
    用于多标签分类实验的整理函数，剔除读取失败(None)的样本
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)
class ECGImageDataset(Dataset):
    """
    用于单标签分类 (如 Abnormal, Normal 等) 的 Dataset。
    输出：(12通道堆叠的ECG图像张量, 单标签索引)
    """
    def __init__(self, dataframe, image_preprocess_fn, class_to_idx):
        self.df = dataframe
        self.image_preprocess = image_preprocess_fn
        self.class_to_idx = class_to_idx
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row.get('filepath') or row.get('image_path')
        label_name = row['label']
        label = torch.tensor(self.class_to_idx[label_name], dtype=torch.long)
        
        try:
            full_image = Image.open(image_path).convert("RGB")
            num_rows, num_cols = 3, 4
            cell_width, cell_height = full_image.width // num_cols, full_image.height // num_rows
            
            lead_tensors = []
            for col in range(num_cols):
                for r in range(num_rows):
                    left, upper = col * cell_width, r * cell_height
                    right, lower = left + cell_width, upper + cell_height
                    lead_crop = full_image.crop((left, upper, right, lower))
                    lead_tensor = self.image_preprocess(lead_crop)
                    lead_tensors.append(lead_tensor)
                    
            return torch.cat(lead_tensors, dim=0), label
        except Exception:
            return None