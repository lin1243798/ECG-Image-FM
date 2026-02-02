import os
import torch
import torch.nn as nn
import numpy as np
import wfdb
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import torch.nn.functional as F

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CUR_DIR)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """带通滤波器逻辑"""
    nyquist_freq = 0.5 * fs
    low, high = lowcut / nyquist_freq, highcut / nyquist_freq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def process_ecg_signal(record_base_path, stats=None, target_length=1250, num_target_channels=12):
    """处理并归一化ECG信号逻辑"""
    try:
        record = wfdb.rdrecord(record_base_path)
        signal_data_raw, fs = record.p_signal, record.fs
    except Exception:
        return None
        
    if signal_data_raw is None or signal_data_raw.shape[1] < num_target_channels:
        return None
        
    lowcut, highcut = 0.5, 45.0
    filtered_signal = butter_bandpass_filter(signal_data_raw, lowcut, highcut, fs)
    
    # 转置为 [Channels, Length]
    signal_data = filtered_signal[:, :num_target_channels].T
    
    current_length = signal_data.shape[1]
    if current_length < target_length:
        padding = np.zeros((num_target_channels, target_length - current_length))
        processed_signal = np.concatenate((signal_data, padding), axis=1)
    else:
        processed_signal = signal_data[:, :target_length]
        
    if stats is not None:
        # 归一化逻辑，使用 stats 文件中的均值和标准差
        mean, std = stats['mean'].reshape(-1, 1), stats['std'].reshape(-1, 1)
        processed_signal = (processed_signal - mean) / (std + 1e-8)
        
    return torch.tensor(processed_signal.copy(), dtype=torch.float)

def get_signal_stats(stats_file_path, train_metadata_path):
    """计算或加载信号统计信息的逻辑"""
    if os.path.exists(stats_file_path):
        stats = np.load(stats_file_path)
        return {'mean': stats['mean'], 'std': stats['std']}
    else:
        df = pd.read_csv(train_metadata_path)
        all_signals = []
        for path in tqdm(df['signal_path'], desc="计算信号统计值"):
            signal = process_ecg_signal(path) 
            if signal is not None:
                all_signals.append(signal.numpy())
        
        stacked_signals = np.stack(all_signals)
        mean_per_channel = np.mean(stacked_signals, axis=(0, 2))
        std_per_channel = np.std(stacked_signals, axis=(0, 2))
        np.savez(stats_file_path, mean=mean_per_channel, std=std_per_channel)
        return {'mean': mean_per_channel, 'std': std_per_channel}

def adapt_vit_to_12_channels(clip_model):
    """修改 CLIP ViT 第一层卷积以支持 12 通道输入，"""
    print("Adapting CLIP ViT for 12-channel input...")
    original_conv = clip_model.visual.conv1
    original_weights = original_conv.weight.data
    original_bias = original_conv.bias.data if original_conv.bias is not None else None
    
    # 权重扩展：从 3 通道扩展到 12 通道
    avg_weights = original_weights.mean(dim=1, keepdim=True)
    new_weights = avg_weights.repeat(1, 12, 1, 1)
    
    new_conv = nn.Conv2d(
        12, original_conv.out_channels, 
        kernel_size=original_conv.kernel_size, 
        stride=original_conv.stride, 
        padding=original_conv.padding, 
        bias=(original_bias is not None)
    )
    new_conv.weight.data = new_weights
    if original_bias is not None:
        new_conv.bias.data = original_bias
        
    clip_model.visual.conv1 = new_conv
    print("CLIP ViT successfully adapted for 12 channels.")
    return clip_model

def contrastive_loss(signal_features, image_features, temperature):
    signal_features = F.normalize(signal_features, dim=-1)
    image_features = F.normalize(image_features, dim=-1)
    
    # 计算相似度矩阵
    logits = torch.matmul(signal_features, image_features.T) * temperature.exp()
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    loss_signal = F.cross_entropy(logits, labels)
    loss_image = F.cross_entropy(logits.T, labels)
    
    return (loss_signal + loss_image) / 2.0
