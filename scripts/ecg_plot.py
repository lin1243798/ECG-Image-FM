# plot_ptbdb.py

import wfdb
import numpy as np
import os
import sys
import glob
from scipy.signal import butter, filtfilt
# --- 1. 导入 ecg_plot 模块 ---
from ecg_plot import plot, save_as_png

# --- 新增：基线漂移滤波函数 ---
def remove_baseline_wander(signal, fs):
    """
    使用高通巴特沃斯滤波器去除基线漂移。
    
    Args:
    - signal: ECG信号数组，形状可以是 (N,) 或 (num_leads, N)。
    - fs: 信号的采样率。
    
    Returns:
    - 经过滤波的信号。
    """
    # 设置滤波器参数
    cutoff_freq = 0.5  # Hz, 这是去除基线漂移的标准截止频率
    filter_order = 4   # 滤波器阶数
    
    # 设计巴特沃斯高通滤波器
    b, a = butter(filter_order, cutoff_freq / (fs / 2.0), btype='high')
    
    # 应用滤波器 (使用 filtfilt 来避免相位失真)
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal
# --- 2. 单个记录的处理函数 (核心修改) ---
def process_ptbdb_record(record_path_base, save_directory):
    """
    处理单个PTBDB记录，并以 "patientXXX_sXXXX_re_2.5sec.png" 格式保存。
    """
    # --- 1. 提取文件名和病人文件夹名 ---
    record_name = os.path.basename(record_path_base)
    patient_folder_name = os.path.basename(os.path.dirname(record_path_base))
    # ------------------------------------

    try:
        record = wfdb.rdrecord(record_path_base)
        ecg_data = record.p_signal.T
        fs = record.fs
        lead_names = record.sig_name
        # --- *** 核心优化：在这里应用基线漂移滤波器 *** ---
        ecg_data = remove_baseline_wander(ecg_data, fs)
        if record.units[0].lower() != 'mv':
             print(f"警告: 记录 {record_name} 的单位是 {record.units[0]}，不是 mV。")

    except Exception as e:
        print(f"读取记录 {record_name} 失败: {e}")
        return

    # --- 2. 准备数据 (这部分逻辑不变) ---
    plot_duration = 2.5
    total_plot_samples = int(plot_duration * fs)
    standard_leads_12 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    ecg_data_for_plot_list = []
    lead_data_dict = {name.lower(): ecg_data[i] for i, name in enumerate(lead_names)}
    
    for lead in standard_leads_12:
        lead_lower = lead.lower()
        if lead_lower in lead_data_dict:
            lead_waveform = lead_data_dict[lead_lower]
            segment_to_plot = lead_waveform[:total_plot_samples]
            if len(segment_to_plot) < total_plot_samples:
                padding = np.zeros(total_plot_samples - len(segment_to_plot))
                segment_to_plot = np.concatenate([segment_to_plot, padding])
            ecg_data_for_plot_list.append(segment_to_plot)
        else:
            print(f"警告: 记录 {record_name} 中缺少导联 {lead}。")
            ecg_data_for_plot_list.append(np.zeros(total_plot_samples))

    ecg_data_for_plot = np.array(ecg_data_for_plot_list)
    
    # --- 3. 绘图 (这部分逻辑不变) ---
    final_row_height = 4.5
    
    plot(
        ecg_data_for_plot, 
        sample_rate=fs,
        # 更新标题，使其也包含病人信息
        title=f'ECG - {patient_folder_name}/{record_name} - 2.5s',
        columns=4,
        show_separate_line=False,
        row_height=final_row_height
    )
    
    # --- 4. 构建新的文件名并保存 ---
    # 使用 f-string 拼接成 "patient001_s0010_re_2.5sec" 格式
    output_filename_base = f"{patient_folder_name}_{record_name}_2.5sec"
    
    # 将不带后缀的文件名传给函数
    save_as_png(output_filename_base, path=save_directory)
    
    # 更新打印信息以反映正确的文件名
    full_save_path = os.path.join(save_directory, f"{output_filename_base}.png")
    # --------------------------------
    
    print(f"ECG图已保存为: {full_save_path}")

# --- 3. 主程序入口 ---
if __name__ == "__main__":
    # 定义输入数据集的根目录和输出目录
    # !! 修改为您的PTBDB数据集的根目录
    ptbdb_root_directory = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/"
    save_directory = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/save_records500_png/"
    
    # 确保保存目录存在
    os.makedirs(save_directory, exist_ok=True)
    
    # 查找所有 .hea 文件，os.walk会递归遍历所有子文件夹
    header_files = []
    for dirpath, _, filenames in os.walk(ptbdb_root_directory):
        for f in filenames:
            if f.endswith('.hea'):
                header_files.append(os.path.join(dirpath, f))

    if not header_files:
        print(f"错误：在目录 '{ptbdb_root_directory}' 及其子目录中没有找到任何 .hea 文件。")
    else:
        print(f"找到了 {len(header_files)} 个记录。开始处理...")
        
        # 循环处理每个记录
        for i, header_file_path in enumerate(header_files):
            # 获取记录的基本路径（去掉.hea后缀）
            record_base_path = os.path.splitext(header_file_path)[0]
            
            print(f"\n--- 正在处理记录 [{i+1}/{len(header_files)}]: {os.path.basename(record_base_path)} ---")
            try:
                process_ptbdb_record(record_base_path, save_directory)
            except Exception as e:
                print(f"处理记录 {os.path.basename(record_base_path)} 时发生未知错误: {e}")
    
    print("\n--- 所有记录处理完毕 ---")