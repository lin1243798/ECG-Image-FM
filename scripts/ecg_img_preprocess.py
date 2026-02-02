# scripts/ecg_img_preprocess.py

import cv2
import numpy as np
import os
import argparse

# --- 1. 自动定位项目根目录 ---
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CUR_DIR)

# --- 2. 可调参数 (保持原样) ---
STANDARDIZE_SIZE = (1000, 540)
MIN_PAPER_AREA_RATIO = 0.15
SIGNAL_DARKNESS_THRESHOLD = 70
TIGHT_CROP_PADDING_X = 5 
# 0.78 表示保留网格区域顶部 78% 的高度，用于切除底部的长导联
GRID_HEIGHT_CROP_RATIO = 0.78

def process_ecg_image(input_path, output_path):
    """
    (v4.2 混合策略最终版逻辑)
    1. Y轴 (垂直): 使用固定的比例裁剪，稳定地移除底部长导联。
    2. X轴 (水平): 使用最大连通组件法，精确地裁剪信号主体。
    """
    # --- 阶段 0: 加载图像 ---
    original_image = cv2.imread(input_path)
    if original_image is None:
        print(f"  [错误] 无法加载图像: {input_path}")
        return False

    # --- 阶段 1: 宏观定位ECG网格主区域 ---
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 4)
    kernel = np.ones((7,7), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"  [失败] 阶段1: 未能找到内容轮廓。")
        return False
        
    paper_contour = max(contours, key=cv2.contourArea)
    paper_area = cv2.contourArea(paper_contour)
    total_image_area = original_image.shape[0] * original_image.shape[1]
    
    if paper_area < total_image_area * MIN_PAPER_AREA_RATIO:
        print(f"  [失败] 阶段1: 检测到的主要内容区域过小。")
        return False
        
    x, y, w, h = cv2.boundingRect(paper_contour)
    grid_area_image = original_image[y:y+h, x:x+w]
    
    if grid_area_image.size == 0:
        print(f"  [失败] 阶段1: 裁剪出的网格区域为空。")
        return False

    # --- 阶段 2: 精确裁剪 (混合策略) ---
    
    # 步骤 2a: 使用固定比例进行垂直裁剪 (移除底部长导联)
    grid_h, grid_w = grid_area_image.shape[:2]
    crop_h = int(grid_h * GRID_HEIGHT_CROP_RATIO)
    twelve_lead_area = grid_area_image[0:crop_h, :]
    
    if twelve_lead_area.size == 0:
        print(f"  [失败] 按比例垂直裁剪后区域为空。")
        return False

    # 步骤 2b: 在12导联区域内，找到最大轮廓作为水平裁剪边界
    twelve_lead_hsv = cv2.cvtColor(twelve_lead_area, cv2.COLOR_BGR2HSV)
    # 提取暗色信号线条
    twelve_lead_mask = cv2.inRange(twelve_lead_hsv, np.array([0,0,0]), np.array([180,255,SIGNAL_DARKNESS_THRESHOLD]))
    contours, _ = cv2.findContours(twelve_lead_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"  [警告] 在12导联区域未找到任何信号轮廓。")
        final_crop = twelve_lead_area
    else:
        main_signal_contour = max(contours, key=cv2.contourArea)
        x_start, _, crop_w, _ = cv2.boundingRect(main_signal_contour)
        
        # 左右预留少量 Padding
        final_x_start = max(0, x_start - TIGHT_CROP_PADDING_X)
        final_x_end = min(twelve_lead_area.shape[1], x_start + crop_w + TIGHT_CROP_PADDING_X)
        final_crop = twelve_lead_area[:, final_x_start:final_x_end]

    # --- 阶段 3: 标准化与保存 ---
    if final_crop.size == 0:
        return False
        
    try:
        # 缩放到统一尺寸
        standardized_image = cv2.resize(final_crop, STANDARDIZE_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, standardized_image)
        return True
    except Exception as e:
        print(f"  [错误] 尺寸缩放或保存失败: {e}")
        return False

def main():
    # 使用 argparse 实现相对路径的灵活输入
    parser = argparse.ArgumentParser(description="ECG智能裁剪脚本 v4.2 - 混合策略终极版")
    
    # 默认路径指向项目根目录下的 data 文件夹
    default_input = os.path.join(BASE_DIR, "data", "raw_images")
    default_output = os.path.join(BASE_DIR, "data", "processed_images")
    
    parser.add_argument('--input_dir', type=str, default=default_input, help="原始心电图所在目录")
    parser.add_argument('--output_dir', type=str, default=default_output, help="处理后图像保存目录")
    
    args = parser.parse_args()

    print("="*60)
    print("=== ECG智能裁剪脚本 (v4.2) ===")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print("="*60)
    
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在，请检查路径。")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)

    # 遍历处理
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("未在目录中找到图像文件。")
        return

    success_count = 0
    for i, filename in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] 正在处理: {filename}")
        in_path = os.path.join(args.input_dir, filename)
        # 统一保存为 png 以保证质量
        out_path = os.path.join(args.output_dir, os.path.splitext(filename)[0] + "_processed.png")
        
        if process_ecg_image(in_path, out_path):
            success_count += 1
            print(f"  [成功]")
        else:
            print(f"  [失败]")

    print(f"\n处理完成！成功: {success_count}, 失败: {len(image_files) - success_count}")

if __name__ == "__main__":
    main()
