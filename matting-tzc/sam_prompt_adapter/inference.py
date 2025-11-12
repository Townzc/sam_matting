#!/usr/bin/env python
"""
SAM Prompt Adapter 推理脚本
基于sam_prompt_train_A+B_new.py训练的模型进行推理
"""

import argparse
import os
import sys
import yaml
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import glob
from pathlib import Path

# 抑制matplotlib字体警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 导入数据处理模块
sys.path.append('/raid/Data/huangtao/tangzhice/matting/baseline_A/data')
from data_processor import DataProcessor

# 设备设置（使用base.py配置）
sys.path.append('/raid/Data/huangtao/tangzhice/matting/baseline_A')
from base.base import setup_devices
setup_devices()  # 设置使用GPU 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 推理配置 =====
# 修改以下路径以匹配您的需求
CHECKPOINT_PATH = "/raid/Data/huangtao/tangzhice/matting/baseline_A/test_out/baseline_A+B/checkpoint_last.pth"
INPUT_PATH = "/raid/Data/huangtao/public/LNSM/test/image"  # 输入图像目录
MASK_PATH = "/raid/Data/huangtao/public/LNSM/test/binarymask"  # Binary mask目录  
OUTPUT_PATH = "/raid/Data/huangtao/tangzhice/matting/baseline_A/test_out/inference_out"  # 输出目录
VISUALIZE = True  # 是否显示可视化结果
BATCH_MODE = True  # 是否批量处理模式
# ==================


def load_model(checkpoint_path):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 模型checkpoint路径
        
    Returns:
        model: 加载的模型
    """
    # 导入模型
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    
    try:
        from sam_prompt_adapter import SAMAdapterPrompt
    except ImportError as e:
        print(f"Import error: {e}")
        # 尝试其他导入方式
        import importlib.util
        spec = importlib.util.spec_from_file_location("sam_prompt_adapter", 
                                                     os.path.join(current_dir, "sam_prompt_adapter.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        SAMAdapterPrompt = module.SAMAdapterPrompt
        
        # 创建模型
    model = SAMAdapterPrompt().to(device)
    
    # 加载checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
        
        # 加载权重
        model.load_state_dict(model_state_dict, strict=False)
        print("✅ Checkpoint loaded successfully")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    model.eval()
    return model


def load_image(image_path, target_size=(256, 256)):
    """
    加载和预处理图像
    
    Args:
        image_path: 图像路径
        target_size: 目标尺寸
        
    Returns:
        image_tensor: 预处理后的图像tensor
        original_size: 原始图像尺寸
    """
    # 读取图像
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path
    
    original_size = image.shape[:2]
    
    # 调整尺寸
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 转换为tensor并归一化
    image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    return image_tensor.to(device), original_size


def load_mask(mask_path, target_size=(256, 256)):
    """
    加载和预处理binary mask
    
    Args:
        mask_path: binary mask路径
        target_size: 目标尺寸
        
    Returns:
        mask_tensor: 预处理后的mask tensor
        trimap_tensor: 从mask生成的trimap tensor（用于sample_map）
    """
    if mask_path and os.path.exists(mask_path):
        # 读取binary mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot load mask: {mask_path}")
    else:
        # 如果没有mask，创建全前景的mask
        print("⚠️  No mask provided, using full foreground mask")
        mask = np.ones(target_size, dtype=np.uint8) * 255
    
    # 调整尺寸
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    # 转换binary mask为float并归一化到[0,1]
    mask_normalized = mask.astype(np.float32) / 255.0
    
    # 转换为tensor
    mask_tensor = torch.from_numpy(mask_normalized).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 从binary mask生成trimap（用于sample_map生成）
    # binary mask中：1表示前景，0表示背景，我们生成一个简单的trimap
    trimap = np.ones_like(mask, dtype=np.float32) * 128  # 默认全部为unknown
    trimap[mask > 127] = 255  # 前景区域
    trimap[mask < 128] = 0    # 背景区域
    
    # 为了有unknown区域，我们可以在边界创建一些unknown区域
    # 这里简化处理，直接使用全unknown
    trimap[:] = 128  # 全部设为unknown，让模型在整个区域进行预测
    
    trimap_tensor = torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    trimap_tensor = DataProcessor.normalize_trimap(trimap_tensor)
    
    return mask_tensor.to(device), trimap_tensor.to(device)


# create_mask_prompt函数已移除，直接使用binary mask作为mask提示


def inference_single_image(model, image_path, mask_path=None, output_path=None, visualize=False):
    """
    对单张图像进行推理
    
    Args:
        model: 加载的模型
        image_path: 输入图像路径
        mask_path: binary mask路径（可选）
        output_path: 输出路径
        visualize: 是否可视化结果
        
    Returns:
        pred_alpha: 预测的alpha matte
    """
    print(f"Processing: {image_path}")
    
    # 加载图像和mask
    image_tensor, original_size = load_image(image_path)
    mask_tensor, trimap_tensor = load_mask(mask_path)
    
    # 创建sample_map（从trimap生成，用于指定计算损失的区域）
    sample_map = DataProcessor.create_sample_map(trimap_tensor)
    
    # 直接使用binary mask作为mask提示
    mask_prompt = mask_tensor
    
    # 推理
    with torch.no_grad():
        # 设置模型输入
        model.set_input(image_tensor, trimap_tensor, mask_inputs=mask_prompt, sample_map=sample_map)
        
        # 前向传播
        model.forward()
        
        # 获取预测结果
        pred_alpha = model.pred_mask
        
        # 转换为numpy
        pred_alpha_np = pred_alpha.squeeze().cpu().numpy()
        
        # 调整回原始尺寸
        pred_alpha_np = cv2.resize(pred_alpha_np, (original_size[1], original_size[0]), 
                                  interpolation=cv2.INTER_LINEAR)
        
        # 确保值在[0,1]范围内
        pred_alpha_np = np.clip(pred_alpha_np, 0, 1)
    
    # 保存结果
    if output_path:
        # 转换为0-255范围并保存
        alpha_save = (pred_alpha_np * 255).astype(np.uint8)
        cv2.imwrite(output_path, alpha_save)
        print(f"✅ Result saved to: {output_path}")
    
    # 可视化
    if visualize:
        visualize_result(image_path, mask_path, pred_alpha_np, output_path)
    
    return pred_alpha_np


def visualize_result(image_path, mask_path, pred_alpha, output_path=None):
    """
    可视化推理结果
    
    Args:
        image_path: 原始图像路径
        mask_path: binary mask路径
        pred_alpha: 预测的alpha
        output_path: 输出路径（用于生成可视化文件名）
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 原始图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Binary Mask
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Binary Mask Input')
    else:
        axes[1].text(0.5, 0.5, 'No Mask', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Mask (Not Provided)')
    axes[1].axis('off')
    
    # 预测的Alpha
    axes[2].imshow(pred_alpha, cmap='gray')
    axes[2].set_title('Predicted Alpha')
    axes[2].axis('off')
    
    # 合成结果（如果可能）
    try:
        # 调整alpha到原始图像尺寸
        alpha_resized = cv2.resize(pred_alpha, (image.shape[1], image.shape[0]))
        alpha_3ch = np.stack([alpha_resized] * 3, axis=2)
        composite = image * alpha_3ch
        axes[3].imshow(composite.astype(np.uint8))
        axes[3].set_title('Composite Result')
    except:
        axes[3].text(0.5, 0.5, 'Composite Error', ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('Composite')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # 保存可视化结果
    if output_path:
        vis_path = output_path.replace('.png', '_visualization.png').replace('.jpg', '_visualization.png')
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved to: {vis_path}")
    
    plt.show()


def batch_inference(model, input_dir, mask_dir=None, output_dir=None, visualize=False):
    """
    批量推理
    
    Args:
        model: 加载的模型
        input_dir: 输入图像目录
        mask_dir: binary mask目录（可选）
        output_dir: 输出目录
        visualize: 是否可视化
    """
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} images to process")
    
    # 批量处理
    for image_path in tqdm(image_files, desc="Processing images"):
        # 获取图像名称
        img_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(img_name)[0]
        
        # 构建mask路径
        mask_path = None
        if mask_dir:
            mask_path = os.path.join(mask_dir, img_name)
            if not os.path.exists(mask_path):
                # 尝试其他扩展名
                for ext in ['.png', '.jpg', '.jpeg']:
                    mask_candidate = os.path.join(mask_dir, name_without_ext + ext)
                    if os.path.exists(mask_candidate):
                        mask_path = mask_candidate
                        break
                else:
                    mask_path = None
        
        # 构建输出路径
        if output_dir:
            output_path = os.path.join(output_dir, name_without_ext + '.png')  # 直接使用原文件名
        else:
            output_path = None
        
        try:
            # 执行推理
            inference_single_image(model, image_path, mask_path, output_path, visualize=False)
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
    
    print(f"✅ Batch processing completed. Results saved to: {output_dir}")


def main():
    print("🚀 Starting SAM Prompt Adapter Inference")
    print(f"📄 Checkpoint: {CHECKPOINT_PATH}")
    print(f"📁 Input: {INPUT_PATH}")
    print(f"🎭 Mask: {MASK_PATH}")
    print(f"💾 Output: {OUTPUT_PATH}")
    
    # 加载模型
    model = load_model(CHECKPOINT_PATH)
    
    if BATCH_MODE or os.path.isdir(INPUT_PATH):
        # 批量处理
        batch_inference(
            model=model,
            input_dir=INPUT_PATH,
            mask_dir=MASK_PATH,
            output_dir=OUTPUT_PATH,
            visualize=VISUALIZE
        )
    else:
        # 单张处理
        inference_single_image(
            model=model,
            image_path=INPUT_PATH,
            mask_path=MASK_PATH,
            output_path=OUTPUT_PATH,
            visualize=VISUALIZE
        )
    
    print("🎉 Inference completed!")


if __name__ == '__main__':
        main()