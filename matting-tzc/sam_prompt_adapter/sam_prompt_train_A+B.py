import argparse
import os

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from statistics import mean
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import random
import sys

# 添加dim_dataset路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append('/raid/Data/huangtao/tangzhice/matting/baseline_A/data/dim_dataset.py')
from dim_dataset import DataGenerator, ImageFileTrain

# 添加Lung20Dataset路径
sys.path.append('/raid/Data/huangtao/zhangqy/LNMatte/PFM_Net_reply_v1/data')
from lung_dataset import Lung20Dataset

# 设备和local_rank设置（分布式兼容）
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)


def eval_metrics(loader, model):
    """评估抠图指标"""
    model.eval()
    
    pbar = tqdm(total=len(loader), leave=False, desc='val')

    pred_list = []
    gt_list = []
    
    with torch.no_grad():
        for batch in loader:
            # 移动数据到GPU
            batch = DataProcessor.move_batch_to_device(batch, device)

            # 使用统一的数据提取方法
            inp = DataProcessor.extract_image_from_batch(batch)
            gt = DataProcessor.extract_alpha_from_batch(batch)
            mask_inputs = DataProcessor.extract_mask_from_batch(batch)
            
            # 前向传播
            if hasattr(model, "module"):
                model.module.set_input(inp, gt, mask_inputs=mask_inputs, sample_map=batch.get('sample_map', None))
                model.module.forward()
                pred = model.module.pred_mask
            else:
                model.set_input(inp, gt, mask_inputs=mask_inputs, sample_map=batch.get('sample_map', None))
                model.forward()
                pred = model.pred_mask

            # 单GPU训练，直接收集结果
            pred_list.append(pred)
            gt_list.append(gt)
            
            pbar.update(1)

    pbar.close()

    # 计算指标
    pred_list = torch.cat(pred_list, 0)
    gt_list = torch.cat(gt_list, 0)
    
    # 计算MSE
    mse = F.mse_loss(pred_list, gt_list).item()
    
    # 计算SAD (Sum of Absolute Differences)
    sad = torch.abs(pred_list - gt_list).mean().item()
    
    return mse, sad


def prepare_training():
    # 导入你的模型
    import sys
    import os
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
    
    model = SAMAdapterPrompt().cuda()
    
    # 分组学习率设置
    param_groups = []
    
    # 只训练Adapter与解码器等新层；完全排除ViT主干非Adapter参数
    adapter_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        is_image_encoder = ("image_encoder" in name)
        is_adapter = ("prompt_adapter" in name) or ("adapter" in name.lower())
        
        if is_image_encoder and not is_adapter:
            # 冻结ViT主干非Adapter参数
            param.requires_grad = False
            print(f"Freeze (exclude from optimizer): {name}")
            continue
        
        # 其余参与训练
        param.requires_grad = True
        if is_adapter:
            adapter_params.append(param)
            print(f"Adapter param: {name}")
        else:
            other_params.append(param)
            print(f"Other param: {name}")
    
    # 设置不同学习率（Adapter用更大学习率）
    base_lr = config['optimizer']['lr']
    weight_decay = config['optimizer'].get('weight_decay', 0)
    adapter_lr_multiplier = config['optimizer'].get('adapter_lr_multiplier', 5.0)
    
    if adapter_params:
        param_groups.append({
            'params': adapter_params,
            'lr': base_lr * adapter_lr_multiplier,
            'weight_decay': weight_decay,
        })
        print(f"Adapter: {len(adapter_params)} params, lr={base_lr * adapter_lr_multiplier}")
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
        })
        print(f"Other: {len(other_params)} params, lr={base_lr}")
    
    optimizer = torch.optim.Adam(param_groups)
    
    if config.get('resume') is not None:
        epoch_start = config.get('resume') + 1
    else:
        epoch_start = 1
    
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model: #params={total_params}, #trainable_params={trainable_params}')
    
    return model, optimizer, epoch_start, lr_scheduler


class DataProcessor:
    """统一的数据处理工具类"""
    
    @staticmethod
    def extract_image_from_batch(batch):
        """从batch中提取图像数据"""
        if 'inp' in batch:
            return batch['inp']
        elif 'image' in batch:
            return batch['image']
        elif 'fg' in batch and 'bg' in batch and 'alpha' in batch:
            # 合成图像
            fg = batch['fg']
            bg = batch['bg']
            alpha = batch['alpha']
            return fg * alpha + bg * (1 - alpha)
        else:
            raise ValueError("No valid image found in batch")
    
    @staticmethod
    def extract_alpha_from_batch(batch):
        """从batch中提取alpha数据"""
        if 'gt' in batch:
            return batch['gt']
        elif 'alpha' in batch:
            return batch['alpha']
        else:
            raise ValueError("No valid alpha/gt found in batch")
    
    @staticmethod
    def extract_mask_from_batch(batch):
        """从batch中提取mask数据"""
        # 优先使用DataGenerator自动生成的mask（更精确）
        mask_inputs = batch.get('mask', None)
        if mask_inputs is not None:
            # 找到了自动生成的mask，直接使用
            return mask_inputs.float()
        elif 'trimap' in batch:
            # 降级处理：如果没有mask但有trimap，将trimap转换为mask
            mask_inputs = DataProcessor.trimap_to_mask_prompt(batch['trimap'])
            return mask_inputs
        else:
            # 没有任何引导信息
            return None
    
    @staticmethod
    def move_batch_to_device(batch, device):
        """将batch中的tensor数据移动到指定设备"""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch
    
    @staticmethod
    def normalize_trimap(trimap):
        """强制trimap三值化"""
        trimap = trimap.float()
        trimap[(trimap == 127) | (trimap == 128)] = 0.5
        trimap[trimap == 255] = 1.0
        trimap[trimap == 0] = 0.0
        
        # 验证trimap值
        if not torch.all((trimap == 0) | (trimap == 0.5) | (trimap == 1.0)):
            raise ValueError("trimap中存在非0/0.5/1的值，请检查数据！")
        
        return trimap
    
    @staticmethod
    def create_sample_map(trimap):
        """创建sample_map（unknown区域的mask）"""
        sample_map = torch.isclose(trimap, torch.tensor(0.5, device=trimap.device)).float()
        if sample_map.sum() == 0:
            print("[Warning] 当前batch没有unknown区域，建议检查trimap数据！")
        return sample_map
    
    @staticmethod
    def trimap_to_mask_prompt(trimap):
        """
        将trimap转换为SAM的mask prompt格式
        trimap: [B, C, H, W] 值为0(背景), 0.5(未知), 1.0(前景)
        返回: mask_prompt [B, C, H, W] 适用于SAM的mask提示
        """
        # 创建mask prompt：前景区域为1，背景区域为0，未知区域保持原值作为引导
        mask_prompt = trimap.clone()
        
        # 方法1：直接使用trimap作为mask（推荐）
        # SAM可以理解0.5作为uncertain区域
        return mask_prompt


class VisualizationManager:
    """可视化管理类"""
    
    def __init__(self, save_path, max_samples=3):
        """
        初始化可视化管理器
        
        Args:
            save_path: 保存路径
            max_samples: 最大保存样本数
        """
        self.save_path = save_path
        self.vis_dir = os.path.join(save_path, 'visualizations')
        self.max_samples = max_samples
        self.vis_samples = []
        
        # 创建可视化目录
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def collect_sample(self, inp, gt, pred, batch_idx, natural_batch_size=59):
        """
        收集可视化样本 - 混合训练版本
        同时保存自然图像和医学图像样本
        
        Args:
            inp: 输入图像 [B, C, H, W]
            gt: 真实标签 [B, C, H, W] 
            pred: 预测结果 [B, C, H, W]
            batch_idx: batch索引
            natural_batch_size: 自然图像的batch size
        """
        # 限制总样本数量，但要同时保存自然和医学图像
        max_pairs = self.max_samples // 2  # 每对包含1个自然+1个医学图像
        
        if len(self.vis_samples) < self.max_samples and batch_idx < max_pairs:
            # 保存自然图像样本（第一个）
            nat_sample = {
                'input': inp[0].cpu().detach(),  # 自然图像（索引0）
                'gt': gt[0].cpu().detach(),
                'pred': pred[0].cpu().detach() if pred is not None else None,
                'batch_idx': batch_idx,
                'image_type': 'natural'
            }
            self.vis_samples.append(nat_sample)
            
            # 保存医学图像样本（从natural_batch_size开始）
            if inp.size(0) > natural_batch_size:  # 确保有医学图像
                med_sample = {
                    'input': inp[natural_batch_size].cpu().detach(),  # 医学图像（第一个医学样本）
                    'gt': gt[natural_batch_size].cpu().detach(),
                    'pred': pred[natural_batch_size].cpu().detach() if pred is not None else None,
                    'batch_idx': batch_idx,
                    'image_type': 'medical'
                }
                self.vis_samples.append(med_sample)
                
                return True
        return False
    
    def save_samples(self, epoch):
        """保存所有收集的样本"""
        if not self.vis_samples:
            return
        
        for i, sample in enumerate(self.vis_samples):
            self._save_single_sample(sample, epoch, i)
        
        print(f"Saved {len(self.vis_samples)} visualization samples to {self.vis_dir}")
    
    def _save_single_sample(self, sample, epoch, sample_idx):
        """保存单个样本"""
        # 获取图像类型信息
        img_type = sample.get('image_type', 'unknown')
        img_type_display = '自然图像' if img_type == 'natural' else '医学图像' if img_type == 'medical' else img_type
        
        # 创建图像网格
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 输入图像 (RGB)
        input_img = sample['input'].permute(1, 2, 0).numpy()
        input_img = self._process_image_for_display(input_img)
        
        axes[0].imshow(input_img)
        axes[0].set_title(f'Input Image ({img_type_display})')
        axes[0].axis('off')
        
        # Ground Truth Alpha
        gt_alpha = sample['gt'].squeeze().numpy()
        axes[1].imshow(gt_alpha, cmap='gray')
        axes[1].set_title('Ground Truth Alpha')
        axes[1].axis('off')
        
        # Predicted Alpha
        if sample['pred'] is not None:
            pred_alpha = sample['pred'].squeeze().numpy()
            axes[2].imshow(pred_alpha, cmap='gray')
            axes[2].set_title('Predicted Alpha')
        else:
            axes[2].text(0.5, 0.5, 'No Prediction', ha='center', va='center', 
                        transform=axes[2].transAxes)
        axes[2].axis('off')
        
        # 保存合成图像 - 文件名包含图像类型
        plt.tight_layout()
        composite_path = os.path.join(self.vis_dir, 
                                    f'epoch_{epoch}_{img_type}_sample_{sample_idx+1}_batch_{sample["batch_idx"]}.png')
        plt.savefig(composite_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _process_image_for_display(self, input_img):
        """处理图像用于显示"""
        # 检查并处理无效值
        if np.any(np.isnan(input_img)) or np.any(np.isinf(input_img)):
            print(f"Warning: Input image contains NaN or Inf values, replacing with 0")
            input_img = np.nan_to_num(input_img, nan=0.0, posinf=1.0, neginf=0.0)
        
        input_img = np.clip(input_img, 0, 1)
        
        # 打印调试信息
        print(f"Input image stats - min: {input_img.min():.4f}, max: {input_img.max():.4f}, mean: {input_img.mean():.4f}")
        print(f"Input image shape: {input_img.shape}, dtype: {input_img.dtype}")
        
        return input_img
    
    def clear_samples(self):
        """清空已收集的样本"""
        self.vis_samples.clear()
    
    def save_debug_sample(self, inp, gt, pred, batch_idx, prefix="debug", natural_batch_size=59):
        """
        立即保存调试样本（不加入收集队列）
        同时保存自然图像和医学图像的调试样本
        """
        # 保存自然图像调试样本
        debug_sample_nat = {
            'input': inp[0].cpu().detach() if inp is not None else None,
            'gt': gt[0].cpu().detach() if gt is not None else None,
            'pred': pred[0].cpu().detach() if pred is not None else None,
            'batch_idx': batch_idx,
            'image_type': 'natural'
        }
        self._save_single_sample(debug_sample_nat, prefix, 0)
        
        # 保存医学图像调试样本（如果有）
        if inp is not None and inp.size(0) > natural_batch_size:
            debug_sample_med = {
                'input': inp[natural_batch_size].cpu().detach(),
                'gt': gt[natural_batch_size].cpu().detach() if gt is not None else None,
                'pred': pred[natural_batch_size].cpu().detach() if pred is not None else None,
                'batch_idx': batch_idx,
                'image_type': 'medical'
            }
            self._save_single_sample(debug_sample_med, prefix, 1)


def trimap_to_mask_prompt(trimap):
    """兼容性函数，调用DataProcessor中的静态方法"""
    return DataProcessor.trimap_to_mask_prompt(trimap)


def create_mixed_data_loaders(config, args):
    """创建混合数据加载器"""
    mixed_config = config['mixed_training']
    loader_config = config['data_loader']
    
    natural_batch_size = mixed_config['natural_batch_size']
    medical_batch_size = mixed_config['medical_batch_size']
    image_size = mixed_config['image_size']
    
    print(f"混合训练配置: {natural_batch_size}张自然图像 + {medical_batch_size}张医学图像")
    
    # 1. 创建自然图像数据集
    natural_data = ImageFileTrain(
        alpha_dir=config['train_dataset']['alpha_dir'],
        fg_dir=config['train_dataset']['fg_dir'],
        bg_dir=config['train_dataset']['bg_dir']
    )
    natural_dataset = DataGenerator(natural_data, phase="train", inp_size=image_size)
    natural_loader = DataLoader(
        natural_dataset,
        batch_size=natural_batch_size,
        shuffle=True,
        num_workers=loader_config['num_workers'],
        pin_memory=loader_config['pin_memory'],
        drop_last=loader_config['drop_last']
    )
    
    print(f"自然训练集大小: {len(natural_dataset)}")
    
    # 2. 创建肺部医学图像数据集（记忆池）- 使用修正版
    try:
        from lung_dataset_fixed import Lung20DatasetFixed
        lung_dataset = Lung20DatasetFixed(args.root_lung20)
    except ImportError:
        # 回退到原版（如果修正版不可用）
        lung_dataset = Lung20Dataset(args.root_lung20)
    lung_loader = DataLoader(
        lung_dataset, 
        batch_size=medical_batch_size, 
        shuffle=True, 
        drop_last=False,
        num_workers=loader_config['num_workers'],
        pin_memory=loader_config['pin_memory']
    )
    lung_iter = iter(lung_loader)
    
    print(f"肺部训练集大小: {len(lung_dataset)}")
    
    return natural_loader, lung_loader, lung_iter


def train(natural_loader, lung_iter, lung_loader, model, optimizer, save_path=None, epoch=None):
    """混合数据训练"""
    model.train()
    
    # 从配置中获取参数
    mixed_config = config['mixed_training']
    natural_batch_size = mixed_config['natural_batch_size']
    medical_batch_size = mixed_config['medical_batch_size']
    vis_frequency = mixed_config['visualization_frequency']

    pbar = tqdm(total=len(natural_loader), leave=False, desc='train')

    loss_list = []
    loss_details = {
        'unknown_l1_loss': [],
        'known_l1_loss': [],
        'loss_gradient_penalty': [],
        'loss_pha_laplacian': []
    }
    
    # 创建可视化管理器
    vis_manager = VisualizationManager(save_path) if save_path else None
    
    for batch_idx, nat_batch in enumerate(natural_loader):
        # 清零梯度
        optimizer.zero_grad()
        
        # 获取肺部数据（循环使用）
        try:
            lung_batch = next(lung_iter)
        except StopIteration:
            lung_iter = iter(lung_loader)
            lung_batch = next(lung_iter)
        
        # 移动数据到GPU
        nat_batch = DataProcessor.move_batch_to_device(nat_batch, device)
        lung_batch = DataProcessor.move_batch_to_device(lung_batch, device)
        
        # 拼接batch数据
        if 'image' in nat_batch and 'image' in lung_batch:
            # 使用DataGenerator格式
            img = torch.cat([nat_batch["image"], lung_batch["image"]], dim=0)
            alpha = torch.cat([nat_batch["alpha"], lung_batch["alpha"]], dim=0)
            
            # 处理trimap - 使用DataGenerator自动生成的trimap和mask
            if 'trimap' in nat_batch and 'trimap' in lung_batch:
                # 两个数据集都有trimap，直接合并（推荐方式）
                trimap = torch.cat([nat_batch["trimap"], lung_batch["trimap"]], dim=0)
                if batch_idx == 0:  # 只在第一个batch打印信息
                    print("[Info] 使用自然图像自动生成的trimap + 医学图像精确trimap")
            else:
                # 降级处理：如果自然图像没有trimap（不应该发生）
                if batch_idx == 0:
                    print("[Warning] 自然图像数据缺少trimap，使用降级策略")
                nat_trimap = torch.ones_like(nat_batch["alpha"]) * 0.5  # 全部设为unknown
                trimap = torch.cat([nat_trimap, lung_batch["trimap"]], dim=0)
                
            # 处理mask - 优先使用DataGenerator生成的mask
            if 'mask' in nat_batch and 'mask' in lung_batch:
                # 合并自然图像的自动生成mask和医学图像mask
                masks = torch.cat([nat_batch["mask"], lung_batch.get("mask", torch.ones_like(lung_batch["alpha"]))], dim=0)
                if batch_idx == 0:
                    print("[Info] 使用自然图像自动生成的mask + 医学图像mask")
            elif 'mask' in nat_batch:
                # 只有自然图像有mask，为医学图像创建默认mask
                lung_mask = torch.ones_like(lung_batch["alpha"])  # 医学图像全图mask
                masks = torch.cat([nat_batch["mask"], lung_mask], dim=0)
                if batch_idx == 0:
                    print("[Info] 使用自然图像自动生成的mask + 医学图像默认全图mask")
            else:
                masks = None
                if batch_idx == 0:
                    print("[Warning] 未找到自然图像的mask数据")
                
        else:
            # 兼容其他数据格式
            if 'fg' in nat_batch and 'bg' in nat_batch and 'alpha' in nat_batch:
                # 合成自然图像
                fg = nat_batch['fg']
                bg = nat_batch['bg'] 
                alpha_nat = nat_batch['alpha']
                img_nat = fg * alpha_nat + bg * (1 - alpha_nat)
                
                img = torch.cat([img_nat, lung_batch["image"]], dim=0)
                alpha = torch.cat([alpha_nat, lung_batch["alpha"]], dim=0)
                
                # 检查是否有自动生成的trimap和mask
                if 'trimap' in nat_batch:
                    trimap = torch.cat([nat_batch["trimap"], lung_batch["trimap"]], dim=0)
                    if batch_idx == 0:
                        print("[Info] 使用自然图像自动生成的trimap（fg/bg格式）")
                else:
                    # 生成trimap
                    nat_trimap = torch.ones_like(alpha_nat) * 0.5
                    trimap = torch.cat([nat_trimap, lung_batch["trimap"]], dim=0)
                    if batch_idx == 0:
                        print("[Warning] fg/bg格式数据缺少trimap，使用全unknown策略")
                
                if 'mask' in nat_batch:
                    lung_mask = torch.ones_like(lung_batch["alpha"])
                    masks = torch.cat([nat_batch["mask"], lung_mask], dim=0)
                    if batch_idx == 0:
                        print("[Info] 使用自然图像自动生成的mask（fg/bg格式）")
                else:
                    masks = None
            else:
                raise ValueError("无法找到有效的图像数据格式")
        
        # 使用统一的trimap处理方法
        trimap = DataProcessor.normalize_trimap(trimap)
        sample_map = DataProcessor.create_sample_map(trimap)
        
        # 准备输入数据（SAM格式）
        inp = img[:, :3] if img.shape[1] > 3 else img  # 确保只有RGB通道
        gt = alpha
        
        # 准备mask提示 - 优先使用自动生成的mask，否则从trimap转换
        if masks is not None:
            # 使用DataGenerator自动生成的精确mask（推荐）
            mask_prompt = masks.float()
            if batch_idx == 0:
                print("[Info] 使用自动生成的精确mask作为SAM引导")
        else:
            # 降级处理：从trimap转换为mask提示
            mask_prompt = DataProcessor.trimap_to_mask_prompt(trimap)
            if batch_idx == 0:
                print("[Warning] 使用trimap转换的mask作为SAM引导")
        
        # SAM模型训练
        if hasattr(model, "module"):
            model.module.set_input(inp, gt, mask_inputs=mask_prompt, sample_map=sample_map)
            model.module.forward()
            result = model.module.optimize_parameters()
        else:
            model.set_input(inp, gt, mask_inputs=mask_prompt, sample_map=sample_map)
            model.forward()
            result = model.optimize_parameters()
        
        if result is not None:
            total_loss, losses = result
            # 确保loss是标量，避免内存累积
            if hasattr(total_loss, 'item'):
                loss_list.append(total_loss.item())
            else:
                loss_list.append(total_loss)
            
            # 收集详细损失信息
            for loss_name, loss_value in losses.items():
                if loss_name in loss_details:
                    if hasattr(loss_value, 'item'):
                        loss_details[loss_name].append(loss_value.item())
                    else:
                        loss_details[loss_name].append(loss_value)
            
            # 收集可视化样本
            if vis_manager and batch_idx % vis_frequency == 0:
                pred_mask = model.module.pred_mask if hasattr(model, "module") else model.pred_mask
                # 传递自然图像batch size用于正确分离自然和医学图像样本
                natural_batch_size = mixed_config['natural_batch_size']
                collected = vis_manager.collect_sample(inp, gt, pred_mask, batch_idx, natural_batch_size)
                
                # 立即保存调试样本（第一个样本）
                if collected and len(vis_manager.vis_samples) <= 2:  # 现在每次收集2个样本（自然+医学）
                    vis_manager.save_debug_sample(inp, gt, pred_mask, batch_idx, f"batch_{batch_idx}", natural_batch_size)
            
            # 执行优化器步骤
            optimizer.step()
        
        pbar.update(1)
        
        # 更新进度条显示详细信息
        if len(loss_list) > 0:
            current_loss = loss_list[-1]
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'avg_loss': f'{sum(loss_list)/len(loss_list):.4f}',
                'nat_batch': natural_batch_size,
                'med_batch': medical_batch_size
            })

    pbar.close()

    if len(loss_list) == 0:
        return 0.0, lung_iter  # 返回更新后的lung_iter
    
    # 计算平均损失
    avg_loss = sum(loss_list) / len(loss_list)
    
    # 计算详细损失的平均值
    avg_loss_details = {}
    for loss_name, loss_values in loss_details.items():
        if loss_values:
            avg_loss_details[loss_name] = sum(loss_values) / len(loss_values)
    
    # 打印训练统计信息
    print(f"\n=== 混合数据训练统计 ===")
    print(f"总批次数: {len(natural_loader)}, 每批次: {natural_batch_size}自然图像 + {medical_batch_size}医学图像")
    print(f"平均总损失: {avg_loss:.4f}")
    for loss_name, avg_value in avg_loss_details.items():
        print(f"  {loss_name}: {avg_value:.4f}")
    
    # 保存可视化结果
    if vis_manager and epoch is not None:
        vis_manager.save_samples(epoch)
        vis_manager.clear_samples()  # 清空样本以释放内存
    
    # 清理内存
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return avg_loss, lung_iter




def main(config_, save_path, args):
    global config
    config = config_
    
    # 初始化分布式环境（如果是分布式训练）
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        print(f"Distributed training: rank {dist.get_rank()} / world {dist.get_world_size()}")
    else:
        print("Single process training")

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # 创建混合数据加载器 
    natural_loader, lung_loader, lung_iter = create_mixed_data_loaders(config, args)
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    
    # 分布式包装模型
    if dist.is_available() and dist.is_initialized():
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 加载预训练权重
    if config.get('sam_checkpoint') and os.path.exists(config['sam_checkpoint']):
        print(f"Loading SAM checkpoint: {config['sam_checkpoint']}")
        checkpoint = torch.load(config['sam_checkpoint'], map_location='cpu')
        model_dict = model.state_dict()
        
        # 分类处理参数
        loaded_params = []
        adapted_params = []
        skipped_params = []
        frozen_params = []

        for name, param in checkpoint.items():
            if name in model_dict:
                if model_dict[name].shape == param.shape:
                    # 形状完全匹配，直接加载
                    model_dict[name] = param
                    loaded_params.append(name)
                else:
                    # 形状不匹配，跳过
                    skipped_params.append(name)
            else:
                # 模型中没有的参数，跳过
                skipped_params.append(name)

        # 加载权重
        model.load_state_dict(model_dict, strict=False)
        
        print(f"=== 权重加载统计 ===")
        print(f"直接加载: {len(loaded_params)} 个参数")
        print(f"智能适配: {len(adapted_params)} 个参数")
        print(f"跳过参数: {len(skipped_params)} 个参数")
        
        if loaded_params:
            print("直接加载的参数示例:")
            for i, name in enumerate(loaded_params[:3]):
                print(f"  {name}")
        if adapted_params:
            print("智能适配的参数:")
            for name in adapted_params:
                print(f"  {name}")
    else:
        print("No SAM checkpoint provided or file not found, training from scratch")
    
    # 设置参数训练策略：冻结ViT主干，只训练Adapter和新层
    print("\n=== 参数训练策略 ===")
    frozen_params = []  # 初始化frozen_params列表
    for name, para in model.named_parameters():
        if "image_encoder" in name:
            if "prompt_generator" in name or "adapter" in name.lower():
                # Adapter相关参数 - 训练
                para.requires_grad_(True)
                print(f"  Training: {name} (Adapter)")
            else:
                # ViT主干参数 - 冻结
                para.requires_grad_(False)
                frozen_params.append(name)
                print(f"  Frozen: {name} (ViT backbone)")
        else:
            # 其他参数（如decoder等） - 训练
            para.requires_grad_(True)
            print(f"  Training: {name} (Other)")

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== 参数统计 ===")
    print(f"总参数数: {model_total_params:,}")
    print(f"可训练参数: {model_grad_params:,}")
    print(f"冻结参数: {model_total_params - model_grad_params:,}")
    print(f"训练比例: {model_grad_params/model_total_params*100:.2f}%")

    # 单GPU训练
    # model = model.to(device) # 分布式训练时，模型已移动到GPU

    epoch_max = config['epoch_max']
    checkpoint_frequency = config['mixed_training']['checkpoint_frequency']
    
    for epoch in range(epoch_start, epoch_max + 1):
        # 使用混合数据训练
        train_loss, lung_iter = train(natural_loader, lung_iter, lung_loader, model, optimizer, save_path, epoch)
        lr_scheduler.step()

        print(f'epoch {epoch}/{epoch_max}')
        print(f'train loss: {train_loss:.4f}')

        # 保存检查点
        print(f"Attempting to save 'last' checkpoint...")
        save(config, model, save_path, 'last')
        
        # 按配置频率保存检查点
        if epoch % checkpoint_frequency == 0:
            print(f"Attempting to save epoch {epoch} checkpoint...")
            save(config, model, save_path, f'epoch_{epoch}')




def save(config, model, save_path, name):
    """保存模型检查点"""
    try:
        # 如果是分布式训练，保存原始模型的状态字典
        if hasattr(model, "module"):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state_dict,
            'config': config
        }
        
        # 只在主进程中保存，避免多进程冲突
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            save_file = os.path.join(save_path, f"checkpoint_{name}.pth")
            print(f"Saving checkpoint to: {save_file}")
            torch.save(checkpoint, save_file)
            print(f"Successfully saved checkpoint: {save_file}")
        else:
            print(f"Rank {dist.get_rank()}: Skipping save (not main process)")
            
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        import traceback
        traceback.print_exc()


def make_data_loader(dataset_config, tag='train', distributed=False):
    """创建数据加载器"""
    if tag == 'train':
        # 训练集使用ImageFileTrain
        data = ImageFileTrain(
            alpha_dir=dataset_config['alpha_dir'],
            fg_dir=dataset_config['fg_dir'],
            bg_dir=dataset_config['bg_dir']
        )
    else:
        # 验证集也使用ImageFileTrain，但phase设为'val'
        data = ImageFileTrain(
            alpha_dir=dataset_config['alpha_dir'],
            fg_dir=dataset_config['fg_dir'],
            bg_dir=dataset_config['bg_dir']
        )
    
    # 创建DataGenerator
    dataset = DataGenerator(
        data=data,
        phase=tag,
        inp_size=dataset_config['size']
    )
    
    print(f"{tag} dataset: size={len(dataset)}")
    
    # 调试第一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Debug - Sample 0:")
        # 注意：DataGenerator返回的键名可能不同
        if 'inp' in sample:
            print(f"  Image shape: {sample['inp'].shape}, range: [{sample['inp'].min():.4f}, {sample['inp'].max():.4f}]")
        if 'image' in sample:
            print(f"  Image shape: {sample['image'].shape}, range: [{sample['image'].min():.4f}, {sample['image'].max():.4f}]")
        if 'alpha' in sample:
            print(f"  Alpha shape: {sample['alpha'].shape}, range: [{sample['alpha'].min():.4f}, {sample['alpha'].max():.4f}]")
        for k, v in sample.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={tuple(v.shape)}")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")
    
    loader_config = config.get('data_loader', {})
    dataloader_args = {
        'batch_size': dataset_config['batch_size'],
        'shuffle': (tag == 'train') and not distributed,
        'num_workers': loader_config.get('num_workers', 4),
        'pin_memory': loader_config.get('pin_memory', True)
    }
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        dataloader_args['sampler'] = DistributedSampler(dataset)
        dataloader_args['shuffle'] = False
    dataloader = DataLoader(dataset, **dataloader_args)
    
    return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train_sam_matting.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local-rank", type=int, default=-1, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")  # 兼容两种参数名
    
    # 添加医学数据集路径参数（参考train_reply_v1.py）
    parser.add_argument('--root_lung20', type=str, 
                       default="/raid/Data/huangtao/zhangqy/dataset/LNSM/lung20",
                       help='20张肺部输入目录')
    args = parser.parse_args()

    # 默认配置
    config = {
        'train_dataset': {
            'alpha_dir': "/raid/Data/huangtao/public/matting/vitmatte/alpha",
            'fg_dir': "/raid/Data/huangtao/public/matting/vitmatte/fg", 
            'bg_dir': "/raid/Data/huangtao/public/matting/vitmatte/bg",
            'batch_size': 64,  # 总batch size: natural_batch_size + medical_batch_size
            'size': 256
        },
        'val_dataset': {
            'alpha_dir': "/raid/Data/huangtao/public/matting/vitmatte/alpha",
            'fg_dir': "/raid/Data/huangtao/public/matting/vitmatte/fg",
            'bg_dir': "/raid/Data/huangtao/public/matting/vitmatte/bg", 
            'batch_size': 4,
            'size': 256
        },
        'mixed_training': {
            'natural_batch_size': 59,  # 每个batch中自然图像的数量
            'medical_batch_size': 5,   # 每个batch中医学图像的数量
            'image_size': 256,         # 输入图像尺寸
            'visualization_frequency': 100,  # 每多少个batch保存一次可视化
            'checkpoint_frequency': 10,      # 每多少个epoch保存一次检查点
        },
        'data_loader': {
            'num_workers': 4,          # DataLoader的工作进程数
            'pin_memory': True,        # 是否将数据加载到固定内存
            'drop_last': True,         # 是否丢弃最后不完整的batch
        },
        'sam_checkpoint': "/raid/Data/huangtao/tangzhice/matting/baseline_A/pretrained/sam_vit_b_01ec64.pth",
        'optimizer': {
            'lr': 5e-5,
            'weight_decay': 1e-4,
            'adapter_lr_multiplier': 5.0,  # Adapter层的学习率倍数
        },
        'epoch_max': 100,
        'epoch_val': 5,
        'lr_min': 1e-6
    }

    save_name = args.name
    if save_name is None:
        save_name = 'baseline_A+B'
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('/raid/Data/huangtao/tangzhice/matting/baseline_A/test_out', save_name)

    main(config, save_path, args=args)
