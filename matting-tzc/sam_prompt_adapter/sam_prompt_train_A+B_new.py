import argparse
import os
import sys

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

# 抑制matplotlib字体警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 这些导入现在通过data_processor模块处理
# 保留current_dir变量，可能在其他地方使用
current_dir = os.path.dirname(os.path.abspath(__file__))

# 导入数据处理模块
sys.path.append('/raid/Data/huangtao/tangzhice/matting/baseline_A/data')
from data_processor import DataProcessor, create_mixed_data_loaders

# 设备设置（使用base.py配置）
sys.path.append('/raid/Data/huangtao/tangzhice/matting/baseline_A')
from base.base import setup_devices
setup_devices()  # 设置使用GPU 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_training():
    # 导入你的模型
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
    
    model = SAMAdapterPrompt().cuda() # 创建SAM适配器模型实例并移动到GPU
    
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
        img_type_display = 'Natural' if img_type == 'natural' else 'Medical' if img_type == 'medical' else img_type.title()
        
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
        
        # 使用优化后的batch数据合并方法
        try:
            img, alpha, trimap, masks, _ = DataProcessor.merge_batch_data(
                nat_batch, lung_batch, batch_idx, debug_log=True
            )
        except ValueError as e:
            print(f"[Error] 数据合并失败: {e}")
            continue
        
        # 使用统一的trimap处理方法
        trimap = DataProcessor.normalize_trimap(trimap)
        sample_map = DataProcessor.create_sample_map(trimap)
        
        # 准备输入数据（SAM格式）
        inp = img[:, :3] if img.shape[1] > 3 else img  # 确保只有RGB通道
        gt = alpha
        
        # 准备mask提示
        mask_prompt = DataProcessor.prepare_mask_prompt(masks, trimap, batch_idx, debug_log=True)
        
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
    print(f"\n=== Mixed Training Statistics ===")
    print(f"Total batches: {len(natural_loader)}, per batch: {natural_batch_size} natural + {medical_batch_size} medical images")
    print(f"Average total loss: {avg_loss:.4f}")
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
    
    setup_devices()  # 设置GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Single process training")

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # 创建混合数据加载器 
    natural_loader, lung_loader, lung_iter = create_mixed_data_loaders(config, args)
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    
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
        
        save_file = os.path.join(save_path, f"checkpoint_{name}.pth")
        print(f"Saving checkpoint to: {save_file}")
        torch.save(checkpoint, save_file)
        print(f"Successfully saved checkpoint: {save_file}")
            
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        import traceback
        traceback.print_exc()


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
        save_name = 'baseline_A+B111'
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('/raid/Data/huangtao/tangzhice/matting/baseline_A/test_out', save_name)

    main(config, save_path, args=args)
