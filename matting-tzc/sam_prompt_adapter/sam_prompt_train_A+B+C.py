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
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append('/raid/Data/huangtao/huangtao/Codes/Others/ADMatting_SAM/SAM_Adapter/data')
from dim_dataset import DataGenerator, ImageFileTrain

# 添加Lung20Dataset路径
sys.path.append('/raid/Data/huangtao/zhangqy/LNMatte/PFM_Net_reply_v1/data')
from lung_dataset import Lung20Dataset

# 设备和local_rank设置（分布式兼容）
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# 内存优化设置
if torch.cuda.is_available():
    # 设置CUDA内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # 清理GPU缓存
    torch.cuda.empty_cache()
    print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


def cleanup_memory():
    """清理显存和释放不需要的变量"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


def release_tensors(*tensors):
    """释放tensor变量"""
    for tensor in tensors:
        if tensor is not None:
            del tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mask_to_trimap_with_transition(mask, erode_kernel=7, dilate_kernel=15):
    """
    将二值mask转换为带过渡区域的trimap
    
    Args:
        mask: 二值mask (0背景, 1前景) [B, C, H, W] 或 [H, W]
        erode_kernel: 腐蚀核大小，用于生成确定前景
        dilate_kernel: 膨胀核大小，用于生成确定背景
    
    Returns:
        trimap: 三值图 (0背景, 0.5未知, 1前景)
    """
    original_shape = mask.shape
    is_tensor = isinstance(mask, torch.Tensor)
    device_type = mask.device if is_tensor else None
    
    # 处理batch维度
    if len(original_shape) == 4:  # [B, C, H, W]
        batch_size = original_shape[0]
        trimaps = []
        
        for i in range(batch_size):
            single_mask = mask[i, 0] if original_shape[1] > 0 else mask[i]
            single_trimap = mask_to_trimap_with_transition(single_mask, erode_kernel, dilate_kernel)
            trimaps.append(single_trimap.unsqueeze(0).unsqueeze(0))  # [1, 1, H, W]
        
        return torch.cat(trimaps, dim=0)  # [B, 1, H, W]
    
    # 单个mask处理 [C, H, W] 或 [H, W]
    if len(original_shape) == 3:
        mask_2d = mask[0]  # 取第一个通道
    else:
        mask_2d = mask
    
    # 转换为numpy处理
    if is_tensor:
        mask_np = mask_2d.cpu().numpy().astype(np.uint8)
    else:
        mask_np = mask_2d.astype(np.uint8)
    
    # 生成确定前景（腐蚀原mask）
    erode_kernel_np = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel, erode_kernel))
    fg_sure = cv2.erode(mask_np, erode_kernel_np, iterations=1)
    
    # 生成确定背景（膨胀原mask取反）
    dilate_kernel_np = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    mask_dilated = cv2.dilate(mask_np, dilate_kernel_np, iterations=1)
    bg_sure = 1 - mask_dilated
    
    # 生成trimap
    trimap = np.full_like(mask_np, 0.5, dtype=np.float32)  # 默认未知区域
    trimap[fg_sure == 1] = 1.0  # 确定前景
    trimap[bg_sure == 1] = 0.0  # 确定背景
    
    # 转换回原始格式
    if is_tensor:
        result = torch.from_numpy(trimap).to(device_type)
        if len(original_shape) == 3:
            result = result.unsqueeze(0)  # 添加通道维度
        return result
    else:
        return trimap


class EnhancedMattingLoss(torch.nn.Module):
    """增强的抠图损失，结合known loss和DDC loss"""
    
    def __init__(self, known_weight=1.0, ddc_weight=0.1, consistency_kernel=11):
        super().__init__()
        self.known_weight = known_weight
        self.ddc_weight = ddc_weight
        self.consistency_kernel = consistency_kernel
    
    def forward(self, pred_alpha, gt_trimap, input_image, sample_map=None):
        """
        Args:
            pred_alpha: 预测的alpha [B, 1, H, W]
            gt_trimap: 生成的trimap [B, 1, H, W] (0, 0.5, 1)
            input_image: 输入图像 [B, 3, H, W]
            sample_map: 未知区域mask [B, 1, H, W] (trimap==0.5的区域)
        """
        losses = {}
        total_loss = 0.0
        
        # 1. Known Loss - 在已知区域计算L1损失
        known_mask = (gt_trimap != 0.5).float()  # 前景和背景区域
        if known_mask.sum() > 0:
            known_loss = F.l1_loss(pred_alpha * known_mask, gt_trimap * known_mask)
            losses['known_l1_loss'] = known_loss * self.known_weight
            total_loss += losses['known_l1_loss']
        else:
            losses['known_l1_loss'] = torch.tensor(0.0, device=pred_alpha.device)
        
        # 2. DDC Loss - 在未知区域计算深度细节一致性损失
        if sample_map is None:
            sample_map = (gt_trimap == 0.5).float()  # 未知区域
        
        if sample_map.sum() > 0:
            # 计算图像和alpha的一致性
            image_dist, alpha_dist = self.compute_consistency(input_image, pred_alpha, self.consistency_kernel)
            
            # 只在未知区域计算DDC loss
            ddc_loss = F.l1_loss(
                image_dist * sample_map, 
                alpha_dist * sample_map
            )
            losses['ddc_loss'] = ddc_loss * self.ddc_weight
            total_loss += losses['ddc_loss']
        else:
            losses['ddc_loss'] = torch.tensor(0.0, device=pred_alpha.device)
        
        # 返回与原始损失函数相同的格式
        losses['total_loss'] = total_loss
        
        return total_loss, losses
    
    def compute_consistency(self, image, alpha, kernel_size=11):
        """计算DDC一致性（借鉴detail_capture.py的实现）"""
        b, c, h, w = image.shape
        
        # 图像归一化处理
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image_norm = image * std + mean
        
        # 展开邻域
        unfold_image = F.unfold(image_norm, kernel_size=kernel_size, padding=kernel_size // 2).view(b, c, kernel_size ** 2, h, w)
        unfold_alpha = F.unfold(alpha, kernel_size=kernel_size, padding=kernel_size // 2).view(b, kernel_size ** 2, h, w)
        
        # 计算图像距离
        image_dist = torch.norm(image_norm.view(b, c, 1, h, w) - unfold_image, 2, dim=1)
        image_dist, indices = torch.topk(image_dist, k=kernel_size, dim=1, largest=False)
        
        # 计算对应的alpha距离
        center_alpha = alpha.view(b, 1, h, w)
        alpha_dist = torch.gather(center_alpha - unfold_alpha, dim=1, index=indices)
        
        return image_dist, alpha_dist


class MedicalDatasetWithDDC(Dataset):
    """
    医学图像数据集，支持从image和mask生成trimap，用于DDC训练
    """
    
    def __init__(self, image_dir, mask_dir, image_size=256, 
                 erode_kernel=7, dilate_kernel=15, use_ddc=True):
        """
        Args:
            image_dir: 图像目录路径
            mask_dir: mask目录路径
            image_size: 输出图像尺寸
            erode_kernel: trimap生成的腐蚀核大小
            dilate_kernel: trimap生成的膨胀核大小
            use_ddc: 是否使用DDC训练模式
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.erode_kernel = erode_kernel
        self.dilate_kernel = dilate_kernel
        self.use_ddc = use_ddc
        
        # 获取所有图像文件
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        
        # 验证对应的mask文件存在
        self.valid_samples = []
        for img_path in image_files:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # 尝试不同的mask文件扩展名
            mask_path = None
            for mask_ext in ['.png', '.jpg', '.jpeg']:
                potential_mask = os.path.join(mask_dir, img_name + mask_ext)
                if os.path.exists(potential_mask):
                    mask_path = potential_mask
                    break
            
            if mask_path:
                self.valid_samples.append((img_path, mask_path))
            else:
                print(f"Warning: No mask found for {img_path}")
        
        print(f"MedicalDatasetWithDDC: Found {len(self.valid_samples)} valid samples")
        print(f"Image dir: {image_dir}")
        print(f"Mask dir: {mask_dir}")
        print(f"DDC mode: {use_ddc}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_samples[idx]
        
        # 加载图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # [C, H, W]
        
        # 加载mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = (mask > 127).astype(np.float32)  # 二值化
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        
        # 生成alpha（从mask，这里假设mask就是ground truth alpha的二值版本）
        alpha = mask.clone()
        
        sample = {
            'image': image,
            'alpha': alpha,
            'mask': mask,
            'name': os.path.basename(img_path)
        }
        
        # 如果启用DDC模式，生成trimap
        if self.use_ddc:
            trimap = mask_to_trimap_with_transition(
                mask, self.erode_kernel, self.dilate_kernel
            )
            sample['trimap'] = trimap
            sample['ddc_mode'] = True
        else:
            # 兼容原有模式，直接使用mask
            sample['ddc_mode'] = False
        
        return sample


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
    
    def collect_sample(self, inp, gt, pred, batch_idx, natural_batch_size=59, medical_batch_size=5, ddc_batch_size=3):
        """
        收集可视化样本 - 三路混合训练版本
        同时保存自然图像、医学图像(原有)和医学图像(DDC)样本
        
        Args:
            inp: 输入图像 [B, C, H, W]
            gt: 真实标签 [B, C, H, W] 
            pred: 预测结果 [B, C, H, W]
            batch_idx: batch索引
            natural_batch_size: 自然图像的batch size
            medical_batch_size: 医学图像(原有)的batch size
            ddc_batch_size: 医学图像(DDC)的batch size
        """
        # 限制总样本数量，保存三种类型的图像
        max_triplets = self.max_samples // 3  # 每组包含1个自然+1个医学(原有)+1个医学(DDC)
        
        if len(self.vis_samples) < self.max_samples and batch_idx < max_triplets:
            samples_added = 0
            
            # 1. 保存自然图像样本（第一个）
            nat_sample = {
                'input': inp[0].cpu().detach(),  # 自然图像（索引0）
                'gt': gt[0].cpu().detach(),
                'pred': pred[0].cpu().detach() if pred is not None else None,
                'batch_idx': batch_idx,
                'image_type': 'natural'
            }
            self.vis_samples.append(nat_sample)
            samples_added += 1
            
            # 2. 保存医学图像(原有)样本
            medical_end = natural_batch_size + medical_batch_size
            if inp.size(0) > natural_batch_size:  # 确保有医学图像(原有)
                med_sample = {
                    'input': inp[natural_batch_size].cpu().detach(),  # 医学图像(原有)（第一个医学样本）
                    'gt': gt[natural_batch_size].cpu().detach(),
                    'pred': pred[natural_batch_size].cpu().detach() if pred is not None else None,
                    'batch_idx': batch_idx,
                    'image_type': 'medical_precise'
                }
                self.vis_samples.append(med_sample)
                samples_added += 1
            
            # 3. 保存医学图像(DDC)样本
            ddc_start = medical_end
            ddc_end = ddc_start + ddc_batch_size
            if inp.size(0) > ddc_start:  # 确保有医学图像(DDC)
                ddc_sample = {
                    'input': inp[ddc_start].cpu().detach(),  # 医学图像(DDC)（第一个DDC样本）
                    'gt': gt[ddc_start].cpu().detach(),
                    'pred': pred[ddc_start].cpu().detach() if pred is not None else None,
                    'batch_idx': batch_idx,
                    'image_type': 'medical_ddc'
                }
                self.vis_samples.append(ddc_sample)
                samples_added += 1
                
            return samples_added > 0
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
        if img_type == 'natural':
            img_type_display = '自然图像'
        elif img_type == 'medical_precise':
            img_type_display = '医学图像(精确标签)'
        elif img_type == 'medical_ddc':
            img_type_display = '医学图像(DDC)'
        else:
            img_type_display = img_type
        
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
    
    def save_debug_sample(self, inp, gt, pred, batch_idx, prefix="debug", natural_batch_size=59, medical_batch_size=5, ddc_batch_size=3):
        """
        立即保存调试样本（不加入收集队列）
        同时保存自然图像、医学图像(原有)和医学图像(DDC)的调试样本
        """
        sample_count = 0
        
        # 1. 保存自然图像调试样本
        if inp is not None and inp.size(0) > 0:
            debug_sample_nat = {
                'input': inp[0].cpu().detach(),
                'gt': gt[0].cpu().detach() if gt is not None else None,
                'pred': pred[0].cpu().detach() if pred is not None else None,
                'batch_idx': batch_idx,
                'image_type': 'natural'
            }
            self._save_single_sample(debug_sample_nat, prefix, sample_count)
            sample_count += 1
        
        # 2. 保存医学图像(原有)调试样本
        medical_end = natural_batch_size + medical_batch_size
        if inp is not None and inp.size(0) > natural_batch_size:
            debug_sample_med = {
                'input': inp[natural_batch_size].cpu().detach(),
                'gt': gt[natural_batch_size].cpu().detach() if gt is not None else None,
                'pred': pred[natural_batch_size].cpu().detach() if pred is not None else None,
                'batch_idx': batch_idx,
                'image_type': 'medical_precise'
            }
            self._save_single_sample(debug_sample_med, prefix, sample_count)
            sample_count += 1
        
        # 3. 保存医学图像(DDC)调试样本
        ddc_start = medical_end
        if inp is not None and inp.size(0) > ddc_start:
            debug_sample_ddc = {
                'input': inp[ddc_start].cpu().detach(),
                'gt': gt[ddc_start].cpu().detach() if gt is not None else None,
                'pred': pred[ddc_start].cpu().detach() if pred is not None else None,
                'batch_idx': batch_idx,
                'image_type': 'medical_ddc'
            }
            self._save_single_sample(debug_sample_ddc, prefix, sample_count)
            sample_count += 1


def trimap_to_mask_prompt(trimap):
    """兼容性函数，调用DataProcessor中的静态方法"""
    return DataProcessor.trimap_to_mask_prompt(trimap)


def create_mixed_data_loaders(config, args):
    """创建三路混合数据加载器：自然图像 + 原有医学图像 + DDC医学图像"""
    mixed_config = config['mixed_training']
    loader_config = config['data_loader']
    
    natural_batch_size = mixed_config['natural_batch_size']
    medical_batch_size = mixed_config['medical_batch_size']
    image_size = mixed_config['image_size']
    
    # DDC配置
    ddc_config = mixed_config.get('ddc_training', {})
    use_ddc = ddc_config.get('enabled', False)
    ddc_batch_size = ddc_config.get('ddc_batch_size', 3)  # DDC医学图像的batch size
    
    print(f"三路混合训练配置:")
    print(f"  - 自然图像: {natural_batch_size}张 (alpha GT监督)")
    print(f"  - 医学图像(原有): {medical_batch_size}张 (alpha GT监督)")
    if use_ddc:
        print(f"  - 医学图像(DDC): {ddc_batch_size}张 (DDC监督)")
    
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
    
    # 2. 创建原有医学图像数据集 (alpha GT监督)
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
    
    print(f"原有医学训练集大小: {len(lung_dataset)}")
    
    # 3. 创建DDC医学图像数据集 (DDC监督)
    ddc_loader = None
    ddc_iter = None
    if use_ddc and 'medical_image_dir' in ddc_config and 'medical_mask_dir' in ddc_config:
        print(f"使用DDC医学数据集: {ddc_config['medical_image_dir']}")
        ddc_dataset = MedicalDatasetWithDDC(
            image_dir=ddc_config['medical_image_dir'],
            mask_dir=ddc_config['medical_mask_dir'],
            image_size=image_size,
            erode_kernel=ddc_config.get('erode_kernel', 7),
            dilate_kernel=ddc_config.get('dilate_kernel', 15),
            use_ddc=True
        )
        ddc_loader = DataLoader(
            ddc_dataset, 
            batch_size=ddc_batch_size, 
            shuffle=True, 
            drop_last=False,
            num_workers=loader_config['num_workers'],
            pin_memory=loader_config['pin_memory']
        )
        ddc_iter = iter(ddc_loader)
        print(f"DDC医学训练集大小: {len(ddc_dataset)}")
        print(f"DDC训练模式: 启用 (腐蚀核={ddc_config.get('erode_kernel', 7)}, 膨胀核={ddc_config.get('dilate_kernel', 15)})")
    
    return natural_loader, lung_loader, lung_iter, ddc_loader, ddc_iter


def train(natural_loader, lung_iter, lung_loader, model, optimizer, save_path=None, epoch=None, ddc_iter=None, ddc_loader=None):
    """三路混合数据训练：自然图像 + 原有医学图像 + DDC医学图像"""
    model.train()
    
    # 启用梯度检查点以节省内存
    if hasattr(model.image_encoder, 'gradient_checkpointing_enable'):
        model.image_encoder.gradient_checkpointing_enable()
        print("梯度检查点已启用")
    
    # 从配置中获取参数
    mixed_config = config['mixed_training']
    natural_batch_size = mixed_config['natural_batch_size']
    medical_batch_size = mixed_config['medical_batch_size']
    vis_frequency = mixed_config['visualization_frequency']
    
    # DDC训练配置
    ddc_config = mixed_config.get('ddc_training', {})
    use_ddc = ddc_config.get('enabled', False) and ddc_iter is not None
    ddc_batch_size = ddc_config.get('ddc_batch_size', 3) if use_ddc else 0
    
    # 初始化DDC损失函数（如果启用）
    enhanced_loss = None
    if use_ddc:
        enhanced_loss = EnhancedMattingLoss(
            known_weight=ddc_config.get('known_weight', 1.0),
            ddc_weight=ddc_config.get('ddc_weight', 0.1),
            consistency_kernel=ddc_config.get('consistency_kernel', 11)
        ).to(device)
        print(f"三路DDC训练已启用:")
        print(f"  - Known weight: {ddc_config.get('known_weight', 1.0)}")
        print(f"  - DDC weight: {ddc_config.get('ddc_weight', 0.1)}")
        print(f"  - DDC batch size: {ddc_batch_size}")

    pbar = tqdm(total=len(natural_loader), leave=False, desc='train')

    loss_list = []
    loss_details = {
        'unknown_l1_loss': [],
        'known_l1_loss': [],
        'loss_gradient_penalty': [],
        'loss_pha_laplacian': [],
        'ddc_loss': [],  # 添加DDC损失记录
        'ddc_known_l1_loss': []  # 添加DDC known损失记录
    }
    
    # 创建可视化管理器
    vis_manager = VisualizationManager(save_path) if save_path else None
    
    for batch_idx, nat_batch in enumerate(natural_loader):
        # 清零梯度
        optimizer.zero_grad()
        
        # 获取原有医学数据（循环使用）
        try:
            lung_batch = next(lung_iter)
        except StopIteration:
            lung_iter = iter(lung_loader)
            lung_batch = next(lung_iter)
        
        # 获取DDC医学数据（循环使用，如果启用）
        ddc_batch = None
        if use_ddc and ddc_iter is not None:
            try:
                ddc_batch = next(ddc_iter)
            except StopIteration:
                ddc_iter = iter(ddc_loader)
                ddc_batch = next(ddc_iter)
        
        # 移动数据到GPU
        nat_batch = DataProcessor.move_batch_to_device(nat_batch, device)
        lung_batch = DataProcessor.move_batch_to_device(lung_batch, device)
        if ddc_batch is not None:
            ddc_batch = DataProcessor.move_batch_to_device(ddc_batch, device)
        
        # 三路拼接batch数据
        if 'image' in nat_batch and 'image' in lung_batch:
            # 基础拼接：自然图像 + 原有医学图像
            img_list = [nat_batch["image"], lung_batch["image"]]
            alpha_list = [nat_batch["alpha"], lung_batch["alpha"]]
            
            # 如果有DDC医学图像，添加到拼接列表
            if ddc_batch is not None and 'image' in ddc_batch:
                img_list.append(ddc_batch["image"])
                alpha_list.append(ddc_batch["alpha"])
                if batch_idx == 0:
                    print(f"[Info] 三路混合训练: 自然({natural_batch_size}) + 医学({medical_batch_size}) + DDC({ddc_batch_size})")
            else:
                if batch_idx == 0:
                    print(f"[Info] 二路混合训练: 自然({natural_batch_size}) + 医学({medical_batch_size})")
            
            # 执行拼接
            img = torch.cat(img_list, dim=0)
            alpha = torch.cat(alpha_list, dim=0)
            
            # 处理trimap - 三路拼接
            trimap_list = []
            if 'trimap' in nat_batch:
                trimap_list.append(nat_batch["trimap"])
            else:
                # 降级处理：如果自然图像没有trimap（不应该发生）
                nat_trimap = torch.ones_like(nat_batch["alpha"]) * 0.5  # 全部设为unknown
                trimap_list.append(nat_trimap)
                if batch_idx == 0:
                    print("[Warning] 自然图像数据缺少trimap，使用降级策略")
            
            if 'trimap' in lung_batch:
                trimap_list.append(lung_batch["trimap"])
            else:
                # 原有医学数据集应该有trimap
                lung_trimap = torch.ones_like(lung_batch["alpha"]) * 0.5
                trimap_list.append(lung_trimap)
                if batch_idx == 0:
                    print("[Warning] 原有医学图像数据缺少trimap")
            
            # DDC医学图像的trimap处理
            if ddc_batch is not None and 'image' in ddc_batch:
                if 'trimap' in ddc_batch:
                    trimap_list.append(ddc_batch["trimap"])
                else:
                    # DDC数据集应该有trimap（从mask生成的）
                    ddc_trimap = torch.ones_like(ddc_batch["alpha"]) * 0.5
                    trimap_list.append(ddc_trimap)
                    if batch_idx == 0:
                        print("[Warning] DDC医学图像数据缺少trimap")
            
            trimap = torch.cat(trimap_list, dim=0)
            
            if batch_idx == 0:  # 只在第一个batch打印信息
                print("[Info] Trimap拼接完成:")
                print(f"  - 自然图像trimap: {nat_batch['trimap'].shape if 'trimap' in nat_batch else 'missing'}")
                print(f"  - 医学图像trimap: {lung_batch['trimap'].shape if 'trimap' in lung_batch else 'missing'}")
                if ddc_batch is not None:
                    print(f"  - DDC图像trimap: {ddc_batch['trimap'].shape if 'trimap' in ddc_batch else 'missing'}")
                
            # 处理mask - 三路拼接
            mask_list = []
            if 'mask' in nat_batch:
                mask_list.append(nat_batch["mask"])
            else:
                # 自然图像应该有mask
                nat_mask = torch.ones_like(nat_batch["alpha"])
                mask_list.append(nat_mask)
                if batch_idx == 0:
                    print("[Warning] 自然图像数据缺少mask")
            
            # 原有医学图像的mask（可能没有，用默认）
            lung_mask = lung_batch.get("mask", torch.ones_like(lung_batch["alpha"]))
            mask_list.append(lung_mask)
            
            # DDC医学图像的mask
            if ddc_batch is not None and 'image' in ddc_batch:
                if 'mask' in ddc_batch:
                    mask_list.append(ddc_batch["mask"])
                else:
                    ddc_mask = torch.ones_like(ddc_batch["alpha"])
                    mask_list.append(ddc_mask)
            
            masks = torch.cat(mask_list, dim=0) if mask_list else None
            
            if batch_idx == 0:
                print("[Info] Mask拼接完成:")
                print(f"  - 自然图像mask: {nat_batch['mask'].shape if 'mask' in nat_batch else 'default'}")
                print(f"  - 医学图像mask: {lung_batch.get('mask', 'default')}")
                if ddc_batch is not None:
                    print(f"  - DDC图像mask: {ddc_batch['mask'].shape if ddc_batch and 'mask' in ddc_batch else 'default'}")
                
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
        
        # 确定数据批次的划分点
        natural_end = natural_batch_size
        medical_end = natural_end + medical_batch_size
        ddc_end = medical_end + (ddc_batch_size if ddc_batch is not None else 0)
        
        # 检查是否有DDC数据
        has_ddc_data = ddc_batch is not None and use_ddc
        
        # 三路混合训练的SAM模型
        if hasattr(model, "module"):
            model.module.set_input(inp, gt, mask_inputs=mask_prompt, sample_map=sample_map)
            model.module.forward()
            pred_alpha = model.module.pred_mask
            
            # 保存完整的预测结果用于可视化（在分段损失计算之前）
            pred_mask_full = pred_alpha.clone()
            
            # 分别计算三种数据的损失
            total_loss = 0.0
            all_losses = {}
            
            # 1. 自然图像 + 原有医学图像：使用原有损失
            if natural_end + medical_batch_size <= inp.shape[0]:
                combined_inp = inp[:medical_end]  # 自然 + 原有医学
                combined_gt = gt[:medical_end]
                combined_mask_prompt = mask_prompt[:medical_end] if mask_prompt is not None else None
                combined_sample_map = sample_map[:medical_end] if sample_map is not None else None
                
                model.module.set_input(combined_inp, combined_gt, 
                                     mask_inputs=combined_mask_prompt, 
                                     sample_map=combined_sample_map)
                result_combined = model.module.optimize_parameters()
                
                if result_combined is not None:
                    loss_combined, losses_combined = result_combined
                    total_loss += loss_combined
                    all_losses.update(losses_combined)
            
            # 2. DDC医学图像：使用DDC损失
            if has_ddc_data and enhanced_loss is not None and medical_end < inp.shape[0]:
                inp_ddc = inp[medical_end:ddc_end]
                gt_ddc = gt[medical_end:ddc_end]
                trimap_ddc = trimap[medical_end:ddc_end]
                
                # 重新设置模型输入（只处理DDC医学图像）
                model.set_input(inp_ddc, gt_ddc, 
                               mask_inputs=mask_prompt[medical_end:ddc_end] if mask_prompt is not None else None,
                               sample_map=sample_map[medical_end:ddc_end] if sample_map is not None else None)
                model.forward()
                pred_ddc = model.pred_mask
                
                # 使用DDC损失
                sample_map_ddc = (trimap_ddc == 0.5).float()
                ddc_loss, ddc_losses = enhanced_loss(pred_ddc, trimap_ddc, inp_ddc, sample_map_ddc)
                
                total_loss += ddc_loss
                # 为DDC损失添加前缀以区分
                for k, v in ddc_losses.items():
                    if k != 'total_loss':
                        all_losses[f'ddc_{k}'] = v
                
                if batch_idx == 0:
                    print(f"[Info] DDC损失计算完成: {ddc_loss.item():.4f}")
            
            result = (total_loss, all_losses) if all_losses else None
            
        else:
            model.set_input(inp, gt, mask_inputs=mask_prompt, sample_map=sample_map)
            model.forward()
            pred_alpha = model.pred_mask
            
            # 保存完整的预测结果用于可视化（在分段损失计算之前）
            pred_mask_full = pred_alpha.clone()
            
            # 分别计算三种数据的损失
            total_loss = 0.0
            all_losses = {}
            
            # 1. 自然图像 + 原有医学图像：使用原有损失
            if natural_end + medical_batch_size <= inp.shape[0]:
                combined_inp = inp[:medical_end]  # 自然 + 原有医学
                combined_gt = gt[:medical_end]
                combined_mask_prompt = mask_prompt[:medical_end] if mask_prompt is not None else None
                combined_sample_map = sample_map[:medical_end] if sample_map is not None else None
                
                # 重新设置模型输入（只处理自然+原有医学图像）
                model.set_input(combined_inp, combined_gt, 
                               mask_inputs=combined_mask_prompt, 
                               sample_map=combined_sample_map)
                model.forward()  # 重新前向传播
                result_combined = model.optimize_parameters()
                
                if result_combined is not None:
                    loss_combined, losses_combined = result_combined
                    total_loss += loss_combined
                    all_losses.update(losses_combined)
            
            # 2. DDC医学图像：使用DDC损失
            if has_ddc_data and enhanced_loss is not None and medical_end < inp.shape[0]:
                inp_ddc = inp[medical_end:ddc_end]
                gt_ddc = gt[medical_end:ddc_end]
                trimap_ddc = trimap[medical_end:ddc_end]
                
                # 重新设置模型输入（只处理DDC医学图像）
                model.set_input(inp_ddc, gt_ddc, 
                               mask_inputs=mask_prompt[medical_end:ddc_end] if mask_prompt is not None else None,
                               sample_map=sample_map[medical_end:ddc_end] if sample_map is not None else None)
                model.forward()
                pred_ddc = model.pred_mask
                
                # 使用DDC损失
                sample_map_ddc = (trimap_ddc == 0.5).float()
                ddc_loss, ddc_losses = enhanced_loss(pred_ddc, trimap_ddc, inp_ddc, sample_map_ddc)
                
                total_loss += ddc_loss
                # 为DDC损失添加前缀以区分
                for k, v in ddc_losses.items():
                    if k != 'total_loss':
                        all_losses[f'ddc_{k}'] = v
                
                if batch_idx == 0:
                    print(f"[Info] DDC损失计算完成: {ddc_loss.item():.4f}")
            
            result = (total_loss, all_losses) if all_losses else None
        
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
            
            # 收集可视化样本（在DDC损失计算之前，确保pred_mask是完整的batch预测）
            if vis_manager and batch_idx % vis_frequency == 0:
                # 使用保存的完整预测结果
                collected = vis_manager.collect_sample(inp, gt, pred_mask_full, batch_idx, natural_batch_size, medical_batch_size, ddc_batch_size)
                
                # 立即保存调试样本（第一个样本）
                if collected and len(vis_manager.vis_samples) <= 9:  # 现在可能收集更多样本（自然+医学+DDC）
                    vis_manager.save_debug_sample(inp, gt, pred_mask_full, batch_idx, f"batch_{batch_idx}", natural_batch_size, medical_batch_size, ddc_batch_size)
            
            # 执行优化器步骤
            optimizer.step()
            
            # 更频繁的显存清理
            if batch_idx % 5 == 0:
                cleanup_memory()
            
            # 释放不需要的变量
            del combined_inp, combined_gt, combined_mask_prompt, combined_sample_map
            if ddc_config.get('enabled', False):
                del inp_ddc, gt_ddc, trimap_ddc, pred_ddc, sample_map_ddc
        
        pbar.update(1)
        
        # 更新进度条显示详细信息
        if len(loss_list) > 0:
            current_loss = loss_list[-1]
            postfix_dict = {
                'loss': f'{current_loss:.4f}',
                'avg_loss': f'{sum(loss_list)/len(loss_list):.4f}',
                'nat': natural_batch_size,
                'med': medical_batch_size
            }
            if has_ddc_data:
                postfix_dict['ddc'] = ddc_batch_size
            pbar.set_postfix(postfix_dict)

    pbar.close()

    if len(loss_list) == 0:
        return 0.0, lung_iter, ddc_iter  # 返回更新后的迭代器
    
    # 计算平均损失
    avg_loss = sum(loss_list) / len(loss_list)
    
    # 计算详细损失的平均值
    avg_loss_details = {}
    for loss_name, loss_values in loss_details.items():
        if loss_values:
            avg_loss_details[loss_name] = sum(loss_values) / len(loss_values)
    
    # 打印训练统计信息
    print(f"\n=== 三路混合数据训练统计 ===")
    train_summary = f"总批次数: {len(natural_loader)}, 每批次: {natural_batch_size}自然图像 + {medical_batch_size}医学图像"
    if has_ddc_data:
        train_summary += f" + {ddc_batch_size}DDC医学图像"
    print(train_summary)
    
    if use_ddc:
        print(f"DDC训练模式: 启用")
    print(f"平均总损失: {avg_loss:.4f}")
    
    # 分类显示损失
    standard_losses = []
    ddc_losses = []
    for loss_name, avg_value in avg_loss_details.items():
        if avg_value > 0:
            if loss_name.startswith('ddc_'):
                ddc_losses.append((loss_name, avg_value))
            else:
                standard_losses.append((loss_name, avg_value))
    
    if standard_losses:
        print("标准损失:")
        for loss_name, avg_value in standard_losses:
            print(f"  {loss_name}: {avg_value:.4f}")
    
    if ddc_losses:
        print("DDC损失:")
        for loss_name, avg_value in ddc_losses:
            print(f"  {loss_name}: {avg_value:.4f}")
    
    # 保存可视化结果
    if vis_manager and epoch is not None:
        vis_manager.save_samples(epoch)
        vis_manager.clear_samples()  # 清空样本以释放内存
    
    # 清理内存
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return avg_loss, lung_iter, ddc_iter




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

    # 创建三路混合数据加载器 
    natural_loader, lung_loader, lung_iter, ddc_loader, ddc_iter = create_mixed_data_loaders(config, args)
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
        # 使用三路混合数据训练
        train_loss, lung_iter, ddc_iter = train(natural_loader, lung_iter, lung_loader, model, optimizer, 
                                               save_path, epoch, ddc_iter, ddc_loader)
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


# 旧的save_visualization_samples函数已被VisualizationManager类替代
# 保留注释版本以供参考
# def save_visualization_samples(vis_samples, save_path, epoch):
#     """保存可视化样本 - 已被VisualizationManager替代"""
#     pass


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


def test_ddc_functionality():
    """测试DDC功能的辅助函数"""
    print("\n=== DDC功能测试 ===")
    
    # 测试mask到trimap转换
    test_mask = torch.ones((1, 1, 64, 64))  # 简单的全前景mask
    test_mask[:, :, 20:44, 20:44] = 1  # 前景区域
    test_mask[:, :, :20, :] = 0  # 背景区域
    test_mask[:, :, 44:, :] = 0  # 背景区域
    test_mask[:, :, :, :20] = 0  # 背景区域
    test_mask[:, :, :, 44:] = 0  # 背景区域
    
    trimap = mask_to_trimap_with_transition(test_mask, erode_kernel=5, dilate_kernel=9)
    
    unique_values = torch.unique(trimap)
    print(f"Trimap unique values: {unique_values.tolist()}")
    print(f"Expected: [0.0, 0.5, 1.0]")
    
    # 测试DDC Loss
    enhanced_loss = EnhancedMattingLoss(known_weight=1.0, ddc_weight=0.1)
    test_pred = torch.rand((1, 1, 64, 64))
    test_image = torch.rand((1, 3, 64, 64))
    
    total_loss, losses = enhanced_loss(test_pred, trimap, test_image)
    print(f"DDC Loss components: {list(losses.keys())}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    # 测试数据集（如果路径存在）
    try:
        test_dataset = MedicalDatasetWithDDC(
            image_dir="/raid/Data/huangtao/public/LNSM/train/image",
            mask_dir="/raid/Data/huangtao/public/LNSM/train/binarymask",
            image_size=256,
            use_ddc=True
        )
        print(f"Medical dataset loaded: {len(test_dataset)} samples")
        if len(test_dataset) > 0:
            sample = test_dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"DDC mode: {sample.get('ddc_mode', False)}")
    except Exception as e:
        print(f"Medical dataset test failed: {e}")
    
    print("=== DDC功能测试完成 ===\n")


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
    parser.add_argument('--test_ddc', action='store_true', help='运行DDC功能测试')
    args = parser.parse_args()
    
    # 运行DDC功能测试（如果请求）
    if args.test_ddc:
        test_ddc_functionality()

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
            'natural_batch_size': 40,  # 每个batch中自然图像的数量（进一步增加以更好利用GPU）
            'medical_batch_size': 5,   # 每个batch中医学图像的数量（进一步增加以更好利用GPU）
            'image_size': 256,         # 输入图像尺寸
            'visualization_frequency': 100,  # 每多少个batch保存一次可视化
            'checkpoint_frequency': 10,      # 每多少个epoch保存一次检查点
            'ddc_training': {
                'enabled': True,  # 启用DDC训练模式
                'medical_image_dir': '/raid/Data/huangtao/public/LNSM/train/image',
                'medical_mask_dir': '/raid/Data/huangtao/public/LNSM/train/binarymask',
                'ddc_batch_size': 5,      # DDC医学图像的batch size（进一步增加以更好利用GPU）
                'known_weight': 1.0,      # known区域L1损失权重
                'ddc_weight': 0.1,        # DDC损失权重
                'consistency_kernel': 11,  # DDC一致性计算的邻域大小
                'erode_kernel': 7,        # trimap生成的腐蚀核大小
                'dilate_kernel': 15,      # trimap生成的膨胀核大小
            }
        },
        'data_loader': {
            'num_workers': 2,          # DataLoader的工作进程数（减少以避免阻塞）
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
        save_name = 'baseline_A+B+C'
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('/raid/Data/huangtao/tangzhice/matting/baseline_A/test_out', save_name)

    main(config, save_path, args=args)
