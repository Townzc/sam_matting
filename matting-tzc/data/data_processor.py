"""
数据处理和加载模块
包含DataProcessor类和混合数据加载器创建功能
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# 添加依赖路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append('/raid/Data/huangtao/tangzhice/matting/baseline_A/data/dim_dataset.py')
from dim_dataset import DataGenerator, ImageFileTrain

# 添加Lung20Dataset路径
sys.path.append('/raid/Data/huangtao/zhangqy/LNMatte/PFM_Net_reply_v1/data')
from lung_dataset_fixed import Lung20DatasetFixed


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
        return mask_prompt

    @staticmethod
    def extract_trimap_from_batch(batch, fallback_alpha=None):
        """
        从batch中提取trimap数据，如果不存在则生成fallback trimap
        
        Args:
            batch: 数据批次
            fallback_alpha: 用于生成fallback trimap的alpha数据
        
        Returns:
            trimap tensor 或 None
        """
        if 'trimap' in batch:
            return batch['trimap']
        elif fallback_alpha is not None:
            # 生成全unknown的trimap作为fallback
            return torch.ones_like(fallback_alpha) * 0.5
        else:
            return None
    
    @staticmethod 
    def merge_batch_data(nat_batch, lung_batch, batch_idx=0, debug_log=True):
        """
        合并自然图像和医学图像的batch数据
        
        Args:
            nat_batch: 自然图像batch
            lung_batch: 医学图像batch  
            batch_idx: batch索引，用于控制日志打印
            debug_log: 是否打印调试信息
            
        Returns:
            tuple: (img, alpha, trimap, masks, log_messages)
        """
        log_messages = []
        
        # 1. 提取图像和alpha数据
        try:
            nat_img = DataProcessor.extract_image_from_batch(nat_batch)
            lung_img = DataProcessor.extract_image_from_batch(lung_batch)
            
            nat_alpha = DataProcessor.extract_alpha_from_batch(nat_batch)
            lung_alpha = DataProcessor.extract_alpha_from_batch(lung_batch)
            
            # 合并图像和alpha
            img = torch.cat([nat_img, lung_img], dim=0)
            alpha = torch.cat([nat_alpha, lung_alpha], dim=0)
            
        except ValueError as e:
            raise ValueError(f"数据格式不匹配: {str(e)}")
        
        # 2. 处理trimap数据
        nat_trimap = DataProcessor.extract_trimap_from_batch(nat_batch, nat_alpha)
        lung_trimap = DataProcessor.extract_trimap_from_batch(lung_batch, lung_alpha)
        
        if nat_trimap is not None and lung_trimap is not None:
            trimap = torch.cat([nat_trimap, lung_trimap], dim=0)
            if nat_trimap is nat_batch.get('trimap'):
                log_messages.append("[Info] 使用自然图像自动生成的trimap + 医学图像精确trimap")
            else:
                log_messages.append("[Warning] 自然图像数据缺少trimap，使用fallback策略")
        else:
            raise ValueError("无法获取有效的trimap数据")
        
        # 3. 处理mask数据
        nat_mask = DataProcessor.extract_mask_from_batch(nat_batch)
        lung_mask = DataProcessor.extract_mask_from_batch(lung_batch)
        
        if nat_mask is not None:
            # 为医学图像创建默认mask（如果没有的话）
            if lung_mask is None:
                lung_mask = torch.ones_like(lung_alpha)
            masks = torch.cat([nat_mask, lung_mask], dim=0)
            log_messages.append("[Info] 使用自然图像自动生成的mask + 医学图像mask")
        else:
            masks = None
            log_messages.append("[Warning] 未找到自然图像的mask数据")
        
        # 4. 打印调试信息（仅在第一个batch）
        if debug_log and batch_idx == 0:
            for msg in log_messages:
                print(msg)
        
        return img, alpha, trimap, masks, log_messages
    
    @staticmethod
    def prepare_mask_prompt(masks, trimap, batch_idx=0, debug_log=True):
        """
        准备SAM的mask prompt输入
        
        Args:
            masks: 提取到的mask数据（可能为None）
            trimap: trimap数据
            batch_idx: batch索引
            debug_log: 是否打印调试信息
            
        Returns:
            mask_prompt tensor
        """
        if masks is not None:
            # 使用自动生成的精确mask（推荐方式）
            mask_prompt = masks.float()
            if debug_log and batch_idx == 0:
                print("[Info] 使用自动生成的精确mask作为SAM引导")
        else:
            # 降级处理：从trimap转换为mask提示
            mask_prompt = DataProcessor.trimap_to_mask_prompt(trimap)
            if debug_log and batch_idx == 0:
                print("[Warning] 使用trimap转换的mask作为SAM引导")
                
        return mask_prompt


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
    
    # 2. 创建肺部医学图像数据集
    lung_dataset = Lung20DatasetFixed(args.root_lung20)
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


# 便于其他模块导入的接口
__all__ = ['DataProcessor', 'create_mixed_data_loaders']