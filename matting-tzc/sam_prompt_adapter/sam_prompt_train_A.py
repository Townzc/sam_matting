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
            for k, v in batch.items():
                batch[k] = v.cuda()

            # DataGenerator返回的键名可能是'image'而不是'inp'
            if 'inp' in batch:
                inp = batch['inp']
            elif 'image' in batch:
                inp = batch['image']
            else:
                # 如果没有找到图像，尝试合成
                if 'fg' in batch and 'bg' in batch and 'alpha' in batch:
                    fg = batch['fg']
                    bg = batch['bg'] 
                    alpha = batch['alpha']
                    inp = fg * alpha + bg * (1 - alpha)
                else:
                    raise ValueError("No valid image found in batch")
            
            # 获取gt alpha
            if 'gt' in batch:
                gt = batch['gt']
            elif 'alpha' in batch:
                gt = batch['alpha']
            else:
                raise ValueError("No valid alpha/gt found in batch")
            
            # 前向传播
            if hasattr(model, "module"):
                model.module.set_input(inp, gt, mask_inputs=batch.get('mask', None), sample_map=batch.get('sample_map', None))
                model.module.forward()
                pred = model.module.pred_mask
            else:
                model.set_input(inp, gt, mask_inputs=batch.get('mask', None), sample_map=batch.get('sample_map', None))
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
    
    if adapter_params:
        param_groups.append({
            'params': adapter_params,
            'lr': base_lr * 5.0,
            'weight_decay': weight_decay,
        })
        print(f"Adapter: {len(adapter_params)} params, lr={base_lr * 5.0}")
    
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


def train(train_loader, model, optimizer, save_path=None, epoch=None):
    model.train()

    pbar = tqdm(total=len(train_loader), leave=False, desc='train')

    loss_list = []
    loss_details = {
        'unknown_l1_loss': [],
        'known_l1_loss': [],
        'loss_gradient_penalty': [],
        'loss_pha_laplacian': []
    }
    
    # 用于可视化的样本
    vis_samples = []
    
    for batch_idx, batch in enumerate(train_loader):
        # 清零梯度
        optimizer.zero_grad()
        
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        # DataGenerator返回的键名可能是'image'而不是'inp'
        if 'inp' in batch:
            inp = batch['inp']
        elif 'image' in batch:
            inp = batch['image']
        else:
            # 如果没有找到图像，尝试合成
            if 'fg' in batch and 'bg' in batch and 'alpha' in batch:
                # 合成图像
                fg = batch['fg']
                bg = batch['bg'] 
                alpha = batch['alpha']
                inp = fg * alpha + bg * (1 - alpha)
            else:
                raise ValueError("No valid image found in batch")
        
        # 获取gt alpha
        if 'gt' in batch:
            gt = batch['gt']
        elif 'alpha' in batch:
            gt = batch['alpha']
        else:
            raise ValueError("No valid alpha/gt found in batch")
        
        # sample_map可能不存在，设为None
        sample_map = batch.get('sample_map', None)
        
        if hasattr(model, "module"):
            model.module.set_input(inp, gt, mask_inputs=batch.get('mask', None), sample_map=sample_map)
            model.module.forward()  # 先进行前向传播
            result = model.module.optimize_parameters()
        else:
            model.set_input(inp, gt, mask_inputs=batch.get('mask', None), sample_map=sample_map)
            model.forward()  # 先进行前向传播
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
            
            # 收集可视化样本（每100个batch保存一个）
            if batch_idx % 100 == 0 and len(vis_samples) < 3:
                pred_mask = model.module.pred_mask if hasattr(model, "module") else model.pred_mask
                vis_samples.append({
                    'input': inp[0].cpu().detach(),  # 取第一个样本
                    'gt': gt[0].cpu().detach(),
                    'pred': pred_mask[0].cpu().detach() if pred_mask is not None else None,
                    'batch_idx': batch_idx
                })
            # 立即保存可视化样本（调试用）
                if len(vis_samples) == 1:  # 只保存第一个样本
                    temp_vis_samples = vis_samples.copy()
                    save_visualization_samples(temp_vis_samples, save_path, f"batch_{batch_idx}")
            # 执行优化器步骤
            optimizer.step()
        
        pbar.update(1)
        
        # 更新进度条显示详细信息
        if len(loss_list) > 0:
            current_loss = loss_list[-1]
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'avg_loss': f'{sum(loss_list)/len(loss_list):.4f}'
            })

    pbar.close()

    if len(loss_list) == 0:
        return 0.0, []  # 如果没有有效的损失值，返回0
    
    # 计算平均损失
    avg_loss = sum(loss_list) / len(loss_list)
    
    # 计算详细损失的平均值
    avg_loss_details = {}
    for loss_name, loss_values in loss_details.items():
        if loss_values:
            avg_loss_details[loss_name] = sum(loss_values) / len(loss_values)
    
    # 打印训练统计信息
    print(f"\n=== Training Statistics ===")
    print(f"Average Total Loss: {avg_loss:.4f}")
    for loss_name, avg_value in avg_loss_details.items():
        print(f"  {loss_name}: {avg_value:.4f}")
    
    # 保存可视化结果
    if vis_samples and save_path is not None and epoch is not None:
        save_visualization_samples(vis_samples, save_path, epoch)
    
    # 清理内存
    del vis_samples
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return avg_loss, []


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

    # 创建训练数据加载器
    train_loader = make_data_loader(config['train_dataset'], 'train', distributed=('WORLD_SIZE' in os.environ))
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
    
    for epoch in range(epoch_start, epoch_max + 1):
        train_loss, vis_samples = train(train_loader, model, optimizer, save_path, epoch)
        lr_scheduler.step()

        print(f'epoch {epoch}/{epoch_max}')
        print(f'train loss: {train_loss:.4f}')

        # 保存检查点
        print(f"Attempting to save 'last' checkpoint...")
        save(config, model, save_path, 'last')
        
        # 每10个epoch保存一次
        if epoch % 10 == 0:
            print(f"Attempting to save epoch {epoch} checkpoint...")
            save(config, model, save_path, f'epoch_{epoch}')


def save_visualization_samples(vis_samples, save_path, epoch):
    """保存可视化样本"""
    vis_dir = os.path.join(save_path, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, sample in enumerate(vis_samples):
        # 创建图像网格
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 输入图像 (RGB)
        input_img = sample['input'].permute(1, 2, 0).numpy()
        
        # 检查并处理无效值
        if np.any(np.isnan(input_img)) or np.any(np.isinf(input_img)):
            print(f"Warning: Input image contains NaN or Inf values, replacing with 0")
            input_img = np.nan_to_num(input_img, nan=0.0, posinf=1.0, neginf=0.0)
        
        input_img = np.clip(input_img, 0, 1)
        
        # 打印调试信息
        print(f"Input image stats - min: {input_img.min():.4f}, max: {input_img.max():.4f}, mean: {input_img.mean():.4f}")
        print(f"Input image shape: {input_img.shape}, dtype: {input_img.dtype}")
        
        axes[0].imshow(input_img)
        axes[0].set_title('Input Image')
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
            axes[2].text(0.5, 0.5, 'No Prediction', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].axis('off')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'epoch_{epoch}_sample_{i+1}_batch_{sample["batch_idx"]}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存单独的alpha matte图像
        if sample['pred'] is not None:
            pred_alpha = sample['pred'].squeeze().numpy()
            pred_alpha = (pred_alpha * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(vis_dir, f'epoch_{epoch}_sample_{i+1}_alpha_pred.png'), pred_alpha)
        
        gt_alpha = (gt_alpha * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(vis_dir, f'epoch_{epoch}_sample_{i+1}_alpha_gt.png'), gt_alpha)
        
        # 保存输入图像
        input_img_save = (input_img * 255).astype(np.uint8)
        # 确保没有无效值
        input_img_save = np.clip(input_img_save, 0, 255)
        cv2.imwrite(os.path.join(vis_dir, f'epoch_{epoch}_sample_{i+1}_input.png'), cv2.cvtColor(input_img_save, cv2.COLOR_RGB2BGR))
    
    print(f"Saved {len(vis_samples)} visualization samples to {vis_dir}")


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
    
    dataloader_args = {
        'batch_size': dataset_config['batch_size'],
        'shuffle': (tag == 'train') and not distributed,
        'num_workers': 4,
        'pin_memory': True
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
    args = parser.parse_args()

    # 默认配置
    config = {
        'train_dataset': {
            'alpha_dir': "/raid/Data/huangtao/public/matting/vitmatte/alpha",
            'fg_dir': "/raid/Data/huangtao/public/matting/vitmatte/fg", 
            'bg_dir': "/raid/Data/huangtao/public/matting/vitmatte/bg",
            'batch_size': 2,  # 双卡训练，可以适当增大batch_size
            'size': 256
        },
        'val_dataset': {
            'alpha_dir': "/raid/Data/huangtao/public/matting/vitmatte/alpha",
            'fg_dir': "/raid/Data/huangtao/public/matting/vitmatte/fg",
            'bg_dir': "/raid/Data/huangtao/public/matting/vitmatte/bg", 
            'batch_size': 4,  # 验证时可以用更大batch_size
            'size': 256
        },
        'sam_checkpoint': "/raid/Data/huangtao/tangzhice/matting/baseline_A/pretrained/sam_vit_b_01ec64.pth",
        'optimizer': {
            'lr': 5e-5,  # 基础学习率，实际学习率会根据参数组调整
            'weight_decay': 1e-4
        },
        'epoch_max': 100,
        'epoch_val': 5,
        'lr_min': 1e-6
    }

    save_name = args.name
    if save_name is None:
        save_name = 'sam_matting_adapter'
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('/raid/Data/huangtao/tangzhice/matting/baseline_A/test_out', save_name)

    main(config, save_path, args=args)
