import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 修复相对导入问题
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from .models import register
    from .image_encoder import ImageEncoderViT
    from .mask_decoder import MattingDecoder
    from .transformer import TwoWayTransformer
except ImportError:
    from models import register
    from image_encoder import ImageEncoderViT
    from mask_decoder import MattingDecoder
    from transformer import TwoWayTransformer

from matting_criterion import MattingCriterion

logger = logging.getLogger(__name__)
from typing import Any, Optional, Tuple, List, Dict

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

@register('sam_adapter_prompt')
class SAMAdapterPrompt(nn.Module):
    """
    SAM-Adapter模型：输入只为image，mask prompt通过adapter注入ViT中间层
    """
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        # 图像编码器（ViT，输入通道为3）
        self.image_encoder = ImageEncoderViT(
            img_size=256,
            patch_size=16,
            in_chans=3,  # 只输入image
            embed_dim=768,
            depth=12,
            num_heads=12,
            out_chans=256
        )
        # 抠图解码器
        self.mask_decoder = MattingDecoder(
            in_chans=256,
            img_chans=3,   # 只输入image
            convstream_out=[48, 96, 192],
            fusion_out=[256, 128, 64, 32],
            merge='mul',
            change_image_resolution=True,
        )
        # 位置编码
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats=128)
        # 损失函数
        if loss is None:
            self.loss = MattingCriterion(
                losses=['unknown_l1_loss', 'known_l1_loss', 'loss_gradient_penalty', 'loss_pha_laplacian']
            )
        else:
            self.loss = loss
        self.mask_threshold = 0.5
        self.input = None
        self.gt_mask = None
        self.mask_inputs = None
        self.original_size = None
        self.pred_mask = None
        self.sample_map = None

    def set_input(self, input, gt_mask, mask_inputs=None, sample_map=None):
        self.input = input
        self.gt_mask = gt_mask
        self.mask_inputs = mask_inputs
        self.sample_map = sample_map
        if input is not None:
            self.original_size = input.shape[-2:]

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(64)

    def forward(self, batched_input=None, multimask_output=False):
        if batched_input is None:
            bs = 1
            # mask prompt通过adapter注入
            mask_prompt = None
            if self.mask_inputs is not None:
                mask_prompt = F.interpolate(
                    self.mask_inputs,
                    size=self.input.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            # 只输入image
            features = self.image_encoder(self.input, mask_prompt=mask_prompt)
            decoder_output = self.mask_decoder(
                features=features,
                sim=None,
                images=self.input
            )
            self.pred_mask = decoder_output['phas']
            if hasattr(self, 'before_sigmoid'):
                del self.before_sigmoid
            if hasattr(self, 'trans'):
                del self.trans
            if hasattr(self, 'features'):
                del self.features
            if hasattr(self, 'pred_mask_binary'):
                del self.pred_mask_binary
            return
        # 有参数调用方式
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        outputs = []
        for image_record in batched_input:
            mask_prompt = None
            if "mask_inputs" in image_record and image_record["mask_inputs"] is not None:
                mask_prompt = F.interpolate(
                    image_record["mask_inputs"],
                    size=image_record["image"].shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            curr_image = self.preprocess(image_record["image"]).unsqueeze(0)
            features = self.image_encoder(curr_image, mask_prompt=mask_prompt)
            decoder_output = self.mask_decoder(
                features=features,
                sim=None,
                images=curr_image
            )
            alpha_matte = decoder_output['phas']
            before_sigmoid = decoder_output.get('befo', None)
            trans = decoder_output.get('trans', None)
            if alpha_matte.shape[-2:] != image_record["original_size"]:
                alpha_matte = F.interpolate(
                    alpha_matte,
                    size=image_record["original_size"],
                    mode='bilinear',
                    align_corners=False
                )
            outputs.append({
                "alpha_matte": alpha_matte,
                "before_sigmoid": before_sigmoid,
                "trans": trans,
            })
        return outputs

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if x.shape[-2:] != (256, 256):  # 修改为256x256
            x = F.interpolate(x.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        return x

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        return masks

    def get_binary_prediction(self):
        if hasattr(self, 'pred_mask_binary') and self.pred_mask_binary is not None:
            return self.pred_mask_binary
        elif self.pred_mask is not None:
            binary_mask = (self.pred_mask > self.mask_threshold).float()
            return binary_mask
        return None

    def backward_G(self):
        if self.gt_mask is not None and self.pred_mask is not None:
            preds = {
                'phas': self.pred_mask,
                'befo': None,
                'trans': None,
            }
            targets = {
                'phas': self.gt_mask,
            }
            sample_map = self.sample_map
            if sample_map is None:
                sample_map = torch.ones_like(self.gt_mask)
            losses = self.loss(sample_map, preds, targets)
            total_loss = sum(losses.values())
            total_loss.backward()
            return total_loss, losses
        return None, None

    def optimize_parameters(self):
        self.zero_grad()
        total_loss, losses = self.backward_G()
        return total_loss, losses

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad 