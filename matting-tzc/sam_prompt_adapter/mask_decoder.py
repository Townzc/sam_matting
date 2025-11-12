# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Tuple, Type

# 直接集成detail_capture_dino.py的相关类

class Basic_Conv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """
    def __init__(self, in_chans, out_chans, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """
    def __init__(self, in_chans=4, out_chans=[48, 96, 192]):
        super().__init__()
        self.convs = nn.ModuleList()
        self.conv_chans = out_chans.copy()
        self.conv_chans.insert(0, in_chans)
        for i in range(len(self.conv_chans)-1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i+1]
            self.convs.append(Basic_Conv3x3(in_chan_, out_chan_))
    def forward(self, x):
        out_dict = {'D0': x}
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            name_ = 'D'+str(i+1)
            out_dict[name_] = x
        return out_dict

class Fusion_Block(nn.Module):
    """
    Simple fusion block to fuse feature from ConvStream and Plain Vision Transformer.
    """
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1)
    def forward(self, x, D):
        F_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.cat([D, F_up], dim=1)
        out = self.conv(out)
        return out

class Matting_Head(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """
    def __init__(self, in_chans=32, mid_chans=16):
        super().__init__()
        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, 3, 1, 1),
            nn.BatchNorm2d(mid_chans),
            nn.ReLU(True),
            nn.Conv2d(mid_chans, 1, 1, 1, 0)
        )
    def forward(self, x):
        x = self.matting_convs(x)
        return x

class MattingDecoder(nn.Module):
    """
    Matting Decoder，直接集成抠图结构。
    输入：
        features: 编码器输出特征 [B, C, H, W]
        sim: mask提示或None [B, C, H, W] 或 [B, 1, H, W]
        images: 输入图像+mask通道 [B, 4, H, W]
    输出：
        dict: { 'phas': alpha matte, 'befo': before sigmoid, 'trans': trimap }
    """
    def __init__(self, in_chans=256, img_chans=4, convstream_out=[48, 96, 192], fusion_out=[256, 128, 64, 32], merge='mul', change_image_resolution=True, **kwargs):
        super().__init__()
        assert len(fusion_out) == len(convstream_out) + 1
        self.convstream = ConvStream(in_chans=img_chans)
        self.conv_chans = self.convstream.conv_chans
        # accept legacy args (merge/change_image_resolution) for backward-compat
        self.merge = merge
        self.other = [0,1,0,0]
        self.fusion_blks = nn.ModuleList()
        self.fus_channs = fusion_out.copy()
        self.change_image_resolution = change_image_resolution
        if self.merge=='mul':
            self.fus_channs.insert(0, in_chans)
        else:
            self.fus_channs.insert(0, in_chans+1)
        for i in range(len(self.fus_channs)-1):
            self.fusion_blks.append(
                Fusion_Block(
                    in_chans = self.fus_channs[i] + self.conv_chans[-(i+1)]+self.other[i],
                    out_chans = self.fus_channs[i+1],
                )
            )
        self.matting_head = Matting_Head(in_chans=fusion_out[-1])
        self.trimap_head = Matting_Head(in_chans=fusion_out[0])
    def forward(self, features, sim=None, images=None, *args, **kwargs):
        # features: 编码器输出
        # 兼容旧/新调用：
        # - 旧: forward(features, sim, images)
        # - 新: forward(features, images)
        if images is None and sim is not None and isinstance(sim, torch.Tensor):
            # 允许把第二个张量参数当作 images 使用（忽略 sim）
            images = sim
            sim = None
        # sim 仅在旧逻辑 merge 路径中被使用；若为 None 则保持原特征
        if self.merge=='mul':
            features = features*sim if sim is not None else features
        else:
            features = torch.cat((features, sim), dim=1) if sim is not None else features
        H,W = images.shape[-2:]
        if self.change_image_resolution:   
            f_h,f_w = features.shape[-2:]
            images = F.interpolate(images, size=(256, 256),mode='bilinear')
        else:
            features = F.interpolate(features, size=(int(H/16), int(W/16)))
        detail_features = self.convstream(images)
        for i in range(len(self.fusion_blks)):
            d_name_ = 'D'+str(len(self.fusion_blks)-i-1)
            features = self.fusion_blks[i](features, detail_features[d_name_])
            if i==0:
                trans = self.trimap_head(features)
                features = torch.cat((features, trans),dim=1)
        before_sigmoid = self.matting_head(features)
        phas = torch.sigmoid(before_sigmoid)
        if self.change_image_resolution:   
            phas = F.interpolate(phas, size=(H,W), mode='bilinear')
        return {'phas': phas, 'befo': before_sigmoid, 'trans':trans}
