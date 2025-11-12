import copy
import sys
import os

# 修复相对导入问题
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from .image_encoder import ImageEncoderViT
    from .mask_decoder import MattingDecoder
    from .transformer import TwoWayTransformer
    from .matting_criterion import MattingCriterion
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from image_encoder import ImageEncoderViT
    from mask_decoder import MattingDecoder
    from transformer import TwoWayTransformer
    from matting_criterion import MattingCriterion

import torch.nn as nn

models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def make(model_spec, args=None, load_sd=False):

    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model
