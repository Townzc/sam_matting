import os
import torch
import torch.nn as nn

def setup_devices():
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

exp_name = "25_9_18_test_for_learning_rate"