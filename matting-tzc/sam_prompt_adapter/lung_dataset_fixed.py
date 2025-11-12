import os
import cv2
import torch
from torch.utils.data import Dataset

class Lung20DatasetFixed(Dataset):
    """
    修正版肺部数据集：适配实际的目录结构
    实际结构: images/, alphas/, trimaps/ (复数形式)
    """
    def __init__(self, root_dir):
        # 使用实际的目录结构
        self.image_dir = os.path.join(root_dir, "images")   # 复数形式
        self.alpha_dir = os.path.join(root_dir, "alphas")   # 复数形式  
        self.trimap_dir = os.path.join(root_dir, "trimaps") # 保持一致

        # 检查目录是否存在
        for name, path in [("images", self.image_dir), ("alphas", self.alpha_dir), ("trimaps", self.trimap_dir)]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"目录不存在: {path}")

        self.image_names = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])[:20]  # 取前20张

        print(f"加载肺部数据集: {len(self.image_names)}张图像")
        print(f"图像目录: {self.image_dir}")
        print(f"Alpha目录: {self.alpha_dir}")
        print(f"Trimap目录: {self.trimap_dir}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, name)
        alpha_path = os.path.join(self.alpha_dir, name)
        trimap_path = os.path.join(self.trimap_dir, name)

        # 检查文件是否存在
        for file_path in [img_path, alpha_path, trimap_path]:
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在: {file_path}")

        # 读取图片
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
        trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)

        if image is None or alpha is None or trimap is None:
            raise ValueError(f"无法读取文件: {name}")

        # 调整尺寸到256x256以匹配SAM训练尺寸
        target_size = (256, 256)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, target_size, interpolation=cv2.INTER_NEAREST)
        trimap = cv2.resize(trimap, target_size, interpolation=cv2.INTER_NEAREST)

        # 统一三值化：0, 127, 255
        trimap_proc = trimap.copy()
        trimap_proc[(trimap_proc > 10) & (trimap_proc < 200)] = 128   # unknown
        trimap_proc[trimap_proc >= 200] = 255                         # 前景
        trimap_proc[trimap_proc <= 10] = 0                            # 背景
        trimap = trimap_proc

        # 归一化
        image = image.astype('float32') / 255.0
        alpha = alpha.astype('float32') / 255.0
        trimap = trimap.astype('float32')

        # 转为tensor
        image = torch.from_numpy(image.transpose(2, 0, 1))  # 3xHxW
        alpha = torch.from_numpy(alpha).unsqueeze(0)        # 1xHxW
        trimap = torch.from_numpy(trimap).unsqueeze(0)      # 1xHxW

        return {
            "image": image,
            "alpha": alpha,
            "trimap": trimap,
            "name": name
        }

if __name__ == "__main__":
    # 测试数据集
    root_dir = "/raid/Data/huangtao/zhangqy/dataset/LNSM/lung20"
    try:
        dataset = Lung20DatasetFixed(root_dir)
        print(f"✅ 数据集加载成功，共{len(dataset)}张图像")
        
        # 测试第一个样本
        sample = dataset[0]
        print(f"✅ 样本测试成功:")
        print(f"   图像形状: {sample['image'].shape}")
        print(f"   Alpha形状: {sample['alpha'].shape}")
        print(f"   Trimap形状: {sample['trimap'].shape}")
        print(f"   文件名: {sample['name']}")
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
