import os
import sys

sys.path.append("../")
sys.path.append("../model/")
sys.path.append("../data_processing/")

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import scipy.ndimage
from skimage.measure import label
import scipy.ndimage.morphology
import csv
from datetime import datetime


# 默认路径配置 - 请根据您的实际情况修改
GT_path = "/raid/Data/huangtao/public/LNSM/test/softmask"  # 真实标签路径
rst_path = "/raid/Data/huangtao/tangzhice/matting/baseline_A/test_out/inference_out"  # 预测结果路径  
mask_path = "/raid/Data/huangtao/public/LNSM/test/binarymask"  # Binary mask路径

# 原始LIDC数据集路径（备份）
# GT_path = "/raid/Data/huangtao/zhangqy/dataset/LIDC/fold_0/test/softmask"
# rst_path = "/raid/Data/huangtao/zhangqy/dataset/LIDC/fold_0/output_alpha/output_of_test_reply_v2/k21_d5_70000_batch_test_20250724_164559"
# mask_path = "/raid/Data/huangtao/zhangqy/dataset/LIDC/fold_0/out_trimap/15201_test_batch_trimap_kernel_21_5"


def setup_devices():
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def calculate_DICE(pred, label, smooth = 1.0):
    # pred = pred.view(-1)
    # label = label.view(-1)
    intersection = (pred * label).sum()
    dice = (2.0 * intersection + smooth)/(pred.sum() + label.sum() + smooth)
    return dice



def compute_sad_loss(pred, target, mask):
    """
    计算SAD损失
    Args:
        pred: 预测结果
        target: 真实标签
        mask: binary mask (0-255) 或 trimap (0, 128, 255)
    """
    error_map = np.abs((pred - target) / 255.0)
    
    # 检测是否为trimap格式（包含128值）或binary mask格式
    unique_values = np.unique(mask)
    if 128 in unique_values:
        # trimap格式：只在unknown区域(128)计算
        mask_region = (mask == 128)
    else:
        # binary mask格式：在前景区域计算损失
        mask_region = (mask > 127)  # 前景区域
    
    loss = np.sum(error_map * mask_region)
    return loss / 1000, np.sum(mask_region) / 1000

def compute_mse_loss(pred, target, mask):
    """
    计算MSE损失
    Args:
        pred: 预测结果
        target: 真实标签  
        mask: binary mask (0-255) 或 trimap (0, 128, 255)
    """
    error_map = (pred - target) / 255.0
    
    # 检测是否为trimap格式（包含128值）或binary mask格式
    unique_values = np.unique(mask)
    if 128 in unique_values:
        # trimap格式：只在unknown区域(128)计算
        mask_region = (mask == 128)
    else:
        # binary mask格式：在前景区域计算损失
        mask_region = (mask > 127)  # 前景区域
        
    loss = np.sum((error_map ** 2) * mask_region) / (np.sum(mask_region) + 1e-8)

    # # if test on whole image (Disitinctions-646), please uncomment this line
    # loss = np.sum(error_map ** 2) / (pred.shape[0] * pred.shape[1])

    return loss

def gauss(x, sigma):
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y


def dgauss(x, sigma):
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y


def gaussgradient(im, sigma):
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.int32)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy


def compute_gradient_loss(pred, target, mask):
    """
    计算Gradient损失
    Args:
        pred: 预测结果
        target: 真实标签
        mask: binary mask (0-255) 或 trimap (0, 128, 255)
    """
    pred = pred / 255.0
    target = target / 255.0

    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    
    # 检测是否为trimap格式（包含128值）或binary mask格式
    unique_values = np.unique(mask)
    if 128 in unique_values:
        # trimap格式：只在unknown区域(128)计算
        loss = np.sum(error_map[mask == 128])
    else:
        # binary mask格式：在前景区域计算损失
        loss = np.sum(error_map[mask > 127])

    return loss / 1000.


def getLargestCC(segmentation):
    labels = label(segmentation, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC


def compute_connectivity_error(pred, target, mask, step):
    """
    计算Connectivity损失
    Args:
        pred: 预测结果
        target: 真实标签
        mask: binary mask (0-255) 或 trimap (0, 128, 255)
        step: 阈值步长
    """
    pred = pred / 255.0
    target = target / 255.0
    h, w = pred.shape

    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones_like(pred, dtype=float) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(int)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(int)

        omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(int)
        flag = ((l_map == -1) & (omega == 0)).astype(int)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(int)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(int)
    
    # 检测是否为trimap格式（包含128值）或binary mask格式
    unique_values = np.unique(mask)
    if 128 in unique_values:
        # trimap格式：只在unknown区域(128)计算
        loss = np.sum(np.abs(pred_phi - target_phi)[mask == 128])
    else:
        # binary mask格式：在前景区域计算损失
        loss = np.sum(np.abs(pred_phi - target_phi)[mask > 127])

    return loss / 1000.

def RefineTrimap(old_trimap, pred, is_use_refine = False, ):
    if is_use_refine == False:
        return old_trimap
    
    # print("pred_max: ", np.max(pred))
    # print("pred_min: ", np.min(pred))

    # 128
    mask_128 = pred > 128
    tm_128 = np.zeros(pred.shape)
    tm_128[mask_128] = 128
    tm_255 = np.zeros(pred.shape)
    tm_255[mask_128] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    dilated_tm_128 = cv2.dilate(tm_128, kernel, iterations=1)

    # 255
    kernel_eroded = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    eroded_tm_255 = cv2.erode(tm_255, kernel_eroded, iterations=1)

    # trimap
    mask_255 = eroded_tm_255 > 250
    rst_trimap = dilated_tm_128
    rst_trimap[mask_255] = 0
    new_trimap = old_trimap.copy()
    new_mask = rst_trimap>100
    new_trimap[new_mask] = 128

    return new_trimap

def test_batch(is_save = False):
    all_gt_file_paths = os.listdir(GT_path)

    all_dice = 0.0
    num = 0.0

    for curr_file in all_gt_file_paths:
        curr_gt_file_path = GT_path + curr_file
        curr_rst_file_path = rst_path + curr_file
        curr_trimap_file_path = trimap_path + curr_file

        gt_img = cv2.imread(curr_gt_file_path, cv2.IMREAD_GRAYSCALE)
        rst_img = cv2.imread(curr_rst_file_path, cv2.IMREAD_GRAYSCALE)
        trimap_img = cv2.imread(curr_trimap_file_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.resize(gt_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        rst_img = cv2.resize(rst_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        trimap_img = cv2.resize(trimap_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        trimap_img = np.ones(trimap_img.shape, dtype=trimap_img.dtype) * 128
        gt_img = gt_img / 255.0
        rst_img = rst_img / 255.0
        # gt_img = np.ones(trimap_img.shape, dtype=trimap_img.dtype)
        # rst_img = np.ones(trimap_img.shape, dtype=trimap_img.dtype)

        curr_dice = calculate_DICE(rst_img, gt_img)
        print(curr_dice)
        all_dice = all_dice + curr_dice
        num = num + 1

    return all_dice/num
    # return all_dice



def evaluate(rst_path, GT_path, mask_path, mse_threshold=0, csv_save_path=None, badcases_save_dir=None):
    img_names = []
    mse_loss_unknown = []
    sad_loss_unknown = []
    grad_loss_unknown = []
    conn_loss_unknown = []

    bad_cases = []
    all_results = []

    for i, img in tqdm(enumerate(os.listdir(rst_path))):
        # if img == "4719.png" :
        #     continue
        if not((os.path.isfile(os.path.join(rst_path, img)) and
                os.path.isfile(os.path.join(GT_path, img)) and
                os.path.isfile(os.path.join(mask_path, img)))):
            print('[{}/{}] "{}" skipping'.format(i, len(os.listdir(GT_path)), img))
            continue
        # if (img == "2415.png") or (img == "14381.png"):
        #     continue
        file_name = os.path.splitext(img)[0]

        pred = cv2.imread(os.path.join(rst_path, img), 0).astype(np.float32)
        label = cv2.imread(os.path.join(GT_path, img), 0).astype(np.float32)
        mask = cv2.imread(os.path.join(mask_path, img), 0).astype(np.float32)
        pred = cv2.resize(pred, (256, 256), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # 检测mask类型并处理
        unique_values = np.unique(mask)
        if 128 in unique_values:
            # 如果是trimap格式，可以使用RefineTrimap
            mask = RefineTrimap(mask, pred, is_use_refine=True)
        # 如果是binary mask，直接使用
        
        # calculate loss
        mse_loss_unknown_ = compute_mse_loss(pred, label, mask)
        sad_loss_unknown_ = compute_sad_loss(pred, label, mask)[0]
        grad_loss_unknown_ = compute_gradient_loss(pred, label, mask)
        conn_loss_unknown_ = compute_connectivity_error(pred, label, mask, step = 0.01)
        # print(i, "  mse: ", mse_loss_unknown_, " sad: ", sad_loss_unknown_)
        print(i, "  ", file_name, "  mse: ", mse_loss_unknown_, " sad: ", sad_loss_unknown_, " grad: ", grad_loss_unknown_, " conn: ", conn_loss_unknown_)

        # save for average
        img_names.append(img)

        mse_loss_unknown.append(mse_loss_unknown_)  # mean l2 loss per unknown pixel
        sad_loss_unknown.append(sad_loss_unknown_)  # l1 loss on unknown area
        grad_loss_unknown.append(grad_loss_unknown_)  
        conn_loss_unknown.append(conn_loss_unknown_)
        # 记录所有结果
        all_results.append({
            'image_name': img,
            'mse_loss': mse_loss_unknown_,
            'sad_loss': sad_loss_unknown_,
            'gradient_loss': grad_loss_unknown_,
            'connectivity_loss': conn_loss_unknown_
        })
        # 判断是否为bad case
        if mse_loss_unknown_ > mse_threshold:
            bad_cases.append({
                'image_name': img,
                'mse_loss': mse_loss_unknown_,
                'sad_loss': sad_loss_unknown_,
                'gradient_loss': grad_loss_unknown_,
                'connectivity_loss': conn_loss_unknown_
            })
  
    print('* Unknown Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean(), ' GRAD:', np.array(grad_loss_unknown).mean(), ' CONN:', np.array(conn_loss_unknown).mean())

    # 保存到csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if csv_save_path is None:
        csv_filename = f"eval_results_{timestamp}.csv"
    else:
        if os.path.isdir(csv_save_path):
            csv_filename = os.path.join(csv_save_path, f"v2_eval_results_{timestamp}.csv")
        else:
            csv_filename = csv_save_path
    # # 自动创建父目录
    # os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'mse_loss', 'sad_loss', 'gradient_loss', 'connectivity_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
        # 写入overall均值
        writer.writerow({
            'image_name': 'overall',
            'mse_loss': np.array(mse_loss_unknown).mean(),
            'sad_loss': np.array(sad_loss_unknown).mean(),
            'gradient_loss': np.array(grad_loss_unknown).mean(),
            'connectivity_loss': np.array(conn_loss_unknown).mean()
        })
    print(f"详细评价结果已保存到: {csv_filename}")
    return bad_cases, csv_filename


def visualize_bad_cases(
    bad_cases, rst_path, GT_path, mask_path, output_folder='bad_cases',
    image_path=None, pred_mask_path=None
):
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    else:
        output_folder = 'bad_cases'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    for case in bad_cases:
        img_name = case['image_name']
        img_path = os.path.join(rst_path, img_name)
        gt_img_path = os.path.join(GT_path, img_name)
        mask_img_path = os.path.join(mask_path, img_name)
        origin_img = cv2.imread(os.path.join(image_path, img_name)) if image_path else None
        pred_mask_img = cv2.imread(os.path.join(pred_mask_path, img_name)) if pred_mask_path else None

        pred_img = cv2.imread(img_path)
        gt_img = cv2.imread(gt_img_path)
        mask_img = cv2.imread(mask_img_path)

        imgs = []
        if origin_img is not None:
            imgs.append(origin_img)
        if pred_mask_img is not None:
            imgs.append(pred_mask_img)
        imgs.extend([pred_img, mask_img, gt_img])

        target_size = (256, 256)
        imgs_resized = []
        for im in imgs:
            if im is not None:
                imgs_resized.append(cv2.resize(im, target_size, interpolation=cv2.INTER_NEAREST))
        combined_img = np.hstack(imgs_resized)

        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, combined_img)

        # print(f'Saved bad case image to {output_path}')


if __name__ == "__main__":
    setup_devices()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print('begin test_batch')
    # average_dice = test_batch(is_save = True)
    # print('average dice: ', average_dice)
    # evaluate()
    # 这里可以自定义保存路径
    csv_path = "/raid/Data/huangtao/tangzhice/matting/baseline_A/test_out/evaluation_results/csv" 
    os.makedirs(csv_path, exist_ok=True)   # CSV保存路径
    badcases_dir = "/raid/Data/huangtao/tangzhice/matting/baseline_A/test_out/evaluation_results/bad_cases"  # badcases保存文件夹
    os.makedirs(badcases_dir, exist_ok=True)  
    bad_cases, csv_filename = evaluate(rst_path, GT_path, mask_path, mse_threshold=0, csv_save_path=csv_path, badcases_save_dir=badcases_dir)
    
    # 原始图像路径（用于可视化）
    image_path = "/raid/Data/huangtao/public/LNSM/test/image"
    pred_mask_path = mask_path  # 使用相同的mask路径进行可视化
    visualize_bad_cases(
        bad_cases, rst_path, GT_path, mask_path,
        output_folder=badcases_dir,
        image_path=image_path,
        pred_mask_path=pred_mask_path
    )
    print('end test_batch')

