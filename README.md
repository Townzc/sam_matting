# SAM 抠图提示适配器

基于 [Segment Anything Model (SAM)](https://segment-anything.com/) 的前景抠图研究原型，通过任务特定的提示适配器（prompt adapter）增强 SAM 的 ViT 编码器，并结合专用的抠图解码器。项目重点在于同时处理自然图像与医学图像：复用 SAM 主干网络，引入轻量化适配层与多损失约束，生成高质量的 alpha matte。

## 仓库结构

```
sam_matting/
├── evaluation_results/        # 验证指标、日志与可视化结果
├── matting-tzc/               # 数据、模型与训练相关源码
│   ├── base/                  # 设备/实验辅助工具
│   ├── config/                # YAML 实验配置
│   ├── criterion/             # 抠图损失函数
│   ├── data/                  # 数据管线与数据集工具
│   └── sam_prompt_adapter/    # 模型、训练、推理、评估脚本
├── pretrained/                # 存放 SAM 预训练权重（例如 sam_vit_b_01ec64.pth）
└── test_out/                  # 训练权重与推理输出
```

## 核心组件

- **基础工具** – `matting-tzc/base/base.py` 暴露 `setup_devices()`，用于指定 CUDA 设备并统一实验命名约定。【F:matting-tzc/base/base.py†L1-L6】
- **实验配置** – `matting-tzc/config/cod-sam-vit-b.yaml` 定义数据集路径、归一化、优化器以及 SAM 编码器超参数。运行前需根据本地环境更新路径与权重。【F:matting-tzc/config/cod-sam-vit-b.yaml†L1-L86】
- **数据处理** – `matting-tzc/data/data_processor.py` 负责融合自然/医学图像批次，将 trimap 转换为 mask prompt，并对提示进行归一化以适配 SAM 输入。【F:matting-tzc/data/data_processor.py†L1-L138】
- **模型定义** – `matting-tzc/sam_prompt_adapter/sam_prompt_adapter.py` 将 ViT 编码器、提示注入、抠图解码器与多损失准则整合为 `SAMAdapterPrompt` 模型。【F:matting-tzc/sam_prompt_adapter/sam_prompt_adapter.py†L1-L142】
- **训练入口** – `matting-tzc/sam_prompt_adapter/sam_prompt_train_A.py` 加载配置、冻结非适配器编码器参数、设置分组学习率，并在验证阶段记录 MSE、SAD 等指标。【F:matting-tzc/sam_prompt_adapter/sam_prompt_train_A.py†L1-L125】
- **推理工具** – `matting-tzc/sam_prompt_adapter/inference.py` 恢复训练权重，准备图像与掩码提示，批量生成 alpha matte，可选可视化输出。【F:matting-tzc/sam_prompt_adapter/inference.py†L1-L113】

## 环境准备

1. 创建 Python 环境（建议 Python ≥ 3.8），确保 PyTorch 与 CUDA 可用。
2. 安装 SAM 与抠图解码器所需依赖（例如 PyTorch、torchvision、numpy、OpenCV、tqdm、Pillow、matplotlib、PyYAML）。可按需编写或调整 `requirements.txt`。
3. 下载 SAM 权重（如 `sam_vit_b_01ec64.pth`）并放入 `pretrained/` 目录。
4. 检查并修改脚本中的绝对路径（如 YAML 配置里的数据集路径、`data_processor.py` 中的 `sys.path` 设置、推理脚本的默认路径）以匹配本地环境。

## 数据准备

- 在 `config/cod-sam-vit-b.yaml` 中配置训练/验证/测试数据集。每个数据集均提供配对的图像与 alpha matte。自然数据集从imagenet节选，医学图像由医院提供（20张），trimap由人工标注。
- `DataProcessor` 依赖 trimap 或 alpha matte 生成 mask prompt，请确保数据集封装返回所需键值（如 `image`/`inp`、`alpha`/`gt` 以及可选的 `mask` 或 `trimap`）。
## 训练

```bash
cd sam_matting/matting-tzc
python sam_prompt_adapter/sam_prompt_train_A.py \
    --config config/cod-sam-vit-b.yaml
```

脚本启动后将构建 `SAMAdapterPrompt`，冻结非适配器的 ViT 参数，并为适配器模块设置更高学习率的分组优化器。【F:matting-tzc/sam_prompt_adapter/sam_prompt_train_A.py†L64-L116】训练过程中会在每个验证 epoch 记录 MSE 与 SAD 指标。【F:matting-tzc/sam_prompt_adapter/sam_prompt_train_A.py†L28-L61】

## 推理

```bash
cd sam_matting/matting-tzc
python sam_prompt_adapter/inference.py \
    --checkpoint ./test_out/checkpoint_last.pth \
    --input_dir <image_dir> \
    --mask_dir <mask_dir> \
    --output_dir ./test_out/inference_out
```

该脚本会加载模型权重，准备图像与二值掩码提示（若缺失则退化为全前景掩码），并导出预测的 alpha matte；可选开启可视化用于质检。【F:matting-tzc/sam_prompt_adapter/inference.py†L41-L113】

## 评估

使用 `sam_prompt_adapter/evaluate.py` 计算 Dice、SAD、MSE、梯度惩罚等指标。运行前请设置预测结果与真值 matte 的路径。

## 提示与后续开发

- 建议将绝对路径的 `sys.path` 修改为相对导入或环境变量以提升可移植性。【F:matting-tzc/data/data_processor.py†L9-L18】【F:matting-tzc/sam_prompt_adapter/inference.py†L20-L29】
- 如需自定义损失组合或权重，可扩展 `MattingCriterion`。
- 可在 YAML 配置中实验适配器深度（如 `tuning_stage`、`prompt_type` 等），以平衡精度与计算成本。【F:matting-tzc/config/cod-sam-vit-b.yaml†L43-L74】
- 若需分布式训练，可在现有脚本基础上集成 PyTorch DDP 或 Lightning。

---

本文档概述了如何在该项目中准备数据、训练 SAM 提示适配器模型以及执行推理与评估。
