import os
import glob
import json
import argparse
import numpy as np
import imageio.v2 as imageio
import cv2
from tqdm import tqdm
import torch
import yaml
import archs
from dataset import Dataset


# ====================================================== 图像切割 ======================================================
def cut_big_image(image_path, output_dir, block_size=512, stride=512):
    """切割单张大图为小图并保存元数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像并获取元数据
    image_data = imageio.imread(image_path)
    original_height, original_width = image_data.shape[:2]
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 计算填充尺寸
    padding_h = ((original_height // block_size) + 1) * block_size
    padding_w = ((original_width // block_size) + 1) * block_size

    # 创建填充图像
    if len(image_data.shape) == 2:
        padding_img = np.zeros((padding_h, padding_w), dtype=image_data.dtype)
    else:
        padding_img = np.zeros((padding_h, padding_w, image_data.shape[2]), dtype=image_data.dtype)
    padding_img[:original_height, :original_width] = image_data

    # 切割并保存小图
    row_num = 0
    for y in tqdm(range(0, padding_h, stride), desc=f"Cutting {image_name}"):
        row_num += 1
        if y + block_size > padding_h:
            continue
        col_num = 0
        for x in range(0, padding_w, stride):
            col_num += 1
            if x + block_size > padding_w:
                continue
            block = padding_img[y:y + block_size, x:x + block_size]
            block_name = f"{image_name}_{row_num}_{col_num}.tif"
            imageio.imwrite(os.path.join(output_dir, block_name), block)

    # 保存元数据
    metadata = {
        "original_height": original_height,
        "original_width": original_width,
        "padding_height": padding_h,
        "padding_width": padding_w
    }
    with open(os.path.join(output_dir, f"{image_name}_metadata.json"), 'w') as f:
        json.dump(metadata, f)


# ====================================================== 模型预测 ======================================================
def load_model(model_name, model_dir="models"):
    model_path = os.path.join(model_dir, model_name)
    config_path = os.path.join(model_path, "config.yml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in {model_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = archs.__dict__[config['arch']](
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision']
    )

    # 自动适配 GPU 或 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), map_location=device))
    model.eval()
    return model.to(device), config


def predict(model, config, input_dir, output_dir):
    # 创建输出目录
    for c in range(config['num_classes']):
        os.makedirs(os.path.join(output_dir, str(c)), exist_ok=True)

    # 获取所有小图路径
    img_ids = glob.glob(os.path.join(input_dir, '*.tif'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # 创建数据集和数据加载器
    dataset = Dataset(
        img_ids=img_ids,
        img_dir=input_dir,
        mask_dir=None,
        img_ext='.tif',
        mask_ext='.tif',
        num_classes=config['num_classes'],
        transform=None
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # 预测并保存结果
    with torch.no_grad():
        for inputs, _, metas in tqdm(loader, desc="Predicting"):
            inputs = inputs.cuda()
            λ = metas['wavelengths'].float().cuda()  # 从元数据中提取 λ 并转换为浮点类型

            # 计算输出
            if config['deep_supervision']:
                outputs = model(inputs, λ)[-1]
            else:
                outputs = model(inputs, λ)

            outputs = torch.sigmoid(outputs).cpu().numpy()

            for i in range(outputs.shape[0]):
                img_id = metas['img_id'][i]
                for c in range(config['num_classes']):
                    output_path = os.path.join(output_dir, str(c), f"{img_id}.png")
                    cv2.imwrite(output_path, (outputs[i, c] * 255).astype(np.uint8))



# ====================================================== 结果拼接 ======================================================
def merge_predictions(pred_dir, output_dir, block_size=512):
    """将预测结果拼接回原图"""
    # 遍历每个类别目录
    for class_dir in tqdm(glob.glob(os.path.join(pred_dir, '*')), desc="Merging"):
        class_id = os.path.basename(class_dir)
        pred_files = glob.glob(os.path.join(class_dir, '*.png'))

        # 按原图分组预测块
        image_groups = {}
        for pred_file in pred_files:
            base_name = os.path.basename(pred_file).replace('.png', '')
            parts = base_name.split('_')
            image_name = '_'.join(parts[:-2])
            row = int(parts[-2])
            col = int(parts[-1])

            if image_name not in image_groups:
                image_groups[image_name] = []
            image_groups[image_name].append((row, col, pred_file))

        # 拼接每个原图
        for image_name, blocks in image_groups.items():
            # 加载元数据
            metadata_path = os.path.join(os.path.dirname(pred_dir), 'cut', f"{image_name}_metadata.json")
            with open(metadata_path) as f:
                metadata = json.load(f)

            # 初始化填充图像
            merged = np.zeros((metadata['padding_height'], metadata['padding_width']), dtype=np.uint8)

            # 填充每个块
            for row, col, pred_file in blocks:
                y = (row - 1) * block_size
                x = (col - 1) * block_size
                pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
                merged[y:y + block_size, x:x + block_size] = pred

            # 裁剪回原图尺寸
            merged = merged[:metadata['original_height'], :metadata['original_width']]

            # 保存结果
            output_path = os.path.join(output_dir, class_id, f"{image_name}.tif")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            imageio.imwrite(output_path, merged)


# ====================================================== 主流程 ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='输入大图目录')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--model', required=True, help='模型名称')
    args = parser.parse_args()

    # 1. 切割大图
    cut_dir = os.path.join(args.output_dir, 'cut')
    print("=" * 50, "\nStep 1: Cutting large images...")
    for img_path in tqdm(glob.glob(os.path.join(args.input_dir, '*.tif'))):
        cut_big_image(img_path, cut_dir)

    # 2. 模型预测
    pred_dir = os.path.join(args.output_dir, 'pred')
    print("\n" + "=" * 50, "\nStep 2: Predicting segments...")
    model, config = load_model(args.model)
    predict(model, config, cut_dir, pred_dir)

    # 3. 拼接结果
    merge_dir = os.path.join(args.output_dir, 'merged')
    print("\n" + "=" * 50, "\nStep 3: Merging predictions...")
    merge_predictions(pred_dir, merge_dir)

    print("\nAll steps completed! Results saved to:", merge_dir)


if __name__ == "__main__":
    main()