#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

# 设置文件路径
ROOT_DIR = 'data/PNGData/train'  # 根目录
IMAGE_DIR = os.path.join(ROOT_DIR, "img")  # 存放原图的文件夹
ANNOTATION_DIR = os.path.join(ROOT_DIR, "our_mask")  # 存放 mask 标签的文件夹

# 数据集信息
INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

# 数据集类别
CATEGORIES = [
    {
        'id': 1,
        'name': 'build',
        'supercategory': 'shape',
    }
]

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files

def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # 遍历图像文件
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        for image_filename in image_files:
            image = Image.open(image_filename)
            # image_info = pycococreatortools.create_image_info(
            #     image_id, os.path.basename(image_filename), image.size)
            image_info = {
                "id": image_id,
                "file_name": os.path.basename(image_filename),
                "width": image.width,
                "height": image.height
            }
            coco_output["images"].append(image_info)

            # 遍历对应的掩码文件
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                for annotation_filename in annotation_files:
                    print(annotation_filename)

                    # 因为只有一类，直接分配类别 ID
                    class_id = CATEGORIES[0]['id']

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename).convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id += 1

            image_id += 1

    # 保存为 COCO 格式的 JSON 文件
    with open('{}/instances_shape_train2018.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

if __name__ == "__main__":
    main()
