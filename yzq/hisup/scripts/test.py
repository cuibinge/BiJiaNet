import os
import argparse
import logging

from hisup.config import cfg
from hisup.detector import BuildingDetector
from hisup.utils.logger import setup_logger
from hisup.utils.checkpoint import DetectronCheckpointer
from tools.test_pipelines import *

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from yacs.config import CfgNode as Cfg
import yaml
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 或者设置为一个较大的值，例如 10亿
def load_config(config_file):
    # 读取 YAML 配置文件并加载为 Python 字典
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # 使用 CfgNode 将字典转换为配置对象
    cfg = Cfg(config_data)
    
    return cfg

# # 调用该方法并读取配置
# config_path = "/qiaowenjiao/HiSup/config-files/lyg_hrnet48.yaml"

# 打印配置内容
print(cfg)

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default=None,
                        )

    parser.add_argument("--eval-type",
                        type=str,
                        help="evalutation type for the test results",
                        default="coco_iou",
                        choices=["coco_iou",  "boundary_iou", "polis"]
                        )

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )

    args = parser.parse_args()
    
    return args


def test(cfg, args):
    logger = logging.getLogger("testing")
    device = cfg.MODEL.DEVICE
    model = BuildingDetector(cfg, test=True)
    model = model.to(device)

    if args.config_file is not None:
        checkpointer = DetectronCheckpointer(cfg,
                                         model,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=True,
                                         logger=logger)
        _ = checkpointer.load()        
        model = model.eval()

    test_pipeline = TestPipeline(cfg, args.eval_type)
    test_pipeline.test(model)
    # test_pipeline.eval()


if __name__ == "__main__":
    args = parse_args()
    print(args.config_file is not None)
    if args.config_file is not None:
#         cfg = get_cfg()  # 获取默认配置
#         print(cfg)
#         cfg.merge_from_file("/qiaowenjiao/HiSup/config-files/lyg_hrnet48.yaml")  # 合并配置文件
#         cfg = get_cfg("/qiaowenjiao/HiSup/config-files/lyg_hrnet48.yaml")  # 获取默认配置
#         cfg = load_config("/qiaowenjiao/HiSup/config-files/lyg_hrnet48.yaml")


#         print(cfg)
        cfg.merge_from_file(args.config_file)
        print(cfg)

    else:
        cfg.OUTPUT_DIR = 'outputs/default'
        os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
    
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
#     if output_dir:
#         if os.path.isdir(output_dir) and args.clean:
#             import shutil
#             shutil.rmtree(output_dir)
#         os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('testing', output_dir)
    logger.info(args)
    if args.config_file is not None:
        logger.info("Loaded configuration file {}".format(args.config_file))
    else:
        logger.info("Loaded the default configuration for testing")

    test(cfg, args)

# import os
# import argparse
# import logging

# from hisup.config import cfg
# from hisup.detector import BuildingDetector
# from hisup.utils.logger import setup_logger
# from hisup.utils.checkpoint import DetectronCheckpointer
# from tools.test_pipelines import TestPipeline

# import torch
# torch.multiprocessing.set_sharing_strategy('file_system')

# def parse_args():
#     parser = argparse.ArgumentParser(description='Testing')

#     parser.add_argument("--config-file",
#                         metavar="FILE",
#                         help="path to config file",
#                         type=str,
#                         default=None,
#                         )

#     parser.add_argument("--eval-type",
#                         type=str,
#                         help="evaluation type for the test results",
#                         default="coco_iou",
#                         choices=["coco_iou",  "boundary_iou", "polis"]
#                         )

#     parser.add_argument("opts",
#                         help="Modify config options using the command-line",
#                         default=None,
#                         nargs=argparse.REMAINDER
#                         )

#     args = parser.parse_args()
    
#     return args


# # def test(cfg, args, logger):
# #     device = cfg.MODEL.DEVICE
# #     logger.info("Creating BuildingDetector model...")
# #     model = BuildingDetector(cfg, test=True)
# #     logger.info("Model created. Moving model to device: {}".format(device))
# #     model = model.to(device)

# #     if args.config_file is not None:
# #         logger.info("Loading checkpoint from config file...")
# #         checkpointer = DetectronCheckpointer(cfg,
# #                                          model,
# #                                          save_dir=cfg.OUTPUT_DIR,
# #                                          save_to_disk=True,
# #                                          logger=logger)
# #         _ = checkpointer.load()        
# #         model = model.eval()
# #         logger.info("Checkpoint loaded and model set to evaluation mode.")

# #     logger.info("Creating test pipeline...")
# #     test_pipeline = TestPipeline(cfg, args.eval_type)
# #     logger.info("Test pipeline created. Starting test...")
# #     test_pipeline.test(model)
# #     # test_pipeline.eval()
# def test(cfg, args, logger):
#     device = cfg.MODEL.DEVICE
#     logger.info("Creating BuildingDetector model...")
#     model = BuildingDetector(cfg, test=True)
#     logger.info("Model created. Moving model to device: {}".format(device))
#     model = model.to(device)

#     if args.config_file is not None:
#         logger.info("Loading checkpoint from config file...")
#         checkpointer = DetectronCheckpointer(cfg,
#                                          model,
#                                          save_dir=cfg.OUTPUT_DIR,
#                                          save_to_disk=True,
#                                          logger=logger)
#         _ = checkpointer.load()
#         model = model.eval()
#         logger.info("Checkpoint loaded and model set to evaluation mode.")

#     logger.info("Creating test pipeline...")
#     test_pipeline = TestPipeline(cfg, args.eval_type)
#     logger.info("Test pipeline created. Starting test...")

#     # 假设test_pipeline有一个方法可以得到输入数据
# #     inputs = test_pipeline.get_inputs(model)

#     # 前向传播并打印输出
#     with torch.no_grad():
#         outputs = model.forward(inputs)
#         logger.info("Model output shape: {}".format(outputs.shape))
#         logger.info("Model output: {}".format(outputs))

#     logger.info("Starting test pipeline test method...")
#     test_pipeline.test(model)
#     # test_pipeline.eval()

# # if __name__ == "__main__":
# #     args = parse_args()
# #     logger = setup_logger('testing', cfg.OUTPUT_DIR if args.config_file is not None else 'outputs/default')
# #     logger.info(args)
# #     if args.config_file is not None:
# #         cfg.merge_from_file(args.config_file)
# #         logger.info("Loaded configuration file {}".format(args.config_file))
# #     else:
# #         cfg.OUTPUT_DIR = 'outputs/default'
# #         os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# #         logger.info("Loaded the default configuration for testing")

# #     cfg.merge_from_list(args.opts)
# #     cfg.freeze()
    
# #     logger.info("Configuration frozen. Starting testing process...")
# #     test(cfg, args, logger)
# # 在主程序中使用
# if __name__ == "__main__":
#     args = parse_args()
#     if args.config_file is not None:
#         cfg.merge_from_file(args.config_file)
#     else:
#         cfg.OUTPUT_DIR = 'outputs/default'
#         os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
    
#     output_dir = cfg.OUTPUT_DIR
#     logger = setup_logger('testing', output_dir)
#     logger.info(args)
#     if args.config_file is not None:
#         logger.info("Loaded configuration file {}".format(args.config_file))
#     else:
#         logger.info("Loaded the default configuration for testing")
#     test(cfg, args, logger)
