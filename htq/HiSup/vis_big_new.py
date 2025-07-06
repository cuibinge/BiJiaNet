import os
import numpy as np
import torch
import argparse
from skimage import io
from skimage.measure import label, regionprops
from shapely.geometry import Polygon
from hisup.config import cfg
from hisup.detector import get_pretrained_model
from hisup.dataset.build import build_transform
from hisup.utils.comm import to_single_device
from hisup.utils.visualizer import show_polygons
from hisup.utils.polygon import *
import cv2
import torch.nn.functional as F
from scipy.spatial import cKDTree
import geopandas as gpd
from rasterio import open as rio_open
from rasterio.windows import Window
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from hisup.detector import BuildingDetector
import rasterio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


# 从第二个代码中引入的函数
def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask

def get_junctions(jloc, joff, topk=300, th=0):
    height, width = jloc.size(1), jloc.size(2)
    jloc = jloc.reshape(-1)
    joff = joff.reshape(2, -1)
    scores, index = torch.topk(jloc, k=topk)
    y = (index // width).float() + torch.gather(joff[1], 0, index) + 0.5
    x = (index % width).float() + torch.gather(joff[0], 0, index) + 0.5
    junctions = torch.stack((x, y)).t()
    return junctions[scores > th], scores[scores > th]

def simple_polygon(poly, thres=30):
    if (poly[0] == poly[-1]).all():
        poly = poly[:-1]
    lines = np.concatenate((poly, np.roll(poly, -1, axis=0)), axis=1)
    vec0 = lines[:, 2:] - lines[:, :2]
    vec1 = np.roll(vec0, -1, axis=0)
    vec0_ang = np.arctan2(vec0[:, 1], vec0[:, 0]) * 180 / np.pi
    vec1_ang = np.arctan2(vec1[:, 1], vec1[:, 0]) * 180 / np.pi
    lines_ang = np.abs(vec0_ang - vec1_ang)
    flag1 = np.roll((lines_ang > thres), 1, axis=0)
    flag2 = np.roll((lines_ang < 360 - thres), 1, axis=0)
    simple_poly = poly[np.bitwise_and(flag1, flag2)]
    simple_poly = np.concatenate((simple_poly, simple_poly[0].reshape(-1, 2)))
    return simple_poly

def is_convex_point(prev, curr, next):
    v1 = (prev[0] - curr[0], prev[1] - curr[1])
    v2 = (next[0] - curr[0], next[1] - curr[1])
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    return cross_product > 0

# 原始的多边形生成函数
def convert_mask_to_polygons(juncs_whole_img, pred_whole_img):
    # match junction and seg results
    polygons = []
    props = regionprops(label(pred_whole_img > 0.5))
    for prop in tqdm(props, leave=False, desc='polygon generation'):
        y1, x1, y2, x2 = prop.bbox
        bbox = [x1, y1, x2, y2]
        select_juncs = juncs_in_bbox(bbox, juncs_whole_img, expand=8)
        poly, juncs_sa, _, _, juncs_index = generate_polygon(prop, pred_whole_img, select_juncs, pid=0, test_inria=True)
        if juncs_sa.shape[0] == 0:
            continue
        
        if len(juncs_index) == 1:
            polygons.append(Polygon(poly))
        else:
            poly_ = Polygon(poly[juncs_index[0]], \
                            [poly[idx] for idx in juncs_index[1:]])
            polygons.append(poly_)
    
    return polygons

# 优化后的多边形生成函数
def convert_mask_to_polygons2(juncs_whole_img, pred_whole_img, thres=0.5):
    """优化后的多边形生成函数"""
    polygons = []
    # 二值化并获取区域
    binary_mask = pred_whole_img > thres
    label_image = label(binary_mask)
    
    # 提前过滤面积过小或过大的区域
    props = [prop for prop in regionprops(label_image) if 100 <= prop.area <= 10000]
    
    # 构建KD树加速junction查询
    junc_tree = cKDTree(juncs_whole_img)
    
    def process_prop(prop):
        y1, x1, y2, x2 = prop.bbox
        bbox = [x1, y1, x2, y2]
        # 使用KD树查询junctions
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        radius = max(x2 - x1, y2 - y1) / 2 + 8  # expand=8
        candidate_ids = junc_tree.query_ball_point(center, r=radius)
        select_juncs = juncs_whole_img[candidate_ids]
        
        # 只保留凸点
        if len(select_juncs) > 0:
            convex_juncs = []
            for i, junc in enumerate(select_juncs):
                prev_idx = (i - 1) % len(select_juncs)
                next_idx = (i + 1) % len(select_juncs)
                if is_convex_point(select_juncs[prev_idx], junc, select_juncs[next_idx]):
                    convex_juncs.append(junc)
            if len(convex_juncs) < 3:
                convex_juncs = select_juncs  # 如果凸点不足3个，使用所有点
            convex_juncs = np.array(convex_juncs)
        else:
            convex_juncs = select_juncs
        
        # 生成轮廓并匹配凸点
        prop_mask = np.zeros_like(pred_whole_img, dtype=np.uint8)
        prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
        contours, _ = cv2.findContours(prop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return None
        
        contour = contours[0].squeeze(1)
        if len(convex_juncs) > 0:
            cj_match_ = np.argmin(cdist(contour, convex_juncs), axis=1)
            cj_dis = cdist(contour, convex_juncs)[np.arange(len(cj_match_)), cj_match_]
            u, ind = np.unique(cj_match_[cj_dis < 3], return_index=True)
            if len(u) > 2:
                init_poly = convex_juncs[u[np.argsort(ind)]]
            else:
                init_poly = contour
        else:
            init_poly = contour
        
        # 简化多边形，角度阈值30度
        simplified_poly = simple_polygon(init_poly, thres=30)
        if len(simplified_poly) >= 4:  # 确保多边形有效
            return Polygon(simplified_poly)
        return None
    
    # 并行处理多边形生成
    with ThreadPoolExecutor(max_workers=min(6, os.cpu_count())) as executor:
        polygons = list(tqdm(
            executor.map(process_prop, props),
            total=len(props),
            desc='Polygon generation'
        ))
    
    return [p for p in polygons if p is not None]

# 其余函数保持不变
def inference_single(cfg, model, image, device):
    transform = build_transform(cfg)
    h_stride, w_stride = 128, 128
    h_crop, w_crop = cfg.DATASETS.ORIGIN.HEIGHT, cfg.DATASETS.ORIGIN.WIDTH
    h_img, w_img, _ = image.shape
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    pred_whole_img = np.zeros([h_img, w_img], dtype=np.float32)
    count_mat = np.zeros([h_img, w_img])
    juncs_whole_img = []
    patch_weight = np.ones((h_crop + 2, w_crop + 2))
    patch_weight[0, :] = 0
    patch_weight[-1, :] = 0
    patch_weight[:, 0] = 0
    patch_weight[:, -1] = 0
    patch_weight = ndi.distance_transform_edt(patch_weight)
    patch_weight = patch_weight[1:-1, 1:-1]
    
    for h_idx in tqdm(range(h_grids), desc='processing on image'):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = image[y1:y2, x1:x2, :]
            crop_img_tensor = transform(crop_img.astype(float))[None].to(device)
            meta = {'height': crop_img.shape[0], 'width': crop_img.shape[1], 'pos': [x1, y1, x2, y2]}
            with torch.no_grad():
                output, _ = model(crop_img_tensor, [meta])
                output = to_single_device(output, 'cpu')
            juncs_pred = output['juncs_pred'][0]
            mask_pred = output['mask_pred'][0]
            juncs_pred += [x1, y1]
            juncs_whole_img.extend(juncs_pred.tolist())
            mask_pred *= patch_weight
            pred_whole_img += np.pad(mask_pred, ((int(y1), int(pred_whole_img.shape[0] - y2)), 
                                               (int(x1), int(pred_whole_img.shape[1] - x2))))
            count_mat[y1:y2, x1:x2] += patch_weight
    
    juncs_whole_img = np.array(juncs_whole_img)
    pred_whole_img = pred_whole_img / count_mat
    return juncs_whole_img, pred_whole_img

def get_pretrained_model_FT(cfg, dataset, device, path_model, pretrained=True):
    model = BuildingDetector(cfg, test=True)
    state_dict = torch.load(path_model)
    model.load_state_dict(state_dict["model"])
    model = model.eval()
    return model

def save_figure_as_tif(fig, ax, output_path):
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize and process GeoTIFF files")
    parser.add_argument("--image", type=str, help="Path to the GeoTIFF file")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for mask prediction")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size for processing")
    parser.add_argument("--stride", type=int, default=128, help="Stride for sliding window")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"GeoTIFF file not found: {args.image}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    if args.config is not None:
        cfg.merge_from_file(args.config)
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {args.checkpoint}")

    os.makedirs(args.output, exist_ok=True)
    src = rio_open(args.image)
    print("src:{}".format(src))
    width, height = src.width, src.height
    print("width:{}, height:{}".format(width, height))
    patchsize = args.patch_size
    width_new = int(width / patchsize) * patchsize
    height_new = int(height / patchsize) * patchsize
    window = Window(1, 1, width_new, height_new)
    transform = src.window_transform(window)
    profile = src.profile
    print("profile1:{}".format(profile))
    profile.update({'height': height_new, 'width': width_new, 'transform': transform})
    print("profile2:{}".format(profile))
    
    img_out_path = os.path.join(args.output, "image_test_cropped.tif")
    with rio_open(img_out_path, 'w', **profile) as dst:
        dst.write(src.read(window=window))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = io.imread(img_out_path)[:, :, :3]
    H, W = patchsize, patchsize
    cfg.DATASETS.ORIGIN.HEIGHT = 256 if H > 256 else H
    cfg.DATASETS.ORIGIN.WIDTH = 256 if W > 256 else W
    
    model = get_pretrained_model_FT(cfg, 'crowdai', device, args.checkpoint, pretrained=True)
    model = model.to(device)
    
    juncs_whole_img, pred_whole_img = inference_single(cfg, model, image, device)
    polygons = convert_mask_to_polygons(juncs_whole_img, pred_whole_img)
    
    path_out_img = os.path.join(args.output, "Pred_Mask.tif")
    with rasterio.open(img_out_path) as src:
        print("meta:{}".format(src.profile))
        print("crs:{}".format(src.crs))
        ras_meta = src.profile
        crs = src.crs
        ras_meta["count"] = 1
        ras_meta["dtype"] = "float32"
    print("pred_whole_img的形状：", pred_whole_img.shape)
    print("pred_whole_img的值为：", pred_whole_img)
    pred_whole_img_ = np.expand_dims(pred_whole_img[...], axis=0)
    print("pred_whole_img_的形状：", pred_whole_img_.shape)
    print("pred_whole_img_的值为：", pred_whole_img_.shape)
    with rasterio.open(path_out_img, 'w', **ras_meta) as dst:
        dst.write(pred_whole_img_)

    # 将像素坐标系的多边形转换为地理坐标系的多边形
    src = rasterio.open(img_out_path)
    left, bottom = src.bounds.left, src.bounds.bottom
    x_resolution = src.res[0]
    y_resolution = src.res[1]
    polygons_geo = []
    for poly in polygons:
        x, y = poly.exterior.coords.xy
        x_ = [i * x_resolution + left for i in x]
        y_ = [j * y_resolution + bottom for j in y]
        poly_geo = Polygon(zip(x_, y_))
        polygons_geo.append(poly_geo)
    
    polygons_gpd = gpd.GeoSeries(polygons_geo)
    origin = ((src.bounds.left + src.bounds.right) / 2, (src.bounds.top + src.bounds.bottom) / 2)
    flip = polygons_gpd.scale(xfact=1.0, yfact=-1.0, origin=origin)
    path_shp_out = os.path.join(args.output, "output_polygons.shp")
    gdf = gpd.GeoDataFrame(crs=src.crs, geometry=flip)
    gdf.to_file(path_shp_out, driver='ESRI Shapefile')
    
    fig, axs = plt.subplots()
    fig.set_size_inches(25, 25)
    axs.set_aspect('equal', 'datalim')
    for geom in flip:
        xs, ys = geom.exterior.xy
        axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')
    output_tif_path = os.path.join(args.output, "visualized_polygons.tif")
    save_figure_as_tif(fig, axs, output_tif_path)
    
    print(f"Processing complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
#     python vis_big_new.py \
#       --image data/tif/rgb1.tif \
#       --output outputs/redTide_hrnet48_new/test_big_new1 \
#       --config outputs/redTide_hrnet48_new/config.yml \
#       --checkpoint outputs/redTide_hrnet48_new/model_00050.pth \
#       --threshold 0.5 \
#       --patch_size 256 \
#       --stride 128