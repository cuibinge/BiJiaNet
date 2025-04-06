import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Dict, Optional
from torchvision.ops import FeaturePyramidNetwork, roi_align

class PolygonPredictionLayer(nn.Module):
    """单阶段多边形预测层"""
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.num_corners = config['num_corners']

        # 注意力机制
        self.self_attn = nn.MultiheadAttention(self.hidden_dim, num_heads=8)
        self.cross_attn = nn.MultiheadAttention(self.hidden_dim, num_heads=8)

        # 预测头
        self.polygon_pred = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_corners * 2)  # 输出顶点坐标
        )
        self.score_pred = nn.Linear(self.hidden_dim, 1)

    def forward(self, features, query_features):
        print(f"输入特征形状 - features: {features.shape}, query_features: {query_features.shape}")

        # query_features处理: [batch_size, C, H, W] -> [batch_size, H*W, C]
        if query_features.dim() == 4:
            # [batch_size, C, H, W] -> [H*W, batch_size, C]
            query_features = query_features.flatten(2).permute(2, 0, 1)  # [256, 2, 256]

        # 自注意力计算
        query_features = self.self_attn(query_features, query_features, query_features)[0]  # [256, 2, 256]

        # features处理: [sum(K), C, 7, 7] -> [sum(K), 49, C]
        if features.dim() == 4:
            features = features.flatten(2).permute(0, 2, 1)  # [200, 49, 256]

        # 按batch分割features
        proposal_counts = [features.size(0) // query_features.size(1)] * query_features.size(1)
        features_split = torch.split(features, proposal_counts, dim=0)  # List[[100, 49, 256], [100, 49, 256]]

        # 初始化输出s
        all_polygons = []
        all_scores = []
        # 逐batch计算交叉注意力
        for i in range(query_features.size(1)):  # 遍历batch
            # 当前batch的数据
            batch_query = query_features[:, i, :]  # [256, 256]
            batch_features = features_split[i]  # [100, 49, 256]

            # 扩展query维度以匹配proposals数量
            batch_query = batch_query.unsqueeze(1).expand(-1, batch_features.size(0), -1)  # [256, 100, 256]

            # 调整维度顺序: [seq_len, batch_size, embed_dim]
            batch_features = batch_features.permute(1, 0, 2)  # [49, 100, 256]

            # 交叉注意力: query来自RPN, key/value来自ROI
            attn_output = self.cross_attn(
                query=batch_query,  # [256, 100, 256]
                key=batch_features,  # [49, 100, 256]
                value=batch_features  # [49, 100, 256]
            )[0]  # [256, 100, 256]

            # 转回原始维度
            attn_output = attn_output.permute(1, 0, 2)  # [100, 256, 256]

            # 预测
            polygons = self.polygon_pred(attn_output)  # [100, 256, num_corners*2]
            scores = torch.sigmoid(self.score_pred(attn_output))  # [100, 256, 1]

            # 平均池化处理
            polygons = polygons.mean(dim=1)  # [100, num_corners*2]
            scores = scores.mean(dim=1)  # [100, 1]

            all_polygons.append(polygons)
            all_scores.append(scores)

        # 合并结果
        polygons = torch.cat(all_polygons, dim=0)  # [200, num_corners*2]
        scores = torch.cat(all_scores, dim=0)  # [200, 1]

        return polygons, scores

class ROIPool(nn.Module):
    """ROI对齐池化层"""
    def __init__(self, output_size=7):
        super().__init__()
        self.output_size = output_size

    def forward(self, features, proposals):
        # boxes格式: [batch_idx, x1, y1, x2, y2]
        """
        参数:
            features: 特征图 [N, C, H, W]
            proposals: List[Tensor[K, 5]], 每个元素格式为[batch_idx, x1, y1, x2, y2]
        返回:
            pooled_features: 池化后的特征 [sum(K), C, output_size, output_size]
        """

        # 1. 分离boxes和batch索引
        boxes_list = []
        for proposal in proposals:
            # 检查坐标是否合法
            assert (proposal[:, 1:] >= 0).all(), "坐标包含负数！"
            assert (proposal[:, 3] > proposal[:, 1]).all() and (proposal[:, 4] > proposal[:, 2]).all(), "框宽高无效！"
            # 确保proposal是5列（batch_idx+坐标）
            assert proposal.size(1) == 5, f"Expected [K,5], got {proposal.shape}"
            # 提取坐标部分（去除batch_idx）
            boxes_list.append(proposal[:, 1:])  # [K,4]

        # 2. 验证boxes格式
        assert all(boxes.dim() == 2 and boxes.size(1) == 4 for boxes in boxes_list), \
            "Boxes must be List[Tensor[L,4]]"

        # 3. 执行ROI Align
        return roi_align(
            input=features,
            boxes=boxes_list,
            output_size=self.output_size,
            spatial_scale=1.0 / 4.0  # 根据实际特征图下采样率调整, P2特征图的步长
        )

class PolygonHead(nn.Module):
    """多边形预测头（基于变形注意力）"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.get('num_heads', 6)
        self.roi_pool = ROIPool(output_size=7)

        # 多阶段预测头
        self.heads = nn.ModuleList([
            PolygonPredictionLayer(config) for _ in range(self.num_heads)
        ])

    def forward(self, features, proposals, proposal_features):
        # ROI对齐
        pooled_features = self.roi_pool(features['p2'], proposals)

        # 多阶段迭代优化
        polygon_preds = []
        for head in self.heads:
            polygons, scores = head(pooled_features, proposal_features)
            polygon_preds.append((polygons, scores))
            # 更新特征用于下一阶段
            proposal_features = self.update_features(polygons, proposal_features)

        # 返回最终预测
        return polygon_preds[-1]

class AnchorGenerator:
    """锚框生成器"""
    def __init__(self):
        self.sizes = [32, 64, 128]  # 锚框基础尺寸
        self.aspect_ratios = [0.5, 1.0, 2.0]  # 宽高比
        self.scales = [1.0, 2 ** (1 / 3), 2 ** (2 / 3)]  # 多尺度缩放
        self.num_proposals = 100  # 添加提案数量控制
        self.min_box_size = 1.0

    def generate_anchors(self, feature_map_size):
        """生成基础锚框"""
        anchors = []
        for size in self.sizes:
            for ratio in self.aspect_ratios:
                for scale in self.scales:
                    w = size * scale * math.sqrt(ratio)
                    h = size * scale / math.sqrt(ratio)
                    # 确保初始锚框尺寸足够大
                    w = max(w, self.min_box_size)
                    h = max(h, self.min_box_size)
                    anchors.append([-w / 2, -h / 2, w / 2, h / 2])  # 中心点坐标格式
        return torch.tensor(anchors)  # [num_anchors, 4]

    def generate_proposals(self, bbox_deltas, objectness, image_size=(256, 256)):
        """
        生成最终候选区域
        参数:
            bbox_deltas: [N, A*4, H, W] 边界框回归参数
            objectness: [N, A, H, W] 目标置信度
            image_size: 输入图像尺寸
        返回:
            proposals: List[Tensor[K, 5]] 每个元素格式为[batch_idx, x1, y1, x2, y2]
        """
        N, _, H, W = bbox_deltas.shape
        A = len(self.sizes) * len(self.aspect_ratios) * len(self.scales)

        # 正确的形状变换
        bbox_deltas = bbox_deltas.view(N, A, 4, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, 4)
        objectness = objectness.view(N, A, H, W).permute(0, 2, 3, 1).reshape(N, -1)

        base_anchors = self.generate_anchors((H, W)).to(bbox_deltas.device)  # [A,4]

        # 1. 生成网格锚点
        shift_x = (torch.arange(0, W) + 0.5) * (image_size[0] / W)
        shift_y = (torch.arange(0, H) + 0.5) * (image_size[1] / H)
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1)  # [H,W,4]

        # 2. 生成所有锚框（保持batch维度）
        # all_anchors = base_anchors.view(1, A, 4) + shifts.view(H, W, 1, 4)  # [H,W,A,4]
        all_anchors = (base_anchors.view(1, 1, A, 4) + shifts.view(1, H, W, 1, 4))  # [1,H,W,A,4]
        # all_anchors = all_anchors.permute(2, 0, 1, 3).reshape(-1, 4)  # [H*W*A,4]
        all_anchors = all_anchors.clamp(min=0)  # 确保坐标不小于0
        all_anchors[..., 2:] = all_anchors[..., 2:].clamp(max=image_size[0])  # 确保不超过图像尺寸
        all_anchors = all_anchors.expand(N, -1, -1, -1, -1).reshape(N, -1, 4)  # [N,H*W*A,4]

        # 3. 应用边界框回归
        proposals = self.apply_deltas(bbox_deltas, all_anchors)

        # 4. 筛选TopK前检查有效性
        valid_mask = (proposals[..., 2] > proposals[..., 0] + self.min_box_size) & \
                     (proposals[..., 3] > proposals[..., 1] + self.min_box_size)

        # 对于无效框，将objectness设为极小值，使其不会被选中
        objectness[~valid_mask] = -float('inf')

        # 5. 添加batch索引并筛选TopK
        topk_idx = torch.topk(objectness, k=self.num_proposals, dim=1)[1]  # [N,K]

        final_proposals = []
        for i in range(N):
            batch_proposals = proposals[i][topk_idx[i]]  # [K,4]
            valid = (batch_proposals[:, 2] > batch_proposals[:, 0] + self.min_box_size) & \
                    (batch_proposals[:, 3] > batch_proposals[:, 1] + self.min_box_size)
            batch_proposals = batch_proposals[valid]
            batch_idx = torch.full((len(batch_proposals), 1), i, device=batch_proposals.device)
            final_proposals.append(torch.cat([batch_idx, batch_proposals], dim=1))
        return final_proposals  # List[[K,5], ...]

    def apply_deltas(self, deltas, anchors):
        """应用边界框变换"""
        assert deltas.shape == anchors.shape
        widths = anchors[..., 2] - anchors[..., 0]  # [N,K]
        heights = anchors[..., 3] - anchors[..., 1]
        ctr_x = anchors[..., 0] + 0.5 * widths
        ctr_y = anchors[..., 1] + 0.5 * heights

        dx = deltas[..., 0] / 10.0  # [N,K]
        dy = deltas[..., 1] / 10.0
        dw = deltas[..., 2] / 5.0
        dh = deltas[..., 3] / 5.0

        pred_boxes = torch.zeros_like(deltas)
        # 计算预测框坐标
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        # 确保预测框尺寸不小于最小值
        pred_w = torch.clamp(pred_w, min=self.min_box_size)
        pred_h = torch.clamp(pred_h, min=self.min_box_size)

        pred_boxes[..., 0] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[..., 1] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[..., 2] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[..., 3] = pred_ctr_y + 0.5 * pred_h  # y2

        # 强制约束坐标范围
        pred_boxes[..., 0::2] = pred_boxes[..., 0::2].clamp(min=0, max=256)
        pred_boxes[..., 1::2] = pred_boxes[..., 1::2].clamp(min=0, max=256)

        # 再次确保宽度和高度有效
        invalid_mask = (pred_boxes[..., 2] - pred_boxes[..., 0] < self.min_box_size) | \
                       (pred_boxes[..., 3] - pred_boxes[..., 1] < self.min_box_size)

        # 对于无效框，恢复原始锚框
        pred_boxes[invalid_mask] = anchors[invalid_mask]
        return pred_boxes

class RegionProposalNetwork(nn.Module):
    """区域提议网络（RPN）"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.anchor_generator = AnchorGenerator()
        self.proposal_heads = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cls_logits = nn.Conv2d(hidden_dim, 27, kernel_size=1)
        self.bbox_pred = nn.Conv2d(hidden_dim, 4 * 27, kernel_size=1)

    def forward(self, features):
        # 使用P4特征图生成提议
        p4_features = features['p4']
        features = self.proposal_heads(p4_features)

        # 预测分类和回归
        logits = self.cls_logits(features)
        bbox_deltas = self.bbox_pred(features)

        # 生成候选区域
        proposals = self.anchor_generator.generate_proposals(bbox_deltas, logits)
        return proposals, features


class BackboneWithFPN(nn.Module):
    """带FPN的骨干网络（支持多尺度特征提取）"""
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        super().__init__()
        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )
        self.return_layers = return_layers

    def forward(self, x):
        # 标准ResNet前向传播
        x = self.body.conv1(x)
        x = self.body.bn1(x)
        x = self.body.relu(x)
        x = self.body.maxpool(x)

        # 获取各阶段特征
        features = {}
        x = self.body.layer1(x);
        features['p2'] = x
        x = self.body.layer2(x);
        features['p3'] = x
        x = self.body.layer3(x);
        features['p4'] = x
        x = self.body.layer4(x);
        features['p5'] = x

        # FPN特征融合
        return self.fpn(features)

class HTQNet(nn.Module):
    def __init__(self, config):
        super(HTQNet, self).__init__()
        # 设备配置
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # 模型超参数
        self.num_classes = config.get('num_classes', 1)  # 二分类（赤潮/背景） 默认为1
        self.num_proposals = config.get('num_proposals', 100)  # 候选区域数量 默认为100
        self.num_corners = config.get('num_corners', 64)  # 多边形顶点数（24个点x,y坐标） 默认为64
        self.hidden_dim = config.get('hidden_dim', 256)  # 特征维度 默认为256
        self.backbone_name = config.get('backbone', 'resnet50')  # 默认骨干网络 默认为resnet50

        # 核心组件
        self.backbone = self.build_backbone()  # 骨干网络
        self.rpn = RegionProposalNetwork(self.hidden_dim)  # 区域提议网络
        self.polygon_head = PolygonHead(config)  # 多边形预测头

        # 初始化参数
        self.init_weights()
        # self.to(self.device)

    def build_backbone(self):
        """构建骨干网络（支持4通道输入）"""
        if self.backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            # 修改第一层卷积支持4通道输入
            backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            return_layers = {'layer1': 'p2', 'layer2': 'p3', 'layer3': 'p4', 'layer4': 'p5'}
            return BackboneWithFPN(backbone, return_layers, in_channels_list=[256, 512, 1024, 2048], out_channels=self.hidden_dim)
        else:
            raise ValueError(f"不支持的骨干网络: {self.backbone_name}")

    def init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images):
        """
        输入:
            images - 4通道遥感图像 [batch_size, 4, H, W]
        输出:
            proposals - 候选区域框 [batch_size, num_proposals, 4] (xyxy格式)
            polygons - 预测多边形 [batch_size, num_proposals, num_corners*2]
            scores - 预测置信度 [batch_size, num_proposals]
        """
        # 特征提取
        features = self.backbone(images)  # 多尺度特征 {"p2":..., "p3":..., "p4":..., "p5":...}

        # 生成候选区域
        proposals, proposal_features = self.rpn(features)

        # 预测多边形
        polygons, scores = self.polygon_head(features, proposals, proposal_features)

        return proposals, polygons, scores

if __name__ == '__main__':
    # 示例配置
    config = {
        'device': 'cuda',
        'num_classes': 1,
        'num_proposals': 100,
        'num_corners': 64,
        'hidden_dim': 256,
        'backbone': 'resnet50'
    }
    # 初始化模型
    model = HTQNet(config)
    # 测试输入
    dummy_input = torch.rand(2, 4, 256, 256)
    # 前向传播
    proposals, polygons, scores = model(dummy_input)
    print(f"输出形状 - 候选框: {proposals.shape}, 多边形: {polygons.shape}, 分数: {scores.shape}")



