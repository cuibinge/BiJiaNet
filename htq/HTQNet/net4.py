import torch
import torchvision
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

class VertexPredictionBranch(nn.Module):
    """LSTM顶点序列预测分支"""
    def __init__(self, in_channels, num_vertices):
        super(VertexPredictionBranch, self).__init__()
        self.num_vertices = num_vertices
        # 将 RoI 特征映射为 LSTM 的输入
        self.fc1 = nn.Linear(in_channels * 7 * 7, 512)
        # LSTM 层
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        # 输出层
        self.fc2 = nn.Linear(256, 2)  # 每个顶点有 (x, y) 两个坐标

    def forward(self, x):
        # 展平 RoI 特征
        x = x.flatten(start_dim=1)
        # 全连接层
        x = F.relu(self.fc1(x))
        # 扩展为序列输入 (N, 1, 512)
        x = x.unsqueeze(1)
        # 重复输入以匹配顶点数量 (N, num_vertices, 512)
        x = x.repeat(1, self.num_vertices, 1)
        # LSTM 层
        lstm_out, _ = self.lstm(x)  # 输出形状 (N, num_vertices, 256)
        # 输出层
        vertices = self.fc2(lstm_out)  # 输出形状 (N, num_vertices, 2)
        return vertices

class HTQNet(nn.Module):

    def __init__(self, backbone, num_classes, num_vertices):
        super(HTQNet, self).__init__()
        self.backbone = backbone
        # 自定义 AnchorGenerator 参数
        anchor_sizes = ((16, 32, 64, 128, 256),)  # 单层嵌套元组
        aspect_ratios = ((0.5, 1.0, 2.0),)  # 单层嵌套元组
        self.rpn = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2)
        self.detection_head = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes + 4)  # 分类 + 边界框回归
        )
        self.vertex_head = VertexPredictionBranch(2048, num_vertices)

    def forward(self, images):
        # 提取特征
        features = self.backbone(images)
        # 获取每张图像的尺寸 (height, width)
        image_sizes = [(image.shape[1], image.shape[2]) for image in images]  # 格式: List[Tuple[int, int]]
        # 将 images 和 image_sizes 包装为 ImageList
        image_list = ImageList(images, image_sizes)
        # 将 features 包装为列表
        feature_maps = [features]
        # 生成候选区域
        proposals = self.rpn(image_list, feature_maps)
        # RoI Align
        rois = self.roi_align(features, proposals)
        # 将 rois 展平为 [N, 2048 * 7 * 7]
        rois_flattened = rois.flatten(start_dim=1)
        # 目标检测分支
        detection_output = self.detection_head(rois_flattened)
        # 顶点序列预测分支
        vertex_output = self.vertex_head(rois)
        return detection_output, vertex_output

    def inference(self, images, score_threshold=0.5, nms_threshold=0.3):
        """
            推理阶段，输出多个多边形
            :param images: 输入图像
            :param score_threshold: 置信度阈值
            :param nms_threshold: NMS 阈值
            :return: 多边形列表
        """
        # 前向传播
        detection_output, vertex_output = self.forward(images)
        # 解析检测结果
        boxes = detection_output[:, :4]  # 边界框
        scores = detection_output[:, 4]  # 置信度分数
        polygons = vertex_output  # 多边形顶点序列

        # 筛选置信度高于阈值的预测
        keep = scores > score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        polygons = polygons[keep]

        # 非极大值抑制 (NMS)
        keep = ops.nms(boxes, scores, nms_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        polygons = polygons[keep]

        # 构建输出结果
        results = []
        for box, polygon, score in zip(boxes, polygons, scores):
            results.append({
                "bbox": box.tolist(),  # 边界框
                "polygon": polygon.tolist(),  # 多边形顶点序列
                "score": score.item()  # 置信度分数
            })

        return results

if __name__ == '__main__':
    backbone = torchvision.models.resnet50(pretrained=True)
    backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 使其支持四通道输入
    backbone = nn.Sequential(*list(backbone.children())[:-2])  # 去掉最后的全连接层和池化层
    model = HTQNet(backbone, num_classes=2, num_vertices=20)  # 2 类（赤潮 vs 背景），20 个顶点
    print(model)