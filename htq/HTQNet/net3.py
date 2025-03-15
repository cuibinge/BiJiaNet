import math

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torchvision import transforms
from dataset import RedTideDataset
from dataset import custom_collate_fn
from torch.utils.data import Dataset, DataLoader
from timm.models.vision_transformer import VisionTransformer

class PolygonTransformer(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=256, num_heads=8, num_layers=4, max_vertices=150):
        super(PolygonTransformer, self).__init__()
        self.max_vertices = max_vertices

        # 线性投影层，将输入特征映射到隐藏维度
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 位置编码
        # self.positional_encoding = nn.Parameter(torch.zeros(1, max_vertices, hidden_dim))

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 顶点坐标预测层
        self.vertex_predictor = nn.Linear(hidden_dim, 2)

    def forward(self, x):

        # x形状为 (batch_size, num_vertices, input_dim)
        batch_size, num_vertices, _ = x.shape

        # 线性投影
        x = self.input_proj(x)  # (batch_size, num_vertices, hidden_dim)

        # 动态生成位置编码
        positional_encoding = torch.zeros(num_vertices, x.size(-1), device=x.device)  # (num_vertices, hidden_dim)
        position = torch.arange(0, num_vertices, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, x.size(-1), 2, dtype=torch.float, device=x.device) * (-math.log(10000.0) / x.size(-1)))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)  # (1, num_vertices, hidden_dim)

        # 添加位置编码
        x = x + positional_encoding

        # 调整形状以适应 Transformer 的输入 (seq_len, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)  # (num_vertices, batch_size, hidden_dim)

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)  # (num_vertices, batch_size, hidden_dim)

        # 调整形状为 (batch_size, num_vertices, hidden_dim)
        x = x.permute(1, 0, 2)

        # 预测顶点坐标
        vertices = self.vertex_predictor(x)  # (batch_size, num_vertices, 2)

        return vertices

class FPN(nn.Module):

    def __init__(self, backbone="resnet18", out_channels=256):
        super(FPN, self).__init__()

        # 加载预训练的 ResNet 作为 backbone
        self.backbone = getattr(models, backbone)(pretrained=True)
        self.out_channels = out_channels

        # 修改 ResNet 的输入通道为 4（适配 4 通道的遥感图像）
        self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 获取 ResNet 的中间层输出
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

        # 横向连接的 1x1 卷积
        self.lateral4 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(64, out_channels, kernel_size=1)

        # 自上而下的 3x3 卷积
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):

        # 提取 ResNet 的中间层特征
        c1 = self.backbone.conv1(x)
        c1 = self.backbone.bn1(c1)
        c1 = self.backbone.relu(c1)
        c1 = self.backbone.maxpool(c1)

        c2 = self.layer1(c1)  # C2
        c3 = self.layer2(c2)  # C3
        c4 = self.layer3(c3)  # C4
        c5 = self.layer4(c4)  # C5

        # 自上而下路径
        p5 = self.lateral4(c5)  # P5
        p4 = self.lateral3(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")  # P4
        p3 = self.lateral2(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")  # P3
        p2 = self.lateral1(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")  # P2

        # 平滑处理
        p5 = self.smooth4(p5)
        p4 = self.smooth3(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth1(p2)

        return p2, p3, p4, p5


class HTQNet(nn.Module):

    def __init__(self, backbone="resnet18", hidden_size=256, num_layer=4, max_polygons=10, max_vertices=150):
        super(HTQNet, self).__init__()

        # 使用 FPN 作为 backbone
        self.fpn = FPN(backbone=backbone, out_channels=256)
        self.hidden_size = hidden_size
        self.max_polygons = max_polygons
        self.max_vertices = max_vertices

        # 多边形顶点预测头（自定义的 PolygonTransformer）
        self.polygon_transformer = PolygonTransformer(
            input_dim=256,  # 输入特征维度
            hidden_dim=hidden_size,  # 隐藏层维度
            num_heads=8,  # 注意力头数
            num_layers=num_layer,  # Transformer 层数
            max_vertices=max_vertices  # 最大顶点数
        )

    def forward(self, x):

        # 提取多尺度特征
        p2, p3, p4, p5 = self.fpn(x)

        # 使用最高分辨率的特征图（P2）作为输入
        features = p2  # (batch_size, 256, H/4, W/4)

        # 将特征图展平为序列
        batch_size, channels, H, W = features.shape
        features = features.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, H*W, 256)
        print(features.shape)
        # 逐步生成多个多边形的顶点序列
        all_polygons = []
        for _ in range(self.max_polygons):
            vertices = self.polygon_transformer(features)  # (batch_size, max_vertices, 2)
            all_polygons.append(vertices)

        # 拼接多个多边形
        all_polygons = torch.stack(all_polygons, dim=1)  # (batch_size, max_polygons, max_vertices, 2)
        return all_polygons

if __name__ == '__main__':
    model = HTQNet()
    print("模型架构：", model)
    dataset = RedTideDataset(image_dir='data/train/images', annotation_dir='data/train/labels',
                             transform=transforms.ToTensor())
    print("训练集的长度为：{}".format(len(dataset)))
    dataloader = DataLoader(dataset, 2, collate_fn=custom_collate_fn)
    for imgs, labels in dataloader:
        print("图像形状：", imgs.shape)
        print("标签数量：", len(labels))
        outputs = model(imgs)
        print("预测结果：", outputs)
