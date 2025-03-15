import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

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

    def __init__(self, backbone="resnet18", hidden_size=256, num_layer=2, max_polygons=10, max_vertices=150):
        super(HTQNet, self).__init__()

        # 使用 FPN 作为 backbone
        self.fpn = FPN(backbone=backbone, out_channels=256)
        self.hidden_size = hidden_size
        self.max_polygons = max_polygons
        self.max_vertices = max_vertices

        # 多边形顶点预测头（LSTM）
        self.polygon_head = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
        self.vertex_predictor = nn.Linear(hidden_size, 2)
        self.hidden_to_input = nn.Linear(hidden_size, 256)  # 将LSTM输出映射回输入维度

    def forward(self, x):

        # 提取多尺度特征
        p2, p3, p4, p5 = self.fpn(x)

        # 使用最高分辨率的特征图（P2）作为输入
        features = p2  # (batch_size, 256, H/4, W/4)

        # 将特征图展平为序列
        batch_size, channels, H, W = features.shape
        features = features.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, H*W, 256)

        # 初始化LSTM的隐藏状态
        h0 = torch.zeros(self.polygon_head.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.polygon_head.num_layers, batch_size, self.hidden_size).to(x.device)

        # 逐步生成多个多边形的顶点序列
        all_polygons = []
        for _ in range(self.max_polygons):
            vertices = []
            input = features.mean(dim=1, keepdim=True)  # 使用特征图的均值作为初始输入
            for _ in range(self.max_vertices):
                # LSTM前向传播
                output, (h0, c0) = self.polygon_head(input, (h0, c0))  # output: (batch_size, 1, hidden_size)
                # 预测顶点坐标
                vertex = self.vertex_predictor(output.squeeze(1))  # (batch_size, 2)
                vertices.append(vertex)
                # 将当前顶点作为下一个时间步的输入
                input = self.hidden_to_input(output)
            # 拼接顶点序列
            vertices = torch.stack(vertices, dim=1)  # (batch_size, max_vertices, 2)
            all_polygons.append(vertices)
        # 拼接多个多边形
        all_polygons = torch.stack(all_polygons, dim=1)  # (batch_size, max_polygons, max_vertices, 2)
        return all_polygons

if __name__ == '__main__':
    model = HTQNet()
    print(model)
