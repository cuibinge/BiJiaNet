import torch
from torch import nn
import torchvision.models as models

class HTQNet(nn.Module):

    def __init__(self, backbone="resnet18", hidden_size=256, num_layer=2, max_polygons=10, max_vertices=150):
        super(HTQNet, self).__init__()

        # 使用预训练的ResNet作为backbone
        self.backbone = models.resnet18(pretrained=True)
        # 修改输入通道为4适配tif图像
        self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 去掉全连接层
        self.backbone.fc = nn.Identity()
        # 去掉全局平均池化层（后边可以尝试）
        # self.backbone.avgpool = nn.Identity()

        # 多边形顶点预测头（不太理解）
        self.polygon_head = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
        self.vertex_predictor = nn.Linear(hidden_size, 2)
        self.hidden_to_input = nn.Linear(hidden_size, 512)  # 将LSTM输出映射回输入维度

        self.max_polygons = max_polygons  # 最大多边形数量
        self.max_vertices = max_vertices  # 每个多边形的最大顶点数量

    def forward(self, x):
        """
           前向传播
           :param x: 输入图像 (batch_size, 4, 256, 256)
           :param max_vertices: 最大顶点数量（用于控制推理时的输出长度）
           :return: 多边形顶点序列 (batch_size, seq_len, 2)
        """
        # 提取特征
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # 展平

        # 初始化LSTM的隐藏状态
        h0 = torch.zeros(self.polygon_head.num_layers, features.size(0), self.polygon_head.hidden_size).to(x.device)
        c0 = torch.zeros(self.polygon_head.num_layers, features.size(0), self.polygon_head.hidden_size).to(x.device)

        # 逐步生成多个多边形的顶点序列
        all_polygons = []
        for _ in range(self.max_polygons):
            vertices = []
            input = features.unsqueeze(1)  # 在1索引上增加一个维度
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
