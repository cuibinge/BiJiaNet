import os
import torch
import torchvision
import torch.nn as nn
from net4 import HTQNet
from dataset2 import RedTideDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

# 自定义 collate_fn
def collate_fn(batch):

    # 提取图像和标签
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # 将图像堆叠为张量
    images = torch.stack(images, dim=0)

    # 提取 targets 中的 boxes, labels, polygons, mask
    boxes = [t["boxes"] for t in targets]
    labels = [t["labels"] for t in targets]
    polygons = [t["polygons"] for t in targets]
    masks = [t["mask"] for t in targets]

    # 计算每个样本的目标数量
    num_objects = [len(b) for b in boxes]

    # 将 boxes 和 labels 堆叠为张量
    boxes = torch.cat(boxes, dim=0)
    labels = torch.cat(labels, dim=0)

    # 将 polygons 和 masks 堆叠为张量
    polygons = torch.cat(polygons, dim=0)
    masks = torch.cat(masks, dim=0)

    return images, {
        "boxes": boxes,  # 形状: (N, 4)，N 是 batch 中所有目标的总数
        "labels": labels,  # 形状: (N,)
        "polygons": polygons,  # 形状: (N, max_vertices, 2)
        "mask": masks,  # 形状: (N, max_vertices)
        "num_objects": num_objects  # 每个样本的目标数量
    }

# 损失函数
def compute_loss(detection_output, vertex_output, targets):
    print("Detection output shape:", detection_output.shape)
    print("Vertex output shape:", vertex_output.shape)
    print("Targets boxes shape:", targets["boxes"].shape)
    print("Targets labels shape:", targets["labels"].shape)
    print("Targets polygons shape:", targets["polygons"].shape)
    # 目标检测损失
    detection_loss = compute_detection_loss(detection_output, targets)
    # 顶点序列预测损失
    vertex_loss = compute_vertex_loss(vertex_output, targets)
    # 总损失
    total_loss = detection_loss + vertex_loss
    return total_loss

def compute_detection_loss(detection_output, targets):
    # 分类损失（交叉熵）
    print("-----------------------------")
    print(detection_output)
    print(targets)
    cls_loss = F.cross_entropy(detection_output["class_logits"], targets["labels"])
    # 边界框回归损失（Smooth L1）
    box_loss = F.smooth_l1_loss(detection_output["box_regression"], targets["boxes"])
    return cls_loss + box_loss

def compute_vertex_loss(vertex_output, targets):
    # 顶点坐标回归损失（L2）
    vertex_loss = F.mse_loss(vertex_output, targets["polygons"])
    return vertex_loss

# 超参数
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = "weights"

# 数据增强
# transform = Compose([
#     ToTensor(),  # 转换为 Tensor
#     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
# ])

# 加载数据集
train_dataset = RedTideDataset(
    image_dir="./data/train/images",
    json_dir="./data/train/jsons",
    transform=None
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 创建模型
backbone = torchvision.models.resnet50(pretrained=True)
backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 使其支持四通道输入
backbone = nn.Sequential(*list(backbone.children())[:-2])  # 去掉最后的全连接层和池化层
model = HTQNet(backbone, num_classes=2, num_vertices=20)
model.to(DEVICE)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("训练集的长度为：{}".format(len(train_dataset)))

# 开始训练
print("————————————————————————开始训练————————————————————————")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0.0
    print("————————————————————————Epoch:{}————————————————————————".format(epoch + 1))
    for i, (images, targets) in enumerate(train_loader):
        print("Images shape:", images.shape)  # 形状: (batch_size, 4, H, W)
        print("Boxes shape:", targets["boxes"].shape)  # 形状: (N, 4)
        print("Labels shape:", targets["labels"].shape)  # 形状: (N,)
        print("Polygons shape:", targets["polygons"].shape)  # 形状: (N, max_vertices, 2)
        print("Mask shape:", targets["mask"].shape)  # 形状: (N, max_vertices)
        print("Num objects:", targets["num_objects"])  # 每个样本的目标数量
        # 将数据移动到设备
        images = images.to(DEVICE)
        targets = {
            "boxes": targets["boxes"].to(DEVICE),
            "labels": targets["labels"].to(DEVICE),
            "polygons": targets["polygons"].to(DEVICE),
            "mask": targets["mask"].to(DEVICE),
        }
        # 前向传播
        detection_output, vertex_output = model(images)
        # 计算损失
        loss = compute_loss(detection_output, vertex_output, targets)
        total_train_loss += loss.item()
        # 清零梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()
        print("Batch: {}/{}, Batch_loss: {}".format(i + 1, len(train_loader), loss))
    epoch_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

    # 保存模型检查点
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(save_path, "epoch{}_loss{}.pt".format(epoch + 1, epoch_loss)))
