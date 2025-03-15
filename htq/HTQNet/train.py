import os
import torch
import torchvision
from torch import nn
from net2 import HTQNet
from dataset import RedTideDataset
from dataset import custom_collate_fn
from torch.utils.data import DataLoader
from utils.compute_loss import compute_loss
from torchvision.transforms import transforms

mean_list = [358.56197836, 278.51276384, 146.58589797, 48.23313843]
std_list = [28.13160621, 41.51963463, 18.1556377, 9.82935977]

model = HTQNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = RedTideDataset(image_dir='data/train/images', annotation_dir='data/train/labels', transform=transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=custom_collate_fn)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
epoches = 100
save_path = "weights"

# 判断权重保存路径是否存在
if not os.path.exists(save_path):
    os.mkdir(save_path)

print("训练集的长度为：{}".format(len(train_dataset)))

# 开始训练
print("————————————————————————开始训练————————————————————————")
for epoch in range(epoches):
    loss_total = 0
    print("————————————————————————Epoch:{}————————————————————————".format(epoch+1))
    for i, (images, targets) in enumerate(train_dataloader):
        outputs = model(images)
        losses = 0
        for output, target in zip(outputs, targets):
            losses += compute_loss(output, target, criterion)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        loss_total += losses
        print("Batch: {}/{}, Batch_loss: {}".format(i + 1, len(train_dataloader), losses))
    avg_loss = loss_total / len(train_dataloader)
    print(f"Epoch: {epoch + 1}, Loss: {avg_loss}")
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(save_path, "epoch{}_loss{}.pt".format(epoch + 1, avg_loss)))
