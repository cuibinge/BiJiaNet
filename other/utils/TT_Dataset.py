import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
import imageio
import glob
import os


class MyDataset(Dataset):
    def __init__(self, images_path, labels_path, Transform=None):
        """"""
        # 在这里写，获得所有image路径，所有label路径的代码，并将路径放在分别放在images_path_list和labels_path_list中
        """"""
        self.images_path_list = glob.glob(os.path.join(images_path, '*.tif'))
        self.labels_path_list = glob.glob(os.path.join(labels_path, '*.tif'))
        # self.transform = ToTensor()

    def __getitem__(self, index):
        image_path = self.images_path_list[index]
        label_path = self.labels_path_list[index]

        # print(image_path,label_path)

        image = imageio.imread(image_path)
        label = imageio.imread(label_path)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        #         image = torch.permute(image, [2, 0, 1])
        image = image.permute(2, 0, 1)

        # 4：tansform 参数一般为 transforms.ToTensor()，意思是上步image,label 转换为 tensor 类型

        #         if self.transform is not None:
        #             image = self.transform(image)
        #             label = self.transform(label)

        # print(image.shape)
        # print(label.shape)
        label = torch.squeeze(label, 0)

        # label = torch.squeeze(label, 0)
        # print(label.shape)
        # label = one_hot(label.long(), num_classes=10)
        # label = torch.squeeze(label, 0)
        # label = np.transpose(label, ( 2, 0, 1))

        return image, label

    def __len__(self):
        return len(self.images_path_list)


def main():
    imagePath = r"/sunhuan/20230408/Dataset/cut_train_images"
    labelPath = r"/sunhuan/20230408/Dataset/cut_train_labels"
    mydataset = MyDataset(imagePath, labelPath)

    # dataset = MyDataset('.\data_npy\img_covid_poisson_glay_clean_BATCH_64_PATS_100.npy')

    Data = DataLoader(mydataset, batch_size=1, shuffle=False, pin_memory=True)
    for i, data in enumerate(Data):
        img, lab = data
        print(img.shape)
        print(lab.shape)


if __name__ == '__main__':
    main()
