import torch
import os
from PIL import Image
import numpy as np
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from net import *


label_dict = {'butterfly': 0,
              'cat': 1,
              'chicken': 2,
              'cow': 3,
              'dog': 4,
              'elephant': 5,
              'horse': 6,
              'sheep': 7,
              'spider': 8,
              'squirrel': 9}


class Animal_Test_Dataset(Dataset):
    def __init__(self, root):
        self.dataset = []
        with open(root + '/test_label.txt', 'r') as file:
            lines = file.readlines()
            for i in range(len(lines)):
                file_name, tag = lines[i].split(',')
                path = root + "/test/" + file_name
                self.dataset.append((path, tag[:-1]))

    def __len__(self):
        # 返回数据集的长度
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        # 获取图像的路径
        path = data[0]
        # 获取图像的标签
        tag = data[1]
        # 利用opencv读取图片
        #img = cv2.imread(path)
        img = Image.open(path).convert('RGB')

        data_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.504, 0.504, 0.383), std=(0.312, 0.294, 0.327))
        ])

        img_totensor = data_transform(img)

        return img_totensor, int(label_dict[tag])


if __name__ == '__main__':
    dataset_test = Animal_Test_Dataset('animal data')
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)
    net = Animal_Net()
    net.load_state_dict(torch.load('net2_train_118.pth', map_location='cpu'))

    sum_score = 0.0
    for i, (img, label) in enumerate(dataloader_test):
        net.eval()
        test_out = net(img)
        softmax = nn.Softmax(dim=1)
        test_out = softmax(test_out)
        pre = torch.argmax(test_out, dim=1)
        score = torch.mean(torch.eq(pre, label).float())
        sum_score = sum_score + score

    # 求平均分
    test_avg_score = sum_score / len(dataloader_test)

    print("测试得分：", test_avg_score)
