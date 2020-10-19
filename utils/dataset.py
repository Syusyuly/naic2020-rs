import os
import os.path
import cv2
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
matches = [100, 200, 300, 400, 500, 600, 700, 800]

class Mydataset(Dataset):
    def __init__(self, data_root=None, path=None, transform=None):
        data = pd.read_csv(path)
        imgs = []
        for i in range(1,len(data)):
            imgs.append((data_root + data.iloc[i,1], data_root + data.iloc[i,2]))
        self.imgs = imgs
        self.transform = transform


    def __getitem__(self, item):
        image_path, label_path = self.imgs[item]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)  # GRAY 1 channel ndarray with shape H * W
        for m in matches:
            label[label == m] = matches.index(m)
        label = np.float32(label)
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    normMean = [0.46099097, 0.32533738, 0.32106236]
    normStd = [0.20980413, 0.1538582, 0.1491854]
    normTransfrom = transforms.Normalize(normMean, normStd)
    transform = transforms.Compose([
        transforms.ToTensor(),
        normTransfrom,
    ])
    train_data = Mydataset(path='../dataset/train_path_list.csv', transform=transform)
    img, gt = train_data.__getitem__(0)
    print(img.shape,gt)
