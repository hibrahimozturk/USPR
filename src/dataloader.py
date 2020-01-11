import os
import torch
import glob

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import torchvision.transforms as transforms


class USPRDataset(Dataset):
    def __init__(self, imgDir, downsampleRate=2, imgSize=224):

        self.imgDir = imgDir
        self.imgSize = imgSize
        self.imgList = glob.glob(os.path.join(self.imgDir, "*.jpg"))
        self.downsampleRate = downsampleRate
        self.downsampleSize = int(self.imgSize/self.downsampleRate)

        self.transform = transforms.Compose([
            transforms.Resize(self.imgSize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        image_path = self.imgList[idx]
        img = Image.open(image_path).convert('RGB').resize((self.imgSize, self.imgSize))
        imgDownsample = img.resize((self.downsampleSize, self.downsampleSize))

        img = self.transform(img)
        imgDownsample = self.transform(imgDownsample)

        return imgDownsample, img


if __name__ == "__main__":
    usprData = USPRDataset("../data/coco/coco_train2014")
    usprLoader = DataLoader(usprData, batch_size=4, num_workers=4)

    for index, item in enumerate(usprLoader):
        print(index)
