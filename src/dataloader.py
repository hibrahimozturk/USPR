import os
import glob

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import torchvision.transforms as transforms


class USPRDataset(Dataset):
    def __init__(self, imgDir, downsampleRate=2, imgSize=224, topK=None):

        self.imgDir = imgDir
        self.imgSize = imgSize
        self.imgList = glob.glob(os.path.join(self.imgDir, "*.jpg"))
        self.downsampleRate = downsampleRate
        self.downsampleSize = int(self.imgSize/self.downsampleRate)
        self.topK = topK

        self.inputTransform = transforms.Compose([
            transforms.Resize(self.imgSize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transform = transforms.Compose([
            transforms.Resize(self.imgSize),
            transforms.ToTensor(),
        ])

    def __len__(self):
        if self.topK is None:
            return len(self.imgList)
        else:
            return self.topK

    def __getitem__(self, idx):
        image_path = self.imgList[idx]
        img = Image.open(image_path).convert('RGB').resize((self.imgSize, self.imgSize))
        imgDownsample = img.resize((self.downsampleSize, self.downsampleSize))

        img = self.transform(img)
        imgDownsample = self.inputTransform(imgDownsample)

        return imgDownsample, img


if __name__ == "__main__":
    usprData = USPRDataset("../data/coco/coco_train2014")
    usprLoader = DataLoader(usprData, batch_size=4, num_workers=4)

    for index, item in enumerate(usprLoader):
        print(index)
