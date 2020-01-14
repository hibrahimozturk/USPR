import os
import glob

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import torchvision.transforms as transforms


class USPRDataset(Dataset):
    def __init__(self, imgDir, train=False, downsampleRate=2, imgSize=224, topK=None):

        self.imgDir = imgDir
        self.imgSize = imgSize
        self.imgList = glob.glob(os.path.join(self.imgDir, "*.jpg"))
        self.imgList += glob.glob(os.path.join(self.imgDir, "*.png"))
        self.downsampleRate = downsampleRate
        self.downsampleSize = int(self.imgSize/self.downsampleRate)
        self.topK = topK
        self.train = train

        self.inputTransform = transforms.Compose([
            transforms.Resize(self.downsampleSize),
            transforms.Resize(self.imgSize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.trainTransform = transforms.Compose([
            transforms.RandomCrop(self.imgSize),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip()
        ])

        self.evalTransform = transforms.Compose([
            transforms.CenterCrop(self.imgSize),
        ])

        self.toTensor = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        if self.topK is None:
            return len(self.imgList)
        else:
            return min(self.topK, len(self.imgList))

    def __getitem__(self, idx):
        image_path = self.imgList[idx]
        img = Image.open(image_path).resize((256, 256))

        if len(img.getbands()) == 1:
            img = img.convert("RGB")

        if self.train:
            img = self.trainTransform(img)
        else:
            img = self.evalTransform(img)
        imgDownsample = self.inputTransform(img)
        img = self.toTensor(img)

        return imgDownsample, img


if __name__ == "__main__":
    usprData = USPRDataset("../data/Set14")
    usprLoader = DataLoader(usprData, batch_size=4, num_workers=4)

    for index, item in enumerate(usprData):
        print(item[1])
