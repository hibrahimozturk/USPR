import os
import glob

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import torchvision.transforms as transforms


class USPRDataset(Dataset):
    def __init__(self, imgDir, train=False, downsampleRate=2, imgSizeFactor=8, imgSize=128, topK=None):

        self.imgDir = imgDir
        self.imgSize = imgSize
        self.imgList = glob.glob(os.path.join(self.imgDir, "*.jpg"))
        self.imgList += glob.glob(os.path.join(self.imgDir, "*.png"))
        self.downsampleRate = downsampleRate
        self.downsampleSize = int(self.imgSize/self.downsampleRate)
        self.topK = topK
        self.train = train
        self.imgSizeFactor = imgSizeFactor

        self.inputTransform = transforms.Compose([
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
        img = Image.open(image_path)

        w, h = img.size
        if w < self.imgSize or h < self.imgSize:
            print(image_path)
            os.remove(image_path)
        # w = w - w % self.imgSizeFactor
        # h = h - h % self.imgSizeFactor
        # if len(img.getbands()) == 1:
        #     img = img.convert("RGB")

        # if self.train:
        #     img = self.trainTransform(img)
        #     imgDown = img.resize((self.downsampleSize, self.downsampleSize))
        #     imgUp = imgDown.resize((self.imgSize, self.imgSize))
        # else:
        #     img = img.crop((0, 0, w, h))
        #     imgDown = img.resize((int(w/self.downsampleRate), int(h/self.downsampleRate)))
        #     imgUp = imgDown.resize((w, h))
        # imgInput = self.inputTransform(imgUp)
        # img = self.toTensor(img)

        return 0, 0


if __name__ == "__main__":
    usprData = USPRDataset("../data/coco/coco_train2014", train=True)
    usprLoader = DataLoader(usprData, batch_size=16, num_workers=4)

    for index, item in enumerate(usprLoader):
        x = 5
