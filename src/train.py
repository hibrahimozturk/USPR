import torch
from torch.utils.data.dataloader import DataLoader
import os

from dataloader import USPRDataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np
from usprnet import USPRNet
from param import args


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


class USPR:
    def __init__(self, outputDir, epochs=10):
        self.model = USPRNet()
        self.output = outputDir
        os.makedirs(self.output, exist_ok=True)
        self.writer = SummaryWriter(log_dir=outputDir)
        self.stepCounter = 0
        self.epochs = epochs

        self.trainDataset = USPRDataset(args.train_img_dir)
        self.valDataset = USPRDataset(args.val_img_dir)
        self.trainLoader = DataLoader(self.trainDataset, batch_size=args.batch_size, num_workers=args.num_workers)
        self.valLoader = DataLoader(self.valDataset, batch_size=args.batch_size, num_workers=args.num_workers)

        self.mseLoss = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.schedulerStepSize,
                                                         gamma=args.schedulerGamma)

        self.bestScore = 0

    def train(self):
        if torch.cuda.is_available():
            self.model.cuda()
        for epoch in range(self.epochs):
            for step, item in enumerate(self.trainLoader):
                self.model.train()
                self.optim.zero_grad()
                if torch.cuda.is_available():
                    item[0] = item[0].cuda()
                    item[1] = item[1].cuda()
                output = self.model(item[0])
                loss = self.mseLoss(output, item[1])
                loss.backward()

                self.optim.step()
                self.stepCounter += 1

                if self.stepCounter % args.log_step == 0:
                    print("Train [{}][{}/{}]\t"
                          "Step {}\t"
                          "Loss:{:.3f}".format(epoch, step, len(self.trainLoader), self.stepCounter, loss.data.item()))

    def saveCheckpoint(self, path):
        torch.save({"model": self.model.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "stepCounter": self.stepCounter,
                    "bestScore": self.bestScore}, path)

    def loadCheckpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_stated_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.bestScore = checkpoint["bestScore"]
        self.stepCounter = checkpoint["stepCounter"]


if __name__ == "__main__":
    task = USPR(args.output, epochs=args.epochs)
    task.train()
