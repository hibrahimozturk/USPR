import shutil

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
from PIL import Image
from skimage.measure import compare_ssim


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
    def __init__(self, expFolder, epochs=10):
        self.model = USPRNet()
        self.expFolder = expFolder
        os.makedirs(self.expFolder, exist_ok=True)
        self.writer = SummaryWriter(log_dir=expFolder)
        self.stepCounter = 0
        self.epochs = epochs

        self.trainDataset = USPRDataset(args.trainFolder)
        self.valDataset = USPRDataset(args.valFolder)

        self.trainLoader = DataLoader(self.trainDataset, batch_size=args.batchSize, num_workers=args.numWorkers)
        self.valLoader = DataLoader(self.valDataset, batch_size=args.batchSize, num_workers=args.numWorkers)

        self.mseLoss = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=args.schedulerStepSize,
                                                         gamma=args.schedulerGamma)

        self.bestPSNR = 0
        self.bestSSIM = 0

    def train(self):
        if torch.cuda.is_available():
            self.model.cuda()
        for epoch in range(self.epochs):
            for step, (imgDownsample, img) in enumerate(self.trainLoader):
                self.model.train()
                self.optim.zero_grad()
                if torch.cuda.is_available():
                    imgDownsample = imgDownsample.cuda()
                    img = img.cuda()
                output = self.model(imgDownsample)
                loss = self.mseLoss(output, img)
                loss.backward()

                self.optim.step()
                self.stepCounter += 1

                if self.stepCounter % args.logStep == 0:
                    print("Train [{}][{}/{}]\t"
                          "Step {}\t"
                          "Loss:{:.3f}".format(epoch, step, len(self.trainLoader), self.stepCounter, loss.data.item()))
            currentPSNR, currentSSIM = self.validation(epoch)
            self.saveCheckpoint(os.path.join(self.expFolder, "last.pth"))
            if currentPSNR > self.bestPSNR:
                self.bestPSNR = currentPSNR
                shutil.copy(os.path.join(self.expFolder, "last.pth"),
                            os.path.join(self.expFolder, "best.pth"))

    def validation(self, epoch):
        self.model.eval()
        PSNRscore = 0
        SSIMscore = 0
        with torch.no_grad():
            print("########## {} Epoch Validation Starts ##########".format(epoch))
            for step, (imgDownsample, img) in enumerate(self.valLoader):
                if torch.cuda.is_available():
                    imgDownsample = imgDownsample.cuda()
                    img = img.cuda()
                outputs = self.model(imgDownsample)
                loss = self.mseLoss(outputs, img)
                print("Val [{}]:[{}/{}] Loss: {}".format(epoch, step, len(self.valLoader), loss.item()))
                outputs = outputs.cpu().detach().numpy().transpose((0, 3, 2, 1))
                imgs = img.cpu().detach().numpy().transpose((0, 3, 2, 1))
                for output, img in zip(outputs, imgs):
                    PSNRscore += PSNR(output, img)
                    SSIMscore += compare_ssim(output, img, multichannel=True)
                PSNRscore /= len(self.valLoader)
                SSIMscore /= len(self.valLoader)
                self.writer.add_scalar("EvalScore/PSNR", PSNRscore, self.stepCounter)
                self.writer.add_scalar("EvalScore/SSIM", SSIMscore, self.stepCounter)
        print("########## {} Epoch Validation Scores: PSNR: {}, SSIM: {} ##########".format(epoch, PSNRscore, SSIMscore))

        return PSNRscore, SSIMscore

    def saveCheckpoint(self, path):
        torch.save({"model": self.model.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "stepCounter": self.stepCounter,
                    "bestPSNR": self.bestPSNR,
                    "bestSSIM": self.bestSSIM}, path)

    def loadCheckpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_stated_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.bestPSNR = checkpoint["bestPSNR"]
        self.bestSSIM = checkpoint["bestSSIM"]
        self.stepCounter = checkpoint["stepCounter"]


if __name__ == "__main__":
    task = USPR(args.expFolder, epochs=args.epochs)
    task.train()
