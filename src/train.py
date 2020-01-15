import shutil

import torch
from torch.utils.data.dataloader import DataLoader
import os

from dataloader import USPRDataset
from torch.utils.tensorboard import SummaryWriter
from usprnet import USPRNet
from param import args
from skimage.measure import compare_ssim, compare_psnr
from PIL import Image
import adabound
import cv2


class USPR:
    def __init__(self, expFolder, epochs=10,
                 finetunePretrainedNet=False, lossFunction="mse",  checkpoint=None):
        self.model = USPRNet()
        self.expFolder = expFolder
        os.makedirs(self.expFolder, exist_ok=True)
        self.writer = SummaryWriter(log_dir=expFolder)
        self.stepCounter = 0
        self.epochs = epochs

        self.trainDataset = USPRDataset(args.trainFolder, train=True,
                                        downsampleRate=2, imgSizeFactor=8, imgSize=128)
        self.valDataset = USPRDataset(args.valFolder, topK=args.valTopK, imgSizeFactor=8, downsampleRate=2)
        self.TestDataset = USPRDataset(args.testFolder, imgSizeFactor=8, downsampleRate=2)

        self.trainLoader = DataLoader(self.trainDataset, batch_size=args.batchSize,
                                      num_workers=args.numWorkers, shuffle=True)
        self.valLoader = DataLoader(self.valDataset, batch_size=1, num_workers=args.numWorkers)
        self.testLoader = DataLoader(self.TestDataset, batch_size=1, num_workers=args.numWorkers)
        self.finetunePretrainedNet = finetunePretrainedNet
        # self.lossMultiplier = torch.autograd.Variable(torch.tensor(100., requires_grad=True))
        self.lossMultiplier = 1000.

        if torch.cuda.is_available():
            self.model.cuda()
            # self.lossMultiplier = self.lossMultiplier.cuda()

        if lossFunction == "mse":
            self.loss = torch.nn.MSELoss()
        elif lossFunction == "l1":
            self.loss = torch.nn.L1Loss()

        optimParams = list(self.model.superResolution.parameters())
        if self.finetunePretrainedNet:
            optimParams += list(self.model.pretrainedNet.parameters())
        self.optim = adabound.AdaBound(optimParams, lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=args.schedulerStepSize*len(self.trainLoader),
                                                         gamma=args.schedulerGamma)

        self.bestPSNR = 0
        self.bestSSIM = 0

        if checkpoint is not None:
            self.loadCheckpoint(checkpoint)

    def train(self):

        for epoch in range(self.epochs):
            for step, (imgDownsample, img) in enumerate(self.trainLoader):
                self.model.train()
                self.optim.zero_grad()
                if torch.cuda.is_available():
                    imgDownsample = imgDownsample.cuda()
                    img = img.cuda()
                output = self.model(imgDownsample)
                loss = self.loss(output, img)
                lossMultiplier = torch.autograd.Variable(torch.tensor(self.lossMultiplier, requires_grad=True))
                if torch.cuda.is_available():
                    lossMultiplier = lossMultiplier.cuda()
                loss = lossMultiplier * loss
                loss.backward()
                self.optim.step()
                self.scheduler.step()

                self.stepCounter += 1

                self.writer.add_scalar("Learning Rate", self.optim.param_groups[0]["lr"], self.stepCounter)
                self.writer.add_scalar("Loss/Train", loss.item(), self.stepCounter)
                if self.stepCounter % args.logStep == 0:
                    print("Train [{}][{}/{}]\t"
                          "Step {}\t"
                          "Loss:{:.3f}".format(epoch, step, len(self.trainLoader), self.stepCounter, loss.data.item()))

                if self.stepCounter % args.valStep == 0:
                    currentPSNR, currentSSIM = self.validation(epoch)
                    self.saveCheckpoint(os.path.join(self.expFolder, "last.pth"))
                    if currentPSNR > self.bestPSNR:
                        self.bestPSNR = currentPSNR
                        shutil.copy(os.path.join(self.expFolder, "last.pth"),
                                    os.path.join(self.expFolder, "best.pth"))

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
                loss = self.loss(outputs, img)
                if step % args.logStep == 0:
                    print("Val [{}]:[{}/{}] Loss: {}".format(epoch, step, len(self.valLoader), loss.item()))
                outputs = outputs.cpu().detach().numpy().transpose((0, 2, 3, 1))
                imgs = img.cpu().detach().numpy().transpose((0, 2, 3, 1))
                for output, img in zip(outputs, imgs):
                    imgYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
                    outputYCrCb = cv2.cvtColor(output, cv2.COLOR_BGR2YCR_CB)
                    PSNRscore += compare_psnr(imgYCrCb[:, :, 0], outputYCrCb[:, :, 0])
                    SSIMscore += compare_ssim(imgYCrCb[:, :, 0], outputYCrCb[:, :, 0])
            PSNRscore /= len(self.valLoader.dataset)
            SSIMscore /= len(self.valLoader.dataset)
            self.writer.add_scalar("EvalScore/PSNR", PSNRscore, self.stepCounter)
            self.writer.add_scalar("EvalScore/SSIM", SSIMscore, self.stepCounter)
            print("########## {} Epoch Validation Scores: PSNR: {}, SSIM: {} ##########".format(epoch, PSNRscore, SSIMscore))

        return PSNRscore, SSIMscore

    def test(self):
        PSNRscore = 0
        SSIMscore = 0
        imageCounter = 0
        if torch.cuda.is_available():
            self.model.cuda()

        print(self.model.superResolution.multiplier3)
        print(self.model.superResolution.multiplier5)

        for step, (imgDownsample, imgs) in enumerate(self.testLoader):
            if torch.cuda.is_available():
                imgDownsample = imgDownsample.cuda()
            output = self.model(imgDownsample)
            outputs = output.cpu().detach().numpy().transpose((0, 2, 3, 1))
            imgs = imgs.cpu().detach().numpy().transpose((0, 2, 3, 1))

            if not os.path.exists(os.path.join(self.expFolder, "outputs")):
                os.makedirs(os.path.join(self.expFolder, "outputs"))

            for img, output in zip(imgs, outputs):
                imgYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
                outputYCrCb = cv2.cvtColor(output, cv2.COLOR_BGR2YCR_CB)
                PSNRscore += compare_psnr(imgYCrCb[:, :, 0], outputYCrCb[:, :, 0])
                SSIMscore += compare_ssim(imgYCrCb[:, :, 0], outputYCrCb[:, :, 0])

                output *= 255
                outputImg = Image.fromarray(output.astype("uint8"))
                outputImg.save(os.path.join(self.expFolder, "outputs", "output_" + str(imageCounter)+".jpg"))

                img *= 255
                inputImg = Image.fromarray(img.astype("uint8"))
                inputImg.save(os.path.join(self.expFolder, "outputs", "input_" + str(imageCounter)+".jpg"))

                imageCounter += 1
        PSNRscore /= len(self.testLoader.dataset)
        SSIMscore /= len(self.testLoader.dataset)
        print("########## Test Scores: PSNR: {}, SSIM: {} ##########".format(PSNRscore, SSIMscore))

    def saveCheckpoint(self, path):
        torch.save({"model": self.model.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "stepCounter": self.stepCounter,
                    "bestPSNR": self.bestPSNR,
                    "bestSSIM": self.bestSSIM,
                    "finetunePretrainedNet": self.finetunePretrainedNet}, path)

    def loadCheckpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.bestPSNR = checkpoint["bestPSNR"]
        self.bestSSIM = checkpoint["bestSSIM"]
        self.stepCounter = checkpoint["stepCounter"]
        if not (self.finetunePretrainedNet ^ checkpoint["finetunePretrainedNet"]):
            self.optim.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])


if __name__ == "__main__":
    task = USPR(args.expFolder, finetunePretrainedNet=args.finetunePretrainedNet,
                epochs=args.epochs, checkpoint=args.checkpoint)
    if args.test:
        task.test()
    else:
        task.train()
