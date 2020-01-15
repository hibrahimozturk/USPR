from torchvision import models
import torch.nn as nn
import torch


class USPRNet(nn.Module):
    def __init__(self):
        super(USPRNet, self).__init__()
        self.pretrainedNet = Resnet50()
        self.superResolution = SuperResolution()

    def forward(self, inputImg):
        features = self.pretrainedNet(inputImg)
        x = self.superResolution(features, inputImg)
        return x


class SuperResolution(torch.nn.Module):
    def __init__(self):
        super(SuperResolution, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        # self.multiplier1 = torch.autograd.Variable(torch.rand(1, requires_grad=True))
        # self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn1 = nn.BatchNorm2d(1024)
        # self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.multiplier2 = torch.autograd.Variable(torch.rand(1, requires_grad=True))
        # self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.multiplier3 = torch.autograd.Variable(torch.rand(1, requires_grad=True))
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.multiplier5 = torch.autograd.Variable(torch.rand(1, requires_grad=True))
        self.bn6 = nn.BatchNorm2d(3)
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

    def _apply(self, fn):
        super(SuperResolution, self)._apply(fn)
        # self.multiplier1 = fn(self.multiplier1)
        # self.multiplier2 = fn(self.multiplier2)
        self.multiplier3 = fn(self.multiplier3)
        self.multiplier5 = fn(self.multiplier5)
        return self

    def forward(self, features, inputImg):
        # x = self.lrelu(self.deconv1(features["x7"]))
        # x = self.bn1(x*self.multiplier1 + features["x6"])
        # x = self.lrelu(self.deconv2(x))
        # x = self.bn2(x*self.multiplier2 + features["x5"])
        x = self.lrelu(self.deconv3(features["x5"]))
        x = self.bn3(x*self.multiplier3 + features["x4"])
        x = self.bn4(self.lrelu(self.deconv4(x)))
        x = self.lrelu(self.deconv5(x))
        x = self.sigmoid(x*self.multiplier5 + inputImg)
        return x


class Resnet50(torch.nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        features = list(models.resnet50().children())[:-2]
        features = list(models.resnet50().children())[:-4]
        self.features = nn.ModuleList(features)

    def forward(self, x):
        results = {}
        for ii, model in enumerate(self.features):
            x = model(x)
            # if ii in {4, 5, 6, 7}:
            if ii in {4, 5}:
                results["x{}".format(ii)] = x
        return results


if __name__ == "__main__":
    uspr = USPRNet()
    uspr.cuda()
    x = torch.autograd.Variable(torch.randn(1, 3, 16, 16)).cuda()
    result = uspr(x)
    result = result.cpu().detach().numpy()
    print(result.shape)
