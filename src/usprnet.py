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
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(3)
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, features, inputImg):
        x = self.relu(self.deconv1(features["x7"]))
        x = self.bn1(x + features["x6"])
        x = self.relu(self.deconv2(x))
        x = self.bn2(x + features["x5"])
        x = self.relu(self.deconv3(x))
        x = self.bn3(x + features["x4"])
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.deconv5(x)
        x = self.bn6(x + inputImg)
        x = self.sigmoid(self.conv(x))
        return x


class Resnet50(torch.nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        features = list(models.resnet50().children())[:-2]
        self.features = nn.ModuleList(features)

    def forward(self, x):
        results = {}
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {4, 5, 6, 7}:
                results["x{}".format(ii)] = x
        return results


if __name__ == "__main__":
    uspr = USPRNet()
    x = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
    result = uspr(x)
    print("Finish")
