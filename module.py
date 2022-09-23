import torch
import torch.nn.functional as F
from torch import nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
import sys


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ClassSubnet(nn.Module):
    def __init__(self, num_classes, num_anchors,
                 pyramid_feature_size=256, prior_probability=0.01, classification_feature_size=256):
        super(ClassSubnet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=pyramid_feature_size, out_channels=classification_feature_size,
                               kernel_size=3, stride=1, bias=True, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=classification_feature_size, out_channels=classification_feature_size,
                               kernel_size=3, stride=1, bias=True, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=classification_feature_size, out_channels=classification_feature_size,
                               kernel_size=3, stride=1, bias=True, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=classification_feature_size, out_channels=classification_feature_size,
                               kernel_size=3, stride=1, bias=True, padding=1)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=classification_feature_size, out_channels=num_classes * num_anchors,
                               kernel_size=3, stride=1, bias=True, padding=1)
        self.act5 = nn.Sigmoid()
        torch.nn.init.normal_(self.conv1.weight, 0., 0.01)
        torch.nn.init.constant_(self.conv1.bias, 0.)
        torch.nn.init.normal_(self.conv2.weight, 0., 0.01)
        torch.nn.init.constant_(self.conv2.bias, 0.)
        torch.nn.init.normal_(self.conv3.weight, 0., 0.01)
        torch.nn.init.constant_(self.conv3.bias, 0.)
        torch.nn.init.normal_(self.conv4.weight, 0., 0.01)
        torch.nn.init.constant_(self.conv4.bias, 0.)
        torch.nn.init.constant_(self.conv5.weight, 0.)
        torch.nn.init.constant_(self.conv5.bias, - math.log((1 - prior_probability) / prior_probability))

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.conv5(out)
        out = self.act5(out)

        # out is B x C x W x H, with C = n_classes * n_anchors
        out = out.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)

        return out


class BoxSubnet(nn.Module):
    def __init__(self, num_anchors, pyramid_feature_size=256, regression_feature_size=256):
        super(BoxSubnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=pyramid_feature_size, out_channels=regression_feature_size,
                               kernel_size=3, stride=1, bias=True, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=regression_feature_size, out_channels=regression_feature_size,
                               kernel_size=3, stride=1, bias=True, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=regression_feature_size, out_channels=regression_feature_size,
                               kernel_size=3, stride=1, bias=True, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=regression_feature_size, out_channels=regression_feature_size,
                               kernel_size=3, stride=1, bias=True, padding=1)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=regression_feature_size, out_channels=4 * num_anchors,
                               kernel_size=3, stride=1, bias=True, padding=1)
        torch.nn.init.normal_(self.conv1.weight, 0., 0.01)
        torch.nn.init.constant_(self.conv1.bias, 0.)
        torch.nn.init.normal_(self.conv2.weight, 0., 0.01)
        torch.nn.init.constant_(self.conv2.bias, 0.)
        torch.nn.init.normal_(self.conv3.weight, 0., 0.01)
        torch.nn.init.constant_(self.conv3.bias, 0.)
        torch.nn.init.normal_(self.conv4.weight, 0., 0.01)
        torch.nn.init.constant_(self.conv4.bias, 0.)
        torch.nn.init.normal_(self.conv5.weight, 0., 0.01)
        torch.nn.init.constant_(self.conv5.bias, 0.)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.conv5(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)

        return out


class FeaturePyramidNet(nn.Module):
    def __init__(self, backbone_feature_size, pyramid_feature_size):
        super(FeaturePyramidNet, self).__init__()
        # backbone_feature_size: [c3, c4, c5]
        self.p5_conv1 = nn.Conv2d(in_channels=backbone_feature_size[2], out_channels=pyramid_feature_size,
                                  kernel_size=1, stride=1, bias=True, padding=0)
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_conv2 = nn.Conv2d(in_channels=pyramid_feature_size, out_channels=pyramid_feature_size,
                                  kernel_size=3, stride=1, bias=True, padding=1)

        self.p4_conv1 = nn.Conv2d(in_channels=backbone_feature_size[1], out_channels=pyramid_feature_size,
                                  kernel_size=1, stride=1, bias=True, padding=0)
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_conv2 = nn.Conv2d(in_channels=pyramid_feature_size, out_channels=pyramid_feature_size,
                                  kernel_size=3, stride=1, bias=True, padding=1)

        self.p3_conv1 = nn.Conv2d(in_channels=backbone_feature_size[0], out_channels=pyramid_feature_size,
                                  kernel_size=1, stride=1, bias=True, padding=0)
        self.p3_conv2 = nn.Conv2d(in_channels=pyramid_feature_size, out_channels=pyramid_feature_size,
                                  kernel_size=3, stride=1, bias=True, padding=1)

        self.p6_conv = nn.Conv2d(in_channels=backbone_feature_size[2], out_channels=pyramid_feature_size,
                                 kernel_size=3, stride=2, bias=True, padding=1)

        self.p7_relu = nn.ReLU()
        self.p7_conv = nn.Conv2d(in_channels=pyramid_feature_size, out_channels=pyramid_feature_size,
                                 kernel_size=3, stride=2, bias=True, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.p5_conv1(c5)
        p5_upsampled = self.p5_upsample(p5)
        p5 = self.p5_conv2(p5)

        p4 = self.p4_conv1(c4) + p5_upsampled
        p4_upsampled = self.p4_upsample(p4)
        p4 = self.p4_conv2(p4)

        p3 = self.p3_conv1(c3) + p4_upsampled
        p3 = self.p3_conv2(p3)

        p6 = self.p6_conv(c5)
        p7 = self.p7_conv(self.p7_relu(p6))

        return [p3, p4, p5, p6, p7]


# ResNet Backbone
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3, c4, c5


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


class RetinaNet(nn.Module):
    def __init__(self, backbone, pretrained, num_classes, num_anchors):
        super(RetinaNet, self).__init__()
        self.backbone = None
        if backbone == 'ResNet50':
            self.backbone = resnet50(pretrained)
        elif backbone == 'ResNet101':
            self.backbone = resnet101(pretrained)
        self.FPN = FeaturePyramidNet([512, 1024, 2048], 256)
        self.class_subnet = ClassSubnet(num_classes=num_classes, num_anchors=num_anchors, pyramid_feature_size=256,
                                        classification_feature_size=256)
        self.box_subnet = BoxSubnet(num_anchors=num_anchors, pyramid_feature_size=256, regression_feature_size=256)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.backbone.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        features = self.FPN(c3, c4, c5)
        loc_pred = []
        conf_pred = []
        for f in features:
            conf_pred.append(self.class_subnet(f))
            loc_pred.append(self.box_subnet(f))
        return torch.cat(loc_pred, 1), torch.cat(conf_pred, 1)
