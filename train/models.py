"""
Reference:
    https://github.com/sicara/easy-few-shot-learning
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""

# package
from typing import List, Optional, Type, Union

# torch
import torch
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1


# resnet (a little bit modified from typical resnet)
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        use_fc: bool = False,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
    ):
        super().__init__()

        self.inplanes = 64
        self.conv1 = (nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Only used when self.use_fc is True
        self.use_fc = use_fc
        self.fc = nn.Linear(self.inplanes, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)


    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) if self.use_fc else x

        return x

    def set_use_fc(self, use_fc: bool):
        
        # only use when we train under classical fsl
        self.use_fc = use_fc


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def set_resnet(backbone, num_classes, device):

    if backbone == "resnet18":
        return resnet18(num_classes=num_classes).to(device)
    elif backbone == "resnet34":
        return resnet34(num_classes=num_classes).to(device)
    elif backbone == "resnet50":
        return resnet50(num_classes=num_classes).to(device)
    else:
        raise ValueError("No related resnet in train.model")


# PrototypicalNetworks
class PrototypicalNetworks(nn.Module):
    
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()

        # backbone
        self.backbone = backbone

        # storing the processed support set
        self.prototypes = torch.tensor(())


    def compute_prototypes(self, support_imgs, support_labels):
        
        # get the support images features
        latent_support = self.backbone.forward(support_imgs)

        # Prototype i is the mean of all instances of features corresponding to labels == i
        self.prototypes = torch.cat(
            [latent_support[torch.nonzero(support_labels == label)].mean(0) for label in range(len(torch.unique(support_labels)))]
        )

    def forward(self, query_imgs):

        # get the query images features
        latent_query = self.backbone.forward(query_imgs)

        # compute the euclidean distance
        distance = torch.cdist(latent_query, self.prototypes)

        # the classification result must be inversely proportional to the distance
        return -distance