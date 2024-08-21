import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Callable, Any, List
import math

__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int
    ) -> None:
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
            i: int,
            o: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class DownUpBone(nn.Module):
    def __init__(self, inplanes, out_channel):
        super(DownUpBone, self).__init__()
        # 使用平均池化，然后进行bn、relu、再进行一次,group conv 3 1操作
        # 使用平均池化功能类似于卷积功能，相当于模拟卷积操作。增大感受野
        out_planes = inplanes // 2

        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    nn.BatchNorm2d(inplanes),
                                    # nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, out_planes, kernel_size=(1, 1), bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    nn.BatchNorm2d(inplanes),
                                    # nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, out_planes, kernel_size=(1, 1), bias=False),
                                    )
        self.scale3 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=(3, 3), groups=inplanes, stride=(2, 2), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(inplanes),
            nn.Conv2d(inplanes, out_planes, kernel_size=(1, 1), bias=False),
        )
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, out_planes, kernel_size=(1, 1), bias=False),
        )
        self.process1 = nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), bias=False),
        )
        self.process2 = nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), bias=False),
        )
        self.process3 = nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), bias=False),
        )
        self.process4 = nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), bias=False),
        )
        self.compression = nn.Sequential(
            nn.BatchNorm2d(out_planes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes * 4, inplanes, kernel_size=(1, 1), bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, out_channel, kernel_size=(1, 1), bias=False),
        )

    def forward(self, x):
        x_list = []
        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x), scale_factor=2) + x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x), scale_factor=4) + x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x), scale_factor=2) + x_list[2])))
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class MaskBranch(nn.Module):
    def __init__(self,concat_ch=12,mask_classes=1):
        super(MaskBranch,self).__init__()
        #定义每个阶梯的特征融合节点,使用固定的通道数。定义顺序从网络底部到上面

        self.conv0=self.conv_dw(464+232,concat_ch)

        self.conv1=nn.Conv2d(116+concat_ch,concat_ch,kernel_size=(3,3),padding=(1,1))
        self.bn1=nn.BatchNorm2d(concat_ch)
        self.relu1=nn.ReLU(inplace=True)

        self.conv2=nn.Conv2d(24+concat_ch,concat_ch,kernel_size=(3,3),padding=(1,1))
        self.bn2=nn.BatchNorm2d(concat_ch)
        self.relu2=nn.ReLU(inplace=True)

        #网络的输入
        self.conv_d0=nn.Conv2d(concat_ch*2,mask_classes,kernel_size=(3,3),padding=(1,1))
        self.conv_d0_1=nn.Conv2d(mask_classes,mask_classes,kernel_size=(1,1))

        self.downup=DownUpBone(concat_ch,concat_ch)
        #辅助网络输出
        self.conv_d1=nn.Conv2d(concat_ch,mask_classes,kernel_size=(3,3),padding=(1,1))
        self.conv_d1_1=nn.Conv2d(mask_classes,mask_classes,kernel_size=(1,1))
        # 定义upsample
        self.upscore2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upscore4 = nn.Upsample(scale_factor=4, mode="bilinear")

    def conv_dw(self, inp, out_channel):
        return nn.Sequential(
            nn.Conv2d(inp, inp, groups=inp, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, out_channel, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def conv_131(self, inp):
        return nn.Sequential(
            nn.Conv2d(inp, inp // 2, kernel_size=(1, 1)),
            nn.Conv2d(inp // 2, inp // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(inp // 2),
            nn.Conv2d(inp // 2, inp, kernel_size=(1, 1)),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3, x4):
        """
        Args:
            x1: 网络底部的输出特征
            x2:
            x3:
            x4:

        Returns:

        """
        x_up=self.upscore2(x4)
        x=torch.cat([x_up,x3],dim=1)
        x=self.conv0(x)
        x_up=self.upscore2(x)
        x=torch.cat([x_up,x2],dim=1)
        x=self.relu1(self.bn1(self.conv1(x)))
        x_up=self.upscore4(x)
        x=torch.cat([x_up,x1],dim=1)
        x_output1=self.relu2(self.bn2(self.conv2(x))) #(112*112)
        x_up=self.upscore2(x_output1)
        #--output---
        x_downup=self.downup(x_up)
        x_up=torch.cat([x_downup,x_up],1)
        x=self.conv_d0(x_up) #224*224
        x=self.conv_d0_1(x)
        x_output1=self.upscore2(x_output1)
        x_d1=self.conv_d1(x_output1)
        x_d1=self.conv_d1_1(x_d1)
        return x,x_d1


class ShuffleNetV2(nn.Module):
    def __init__(
            self,
            stages_repeats: List[int],
            stages_out_channels: List[int],
            num_classes: int = 1000,
        	mask_classes: int = 1,
            inverted_residual: Callable[..., nn.Module] = InvertedResidual
    ) -> None:
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        # self.fc = nn.Linear(output_channels, num_classes)
        mask_classes = num_classes
        self.mask_branch=MaskBranch(mask_classes=mask_classes)

    def conv_group_bn(self, inp, kernel, stride, padding=1):
        return nn.Sequential(
            nn.Conv2d(inp, inp, kernel, stride, padding, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True))

    def _forward_impl(self, x: Tensor):
        # See note [TorchScript super()]  input #x (b,3,224,224)
        x1 = self.conv1(x)  # (b,24,112,112)
        x = self.maxpool(x1)  # (b,24,56,56)
        x2 = self.stage2(x)  # (b,116.28,28) x 1/8
        x3 = self.stage3(x2)  # (b,232,14,14) x 1/16
        x4 = self.stage4(x3)  # (b,464,7,7) x 1/ 32
        return x1, x2, x3, x4

    def forward(self, x: Tensor):
        x1, x2, x3, x4 = self._forward_impl(x)
        mask1, mask2 = self.mask_branch(x1, x2, x3, x4)
        return [mask2]

class MaskBranchPureMask(MaskBranch):

    def forward(self, x1, x2, x3, x4):
        """
        Args:
            x1: 网络底部的输出特征
            x2:
            x3:
            x4:

        Returns:

        """
        x_up = self.upscore2(x4)
        x = torch.cat([x_up, x3], dim=1)
        x = self.conv0(x)
        x_up = self.upscore2(x)
        x = torch.cat([x_up, x2], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x_up = self.upscore4(x)
        x = torch.cat([x_up, x1], dim=1)
        x_output1 = self.relu2(self.bn2(self.conv2(x)))  # (112*112)
        x_up = self.upscore2(x_output1)
        # --output---
        x_downup = self.downup(x_up)
        x_up = torch.cat([x_downup, x_up], 1)
        x = self.conv_d0(x_up)  # 224*224
        return F.sigmoid(x)


class ShuffleNetV2PureMask(ShuffleNetV2):
    def __init__(
            self,
            stages_repeats: List[int],
            stages_out_channels: List[int],
            num_classes: int = 1000,
            inverted_residual: Callable[..., nn.Module] = InvertedResidual
    ) -> None:
        super(ShuffleNetV2PureMask, self).__init__(stages_repeats,stages_out_channels,num_classes,inverted_residual)
        self.mask_branch = MaskBranchPureMask()

    def forward(self, x: Tensor):
        x1, x2, x3, x4 = self._forward_impl(x)
        mask = self.mask_branch(x1, x2, x3, x4)
        return mask


def _shufflenetv2(arch: str, pretrained: bool, progress: bool, *args: Any, **kwargs: Any) -> ShuffleNetV2:
    model = ShuffleNetV2(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            for key in ['fc.weight', 'fc.bias']:
                state_dict.pop(key)
            model.load_state_dict(state_dict, strict=False)

    return model


def _shufflenetv2_pure_mask(arch: str, pretrained: bool, progress: bool, *args: Any, **kwargs: Any) -> ShuffleNetV2:
    model = ShuffleNetV2PureMask(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            for key in ['fc.weight', 'fc.bias']:
                state_dict.pop(key)
            model.load_state_dict(state_dict, strict=False)

    return model


def shufflenet_v2_x0_5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress,
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(pretrained: bool = False, progress: bool = True, pure_network: bool = False,
                       **kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pure_network:
        return _shufflenetv2_pure_mask('shufflenetv2_x1.0', pretrained, progress,
                             [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)
    else:
        return _shufflenetv2("shufflenetv2_x1.0", pretrained, progress,
                                       [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.5', pretrained, progress,
                         [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained, progress,
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)


if __name__ == '__main__':
    inputs = torch.randn((2, 3, 150, 300))
    model = shufflenet_v2_x1_0()
    with torch.no_grad():
        output = model(inputs)
        print(output)
