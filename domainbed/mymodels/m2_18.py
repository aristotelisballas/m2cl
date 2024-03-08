import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from collections import OrderedDict


__all__ = ['ResNet', 'resnet18']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, extraction=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        # Extraction

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


class ConcentrationPipeline(nn.Module):
    """Process intermediate conv outputs and compress them.
    Parameters
    ----------
    in_filters: int
            Size of the intermediate conv layer filters.
    in_img_size: int
            Size of the extracted input image.
    p_comp: int
            Compression parameter p.
    p_drop : float
            Feature map dropout probability between 0 and 1.
    pool_size: int
            Max pooling kernel size.

    Attributes
    ----------
    compression_out_channels : int
            Output channels of 1x1 convolution.
    comp : nn.Conv2d
            Convolutional layer that compresses the intermediate conv output.
    drop : nn.Dropout2d
            2D Dropout layer for the concentration pipeline.
    max : nn.MaxPool2d
            2D Max pooling layer.
    flat : nn.Flatten
            Flatten layer for the concatenation.
    """
    def __init__(self, in_filters, p_comp, pool_size, p_drop):
        super().__init__()
        self.in_filters = in_filters
        self.p_comp = p_comp
        self.pool_size = pool_size
        self.p_drop = p_drop

        self.compression_out_channels = in_filters // p_comp

        self.comp = nn.Conv2d(in_filters, self.compression_out_channels, kernel_size=(1, 1))

        self.drop = nn.Dropout2d(p_drop)
        self.max = nn.MaxPool2d(pool_size)
        self.relu = nn.ReLU(inplace=True)
        # self.flat = nn.Flatten()

    def forward(self, x):
        """Run Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, in_chans, img_size, img_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(batch_size, flat_dim)`.
        """
        x = self.comp(
            x
        )                       # (batch_size, compression_out_channels, img_size, img_size)
        x = self.drop(x)        # (batch_size, compression_out_channels, img_size, img_size)
        x = self.max(x)         # (batch_size, compression_out_channels, img_size//pool_size, img_size//pool_size)
        x = self.relu(x)
        # x = self.flat(x)        # (batch_size, compression_out_channels * (img_size//pool_size) ** 2 )
        x = x.flatten(-2, -1)

        return x


class MLP(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    in_features : int
        Number of input features.

    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
        Number of output features.

    p : float
        Dropout probability.

    Attirbutes
    ----------
    fc : nn.Linear
        The first linear layer.

    act : nn.GELU
        GELU activation function.

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer.
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, n_patches + 1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape (`batch_size, n_patches + 1, out_features)`.
        """
        x = self.fc1(x)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.flatten(-2, -1)

        return x


class ExtractionBlock(nn.Module):
    """Create an arbitrary number of extraction pipelines.
    Parameters
    ----------
    in_filters: int
            Size of the intermediate conv layer filters.
    in_img_size: int
            Size of the extracted input image.
    p_comp: int
            Compression parameter p.
    p_drop : float
            Feature map dropout probability between 0 and 1.
    pool_sizes: list
            Max pooling kernel size.

    Attributes
    ----------
    compression_out_channels : int
            Output channels of 1x1 convolution.
    pipelines : nn.Modulelist
            Multi-scale Concentration Pipelines for the extraction of intermediate conv layer outputs
    mlps : nn.ModuleList
            MLP implementation for the embedding of the extracted outputs
    """

    def __init__(self, in_filters, p_comp, in_img_size, pool_sizes, p_drop):
        super().__init__()
        pipelines = []
        mlps = []
        for i in range(len(pool_sizes)):
            pipeline = ConcentrationPipeline(
                in_filters,
                p_comp,
                pool_sizes[i],
                p_drop
            )
            # compression_out_channels = in_filters // p_comp
            # dim = compression_out_channels * (56//pool_sizes[i]) ** 2
            dim = (in_img_size//pool_sizes[i]) ** 2
            # mlp = HyperMLP(compression_out_channels * (in_img_size//pool_sizes[i]) ** 2, 64)
            mlp = MLP(in_features=dim,
                      hidden_features=1024,
                      out_features=128)

            pipelines.append(pipeline)
            mlps.append(mlp)

        self.pipelines = nn.ModuleList(pipelines)
        self.mlps = nn.ModuleList(mlps)

    def forward(self, x):
        """Run Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, in_chans, img_size, img_size)`.

        Returns
        -------
        Concatenated List of torch.Tensors
            Shape `(batch_size, 256*len(pool_sizes))`.
        """
        x_init = x.clone()
        out = []
        for pipe, mlp in zip(self.pipelines, self.mlps):
            x = pipe(x_init)
            x = mlp(x)
            out.append(x)
        out = torch.cat(out, 1)
        return out


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.extraction_block = ExtractionBlock(
            in_filters=64,
            p_comp=4,
            pool_sizes=[3, 7, 14],
            p_drop=0.3
        )

    def forward(self, x):
        x = self.conv(x)
        out = self.extraction_block(x)

        hc = torch.cat(out, 1)

        return hc


class M218(nn.Module):
    def __init__(self, classes, features=200, p=4, pretrained=True):
        super(M2, self).__init__()
        resnet = resnet18(pretrained=pretrained)
        self.classes = classes
        self.features = features

        layers = []
        children = list(resnet.children())
        seq = [4, 5, 6, 7]
        for i in range(len(children)):
            if i in seq:
                for j in range(2):
                    block_layers = list(list(children[i].children())[j].children())
                    for k in block_layers:
                        layers.append(k)
            else:
                layers.append(children[i])

        self.flat = nn.Flatten()

        self.ext_block1 = ExtractionBlock(
            in_filters=64,
            in_img_size=56,
            p_comp=p,
            pool_sizes=[3, 7, 14],
            p_drop=0.3
        )

        self.ext_block2 = ExtractionBlock(
            in_filters=64,
            in_img_size=56,
            p_comp=p,
            pool_sizes=[3, 7, 14],
            p_drop=0.3
        )

        self.ext_block3 = ExtractionBlock(
            in_filters=64,
            in_img_size=56,
            p_comp=p,
            pool_sizes=[3, 7, 14],
            p_drop=0.3
        )

        self.ext_block5 = ExtractionBlock(
            in_filters=128,
            in_img_size=28,
            p_comp=p,
            pool_sizes=[3],
            p_drop=0.3
        )

        self.ext_block8 = ExtractionBlock(
            in_filters=128,
            in_img_size=28,
            p_comp=p,
            pool_sizes=[3],
            p_drop=0.3
        )

        self.ext_block11 = ExtractionBlock(
            in_filters=256,
            in_img_size=14,
            p_comp=p,
            pool_sizes=[2],
            p_drop=0.3
        )

        self.ext_block12 = ExtractionBlock(
            in_filters=256,
            in_img_size=14,
            p_comp=p,
            pool_sizes=[2],
            p_drop=0.3
        )

        self.ext_block_14 = ExtractionBlock(
            in_filters=512,
            in_img_size=7,
            p_comp=p,
            pool_sizes=[1],
            p_drop=0.3
        )

        self.ext_block15 = ExtractionBlock(
            in_filters=512,
            in_img_size=7,
            p_comp=p,
            pool_sizes=[1],
            p_drop=0.3
        )

        self.ext_block16 = ExtractionBlock(
            in_filters=512,
            in_img_size=7,
            p_comp=p,
            pool_sizes=[1],
            p_drop=0.3
        )

        self.ext_add_block0 = ExtractionBlock(
            in_filters=128,
            in_img_size=28,
            p_comp=p,
            pool_sizes=[2, 3],
            p_drop=0.6
        )

        self.ext_add_block4 = ExtractionBlock(
            in_filters=256,
            in_img_size=14,
            p_comp=p,
            pool_sizes=[2],
            p_drop=0.6
        )

        self.ext_add_block5 = ExtractionBlock(
            in_filters=256,
            in_img_size=14,
            p_comp=p,
            pool_sizes=[2, 4, 7],
            p_drop=0.3
        )

        self.ext_add_block6 = ExtractionBlock(
            in_filters=512,
            in_img_size=7,
            p_comp=p,
            pool_sizes=[1],
            p_drop=0.3
        )

        self.ext_add_block7 = ExtractionBlock(
            in_filters=512,
            in_img_size=7,
            p_comp=p,
            pool_sizes=[1],
            p_drop=0.3
        )

        self.conv0 = layers[0]
        self.bn0 = layers[1]
        self.relu0 = layers[2]
        self.max0 = layers[3]

        # Layer - 1
        self.conv1_11 = layers[4]
        self.bn1_11 = layers[5]
        self.relu1_11 = layers[6]
        self.conv1_12 = layers[7]
        self.bn1_12 = layers[8]

        self.conv1_21 = layers[9]
        self.bn1_21 = layers[10]
        self.relu1_21 = layers[11]
        self.conv1_22 = layers[12]
        self.bn1_22 = layers[13]

        # Layer - 2
        self.conv2_11 = layers[14]
        self.bn2_11 = layers[15]
        self.relu2_11 = layers[16]
        self.conv2_12 = layers[17]
        self.bn2_12 = layers[18]
        self.down1 = layers[19]

        self.conv2_21 = layers[20]
        self.bn2_21 = layers[21]
        self.relu2_21 = layers[22]
        self.conv2_22 = layers[23]
        self.bn2_22 = layers[24]

        # Layer - 3
        self.conv3_11 = layers[25]
        self.bn3_11 = layers[26]
        self.relu3_11 = layers[27]
        self.conv3_12 = layers[28]
        self.bn3_12 = layers[29]
        self.down2 = layers[30]

        self.conv3_21 = layers[31]
        self.bn3_21 = layers[32]
        self.relu3_21 = layers[33]
        self.conv3_22 = layers[34]
        self.bn3_22 = layers[35]

        # Layer - 4
        self.conv4_11 = layers[36]
        self.bn4_11 = layers[37]
        self.relu4_11 = layers[38]
        self.conv4_12 = layers[39]
        self.bn4_12 = layers[40]
        self.down3 = layers[41]

        self.conv4_21 = layers[42]
        self.bn4_21 = layers[43]
        self.relu4_21 = layers[44]
        self.conv4_22 = layers[45]
        self.bn4_22 = layers[46]

        # Classic Top Layers
        self.avgpool = layers[47]
        self.fc = nn.Linear(512, classes)

        # Hypercolumn top layers
        # self.fc_feat = nn.Linear(276608//p, self.features)
        self.fc_hc = nn.Linear(149504, self.classes)

    def forward_hyper(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.max0(x)

        # Layer - 1
        identity = x
        x = self.conv1_11(x)

        out1 = self.ext_block1(x)

        x = self.bn1_11(x)
        x = self.relu1_11(x)

        x = self.conv1_12(x)
        out2 = self.ext_block2(x)

        x = self.bn1_12(x)

        add0 = x + identity
        x = nn.ReLU(inplace=True)(add0)

        identity = x
        x = self.conv1_21(x)

        out3 = self.ext_block3(x)

        x = self.bn1_21(x)
        x = self.relu1_21(x)
        x = self.conv1_22(x)
        x = self.bn1_22(x)

        add1 = x + identity
        x = nn.ReLU(inplace=True)(add1)

        # Layer - 2
        identity = x
        x = self.conv2_11(x)
        out5 = self.ext_block5(x)

        x = self.bn2_11(x)
        x = self.relu2_11(x)
        x = self.conv2_12(x)
        x = self.bn2_12(x)
        identity = self.down1(identity)

        add2 = identity + x
        add02_0 = self.ext_add_block0(x)

        x = nn.ReLU(inplace=True)(add2)

        identity = x
        x = self.conv2_21(x)
        x = self.bn2_21(x)
        x = self.relu2_21(x)
        x = self.conv2_22(x)

        out8 = self.ext_block8(x)
        x = self.bn2_22(x)

        add3 = identity + x
        x = nn.ReLU(inplace=True)(add3)

        # Layer - 3
        identity = x
        x = self.conv3_11(x)
        x = self.bn3_11(x)
        x = self.relu3_11(x)
        x = self.conv3_12(x)
        x = self.bn3_12(x)
        identity = self.down2(identity)

        add4 = identity + x
        x = nn.ReLU(inplace=True)(add4)
        add04_0 = self.ext_add_block4(x)

        identity = x
        x = self.conv3_21(x)
        out11 = self.ext_block11(x)

        x = self.bn3_21(x)
        x = self.relu3_21(x)
        x = self.conv3_22(x)
        out12 = self.ext_block12(x)

        x = self.bn3_22(x)

        add5 = identity + x
        x = nn.ReLU(inplace=True)(add5)
        add05_0 = self.ext_add_block5(x)

        # Layer - 4
        identity = x
        x = self.conv4_11(x)
        x = self.bn4_11(x)
        x = self.relu4_11(x)
        x = self.conv4_12(x)
        out14 = self.ext_block_14(x)

        x = self.bn4_12(x)
        identity = self.down3(identity)

        add6 = identity + x
        x = nn.ReLU(inplace=True)(add6)
        add06_0 = self.ext_add_block6(x)

        identity = x
        x = self.conv4_21(x)
        out15 = self.ext_block15(x)

        x = self.bn4_21(x)
        x = self.relu4_21(x)
        x = self.conv4_22(x)
        out16 = self.ext_block16(x)

        x = self.bn4_22(x)

        add7 = identity + x
        x = nn.ReLU(inplace=True)(add7)
        add07_0 = self.ext_add_block7(x)

        hc = torch.cat([out1, out2, out3, out5,
                        out8, out11, out12, out14, out15,
                        out16,
            # add02_0, add04_0,
            add05_0, add06_0, add07_0], 1)
        # hc = self.fc_feat(hc)
        # hc = nn.ReLU(inplace=True)(hc)
        hc = self.fc_hc(hc)

        return hc

    def forward(self, x):
        return self.forward_hyper(x)
