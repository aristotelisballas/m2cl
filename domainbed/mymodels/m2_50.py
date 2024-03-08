import torch
import torch.nn as nn
import torchvision
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
        img_sizes = {
            256: 56,
            512: 28,
            1024: 14,
            2048: 7
        }
        filt_sizes = {
            256: [3, 7, 14],
            512: [2, 4, 7],
            1024: [2, 3],
            2048: [1],
        }
        self.ext_block = ExtractionBlock(
            in_filters=planes * self.expansion,
            in_img_size=img_sizes[planes * self.expansion],
            p_comp=4,
            pool_sizes=filt_sizes[planes * self.expansion],
            p_drop=0.3
        )
        # print(f'width:{width}, planes:{planes}')

    def forward(self, x):
        # print(type(x))
        # print(x.shape)
        ext_outs = []
        # ext_convs = []
        if type(x) == tuple:
            ext_outs += x[1]
            # ext_convs += x[2]
            x = x[0]

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
        # ext_out, ext_out_conv = self.ext_block(out)
        ext_out = self.ext_block(out)
        ext_outs.append(ext_out)
        # ext_convs.append(ext_out_conv)
        # print(out.shape)
        return out, ext_outs #, ext_convs


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
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc_hc = nn.Linear(204800, num_classes)

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

        x, out1 = self.layer1(x)
        x, out2 = self.layer2(x)
        x, out3 = self.layer3(x)
        x, out4 = self.layer4(x)

        out = [out1, out2, out3, out4]
        out = [item for sublist in out for item in sublist]

        # Subset of outputs
        indices_out = list(range(0, len(out), 4))

        if (len(out)-1) not in indices_out:
            indices_out[-1] = (len(out)-1)

        out = [out[i] for i in indices_out]

        hc = torch.cat(out, 1)
        x = self.fc_hc(hc)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
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


def M250(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


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
        tuple(torch.Tensor, torch.Tensor)
            Shape `(batch_size, flat_dim)`.
        """
        x = self.comp(
            x
        )                       # (batch_size, compression_out_channels, img_size, img_size)

        # x_conv = x.clone()
        # x_conv = self.relu(x_conv)
        # x_conv = x_conv.flatten(-2, -1)

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
            Shape `(batch_size, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape (`batch_size, out_features)`.
        """
        x = self.fc1(x)
        # x_out = x.clone()
        # x_out = x_out.flatten(-2, -1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.flatten(-2, -1)

        return x


class HyperMLP(nn.Module):
    """Classic MLP for the projection of extracted images.
    Parameters
    ----------
    in_shape: int
            Shape of flattened extracted image.
    embed_dim: int
            Dimension of embedded image.
    """
    def __init__(self, in_shape, embed_dim):
        super().__init__()
        self.in_shape = in_shape
        self.embed_dim = embed_dim

        self.fc = nn.Linear(in_features=in_shape, out_features=embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        """Run Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, in_chans, img_size, img_size)`.

        Returns
        -------
        List of torch.Tensors
            Shape `(batch_size, embed_dim)`.
        """
        x = self.fc(x)
        x = self.act(x)
        # x = torch.sigmoid(x)
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
        # out_energy = []
        for pipe, mlp in zip(self.pipelines, self.mlps):
            x = pipe(x_init)
            x = mlp(x)
            out.append(x)
        out = torch.cat(out, 1)
        return out 

