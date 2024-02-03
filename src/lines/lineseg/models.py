# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" This script holds the line segmentation models."""


import math

import torch
import torch.nn as nn
import torchvision

from .base_network import BaseNetwork


class ConvBlock(nn.Module):
    """U-Net constractive blocks"""

    def __init__(
        self, inchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=True
    ):  # default kenerl 3, stride 1, padding 1
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                inchannels,
                outchannels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outchannels),
            nn.Conv2d(
                outchannels,
                outchannels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outchannels),
        )

    def forward(self, x):
        return self.conv_block(x)


class UpBlock(nn.Module):
    """Up blocks in U-Net. Similar to the down blocks, but incorporates input from skip connections."""

    def __init__(
        self, inchannels, outchannels, kernel_size=2, stride=2
    ):  # default kernel 2, stride 2
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            inchannels, outchannels, kernel_size=kernel_size, stride=stride
        )
        self.conv = ConvBlock(inchannels, outchannels)

    def forward(self, x, skips):
        x = self.upconv(x)
        x = torch.cat([skips, x], 1)
        return self.conv(x)


class FCNUpBlock(nn.Module):
    """Up blocks in U-Net. Similar to the down blocks, but incorporates input from skip connections."""

    def __init__(self, inchannels, outchannels, kernel_size=2, stride=2):
        super().__init__()
        self.upconv = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = ConvBlock(inchannels, outchannels)

    def forward(self, x, skips):
        x = self.upconv(x)
        x = torch.cat([skips, x], 1)
        return self.conv(x)


class LineUnetModel(BaseNetwork):
    """
    Line segmentation model with patch-based classification (assymetric U-Net)
    Note that log2(seg_filter_size) should be less than the network depth (net_depth))
    """

    def __init__(
        self,
        in_channels=3,
        seg_filter_size=8,
        first_layer_filters=8,
        net_depth=4,
        num_classes=2,
        use_maxpool=True,
        stride=1,
    ):
        """Initialize the model

        Args:
            in_channels (int, optional): Input channels. Defaults to 3.
            seg_filter_size (int, optional): Filter size to downsample image output. Defaults to 8.
            first_layer_filters (int, optional): Number of output channels in first layer. Defaults to 8.
            net_depth (int, optional): Depth of the network. Defaults to 4.
            num_classes (int, optional): Number of output classes Defaults to 2.
            use_maxpool (bool, optional): Whether to apply maxpool or avgpool. Defaults to True.
            stride (int, optional): Stride to apply. Defaults to 1.
        """
        super().__init__()
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        if use_maxpool:
            self.pool = nn.MaxPool2d(2, 2)
        else:
            self.pool = nn.AvgPool2d(2, 2)
        self.in_channels = in_channels
        self.out_channels = first_layer_filters
        self.seg_filter_size = seg_filter_size
        self.net_depth = net_depth
        self.num_classes = num_classes
        self.stride = stride

        # down transformations
        for _ in range(self.net_depth):
            conv = ConvBlock(self.in_channels, self.out_channels, stride=self.stride)
            self.downblocks.append(conv)
            self.in_channels, self.out_channels = (
                self.out_channels,
                2 * self.out_channels,
            )

        # midpoint
        self.middle_conv = ConvBlock(self.in_channels, self.out_channels)

        # up transformations
        self.in_channels, self.out_channels = self.out_channels, int(
            self.out_channels / 2
        )

        upconv_depth = int(self.net_depth - math.log2(self.seg_filter_size))
        for _ in range(upconv_depth):
            upconv = UpBlock(self.in_channels, self.out_channels)
            self.upblocks.append(upconv)
            self.in_channels, self.out_channels = self.out_channels, int(
                self.out_channels / 2
            )

        self.seg_layer = nn.Conv2d(
            2 * self.out_channels, self.num_classes, kernel_size=1
        )

    def forward(self, x):
        decoder_outputs = []

        for op in self.downblocks:
            decoder_outputs.append(op(x))
            x = self.pool(decoder_outputs[-1])

        x = self.middle_conv(x)

        for op in self.upblocks:
            x = op(x, decoder_outputs.pop())
        return self.seg_layer(x)


class UnetModel(BaseNetwork):
    """U-Net model for line segmentation"""

    def __init__(
        self,
        in_channels=1,
        first_layer_filters=8,
        net_depth=4,
        num_classes=2,
        use_maxpool=True,
        stride=1,
    ):
        """Initialize the model

        Args:
            in_channels (int, optional): Input channels. Defaults to 3.
            seg_filter_size (int, optional): Filter size to downsample image output. Defaults to 8.
            first_layer_filters (int, optional): Number of output channels in first layer. Defaults to 8.
            net_depth (int, optional): Depth of the network. Defaults to 4.
            num_classes (int, optional): Number of output classes Defaults to 2.
            use_maxpool (bool, optional): Whether to apply maxpool or avgpool. Defaults to True.
            stride (int, optional): Stride to apply. Defaults to 1.
        """
        super().__init__()
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        if use_maxpool:
            self.pool = nn.MaxPool2d(2, 2)
        else:
            self.pool = nn.AvgPool2d(2, 2)
        self.in_channels = in_channels
        self.out_channels = first_layer_filters
        self.net_depth = net_depth
        self.num_classes = num_classes
        self.stride = stride

        # down transformations
        for _ in range(self.net_depth):
            conv = ConvBlock(self.in_channels, self.out_channels, stride=self.stride)
            self.downblocks.append(conv)
            self.in_channels, self.out_channels = (
                self.out_channels,
                2 * self.out_channels,
            )

        # midpoint
        self.middle_conv = ConvBlock(self.in_channels, self.out_channels)

        # up transformations
        self.in_channels, self.out_channels = self.out_channels, int(
            self.out_channels / 2
        )
        for _ in range(self.net_depth):
            upconv = UpBlock(self.in_channels, self.out_channels)
            self.upblocks.append(upconv)
            self.in_channels, self.out_channels = self.out_channels, int(
                self.out_channels / 2
            )

        self.seg_layer = nn.Conv2d(
            2 * self.out_channels, self.num_classes, kernel_size=1
        )

    def forward(self, x):
        decoder_outputs = []

        for op in self.downblocks:
            decoder_outputs.append(op(x))
            x = self.pool(decoder_outputs[-1])

        x = self.middle_conv(x)

        for op in self.upblocks:
            x = op(x, decoder_outputs.pop())
        return self.seg_layer(x)
