import numpy as np
import torch
from torch import nn


@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight)
        m.bias.fill_(0.0)


class ConvSingle(nn.Module):
    def __init__(
        self, inChannel, outChannel, groupChannel=32, kernel_size=3, padding=1
    ):
        super(ConvSingle, self).__init__()
        groups = min(outChannel, groupChannel)
        self.Conv = nn.Sequential(
            nn.Conv3d(
                inChannel,
                outChannel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=True,
            ),
            nn.BatchNorm3d(outChannel),
            nn.ReLU(inplace=True),
        )
        self.Conv.apply(init_weights)

    def forward(self, x):
        x = self.Conv(x)
        return x


class ConvDouble(nn.Module):
    def __init__(
        self, inChannel, outChannel, hasDropout=False, dropout_rate=0.2, groupChannel=32
    ):
        super(ConvDouble, self).__init__()
        groups = min(outChannel, groupChannel)
        self.hasDropout = hasDropout
        self.inplace = not self.hasDropout
        self.Conv1 = nn.Sequential(
            nn.Conv3d(
                inChannel, outChannel, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm3d(outChannel),
            nn.ReLU(inplace=self.inplace),
        )
        self.Dropout = nn.Dropout3d(p=dropout_rate, inplace=False)
        self.Conv2 = nn.Sequential(
            nn.Conv3d(
                outChannel, outChannel, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm3d(outChannel),
            nn.ReLU(inplace=True),
        )
        self.Conv1.apply(init_weights)
        self.Conv2.apply(init_weights)

    def forward(self, x):
        x = self.Conv1(x)
        if self.hasDropout:
            x = self.Dropout(x)
        x = self.Conv2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self, inChannel, outChannel, groupChannel=32, dropout=False, dropout_rate=0.2
    ):
        super(ConvBlock, self).__init__()
        self.Conv_1x1 = nn.Conv3d(
            inChannel, outChannel, kernel_size=1, stride=1, padding=0
        )
        self.Conv = ConvDouble(
            outChannel,
            outChannel,
            groupChannel=groupChannel,
            hasDropout=dropout,
            dropout_rate=dropout_rate,
        )
        init_weights(self.Conv_1x1)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x = self.Conv(x)
        return x


class UNet3D(nn.Module):
    def __init__(
        self,
        inchannels,
        outchannels=2,
        first_channels=32,
        image_size=(102, 127, 106),
        levels=5,
        dropout_enc=False,
        dropout_dec=False,
        dropout_rate=0.2,
        dropout_depth=2,
        concatenation=False,
    ):
        super(UNet3D, self).__init__()
        channels = first_channels
        self.levels = levels
        self.dropout_enc = dropout_enc
        self.dropout_dec = dropout_dec
        self.dropout_depth = dropout_depth
        self.kernel_size = 3
        self.stride = 2
        self.image_size = image_size
        self.concatenation = concatenation
        concat_factor = 1 if not self.concatenation else 2

        self.n_channels = [inchannels] + [
            channels * pow(2, i) for i in range(0, self.levels)
        ]
        print(self.n_channels)

        # Convolutional layer sizes
        def conv_out_size(in_size):
            return np.floor(
                (in_size + 2 * np.floor((self.kernel_size - 1) / 2) - self.kernel_size)
                / self.stride
                + 1
            ).astype(np.int64)

        self.layer_sizes = [list(self.image_size)]
        for idx in range(self.levels - 1):
            out_size = [conv_out_size(in_size) for in_size in self.layer_sizes[idx]]
            self.layer_sizes = self.layer_sizes + [out_size]
        print(self.layer_sizes)

        # Encoder
        encoders = [
            ConvBlock(
                self.n_channels[0],
                self.n_channels[1],
                groupChannel=first_channels,
                dropout=False,
                dropout_rate=dropout_rate,
            )
        ]
        for i in range(1, self.levels):
            block = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                ConvBlock(
                    self.n_channels[i],
                    self.n_channels[i + 1],
                    groupChannel=first_channels,
                    dropout=self.dropout_enc if i > self.dropout_depth else False,
                    dropout_rate=dropout_rate,
                ),
            )
            encoders.append(block)
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = [
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvSingle(
                    self.n_channels[self.levels],
                    self.n_channels[self.levels - 1],
                    groupChannel=first_channels,
                ),
            )
        ]
        for i in range(self.levels - 1, 1, -1):
            block = nn.Sequential(
                ConvBlock(
                    concat_factor * self.n_channels[i],
                    self.n_channels[i],
                    groupChannel=first_channels,
                    dropout=self.dropout_dec if i > self.dropout_depth else False,
                    dropout_rate=dropout_rate,
                ),
                nn.Upsample(scale_factor=2),
                ConvSingle(
                    self.n_channels[i],
                    self.n_channels[i - 1],
                    groupChannel=first_channels,
                ),
            )
            decoders.append(block)
        decoders.append(
            ConvBlock(
                concat_factor * self.n_channels[1],
                self.n_channels[1],
                groupChannel=first_channels,
            )
        )
        self.decoders = nn.ModuleList(decoders)

        self.lastConv = nn.Conv3d(
            self.n_channels[1], outchannels, kernel_size=1, stride=1, padding=0
        )
        init_weights(self.lastConv)

    def forward(self, x):
        inputStack = []
        l_i = x
        for i in range(self.levels):
            l_i = self.encoders[i](l_i)
            if i < self.levels - 1:
                inputStack.append(l_i)

        x = l_i
        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                if not self.concatenation:
                    x = x + inputStack.pop()
                else:
                    x = torch.cat([inputStack.pop(), x], 1)

        x = self.lastConv(x)
        x = torch.sigmoid(x)

        return x


if __name__ == "__main__":
    model = UNet3D(
        inchannels=1,
        outchannels=1,
        first_channels=16,
        image_size=tuple([256, 256, 128]),
        levels=4,
        dropout_enc=False,
        dropout_dec=False,
        dropout_rate=0.0,
        dropout_depth=2,
        concatenation=False,
    )
