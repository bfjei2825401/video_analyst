import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel),
                                nn.Sigmoid()
                                )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualAttentionLayer(nn.Module):
    def __init__(self, in_channel, down_k_size=[3, 2], down_stride=[2, 2], up_k_size=[2, 3], up_stride=[1, 1],
                 down_num=2, up_num=2):
        super(ResidualAttentionLayer, self).__init__()
        self.down_list = []
        self.up_list = []
        self.down_num = down_num
        self.up_num = up_num
        for i in range(0, down_num):
            down_conv = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=down_k_size[i], stride=down_stride[i]),
                                      nn.ReLU(inplace=True)
                                      )
            setattr(self, 'attention_down_conv_%d' % (i + 1), down_conv)
            self.down_list.append(down_conv)
        # last up need sigmoid layer
        for i in range(0, up_num-1):
            up_conv = nn.Sequential(nn.ConvTranspose2d(in_channel, in_channel, kernel_size=up_k_size[i], stride=up_stride[i]),
                                    nn.ReLU(inplace=True)
                                    )
            setattr(self, 'attention_up_conv_%d' % (i + 1), up_conv)
            self.up_list.append(up_conv)
        # last up-sampling layer
        last_up_conv = nn.Sequential(nn.ConvTranspose2d(in_channel, in_channel, kernel_size=up_k_size[-1], stride=up_stride[-1]),
                                     nn.Sigmoid()
                                     )
        setattr(self, 'attention_up_conv_%d' % up_num, last_up_conv)
        self.up_list.append(last_up_conv)

    def forward(self, feature):
        x = feature
        for i in range(0, self.down_num):
            x = getattr(self, 'attention_down_conv_%d' % (i + 1))(x)
        for i in range(0, self.up_num):
            x = getattr(self, 'attention_up_conv_%d' % (i + 1))(x)
        return (1.0 + x) * feature

    def initialize(self, std):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=std)  # conv_weight_std=0.01
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=std)  # conv_weight_std=0.01
