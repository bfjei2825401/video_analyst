import torch.nn.functional as F
import torch.nn as nn
import torch


class NonLocalBlock(nn.Module):
    def __init__(self, in_plane, out_plane, inner_plane):
        super(NonLocalBlock, self).__init__()
        self.in_plane = in_plane
        self.out_plane = out_plane
        self.inner_plane = inner_plane

        self.theta = nn.Conv2d(in_plane, inner_plane, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.phi = nn.Conv2d(in_plane, inner_plane, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.g = nn.Conv2d(in_plane, inner_plane, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        self.out = nn.Conv2d(inner_plane, out_plane, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn = nn.BatchNorm2d(out_plane)

    def forward(self, x):
        residual = x

        batch_size = x.shape[0]
        mp = self.maxpool(x)
        theta = self.theta(x)
        phi = self.phi(mp)
        g = self.g(mp)

        theta_shape = theta.shape
        theta = theta.view(batch_size, self.inner_plane, -1)
        phi = phi.view(batch_size, self.inner_plane, -1)
        g = g.view(batch_size, self.inner_plane, -1)

        theta_phi = torch.bmm(theta.transpose(1, 2), phi)  # (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784)
        theta_phi_sc = theta_phi * (self.inner_plane ** -.5)
        p = F.softmax(theta_phi_sc, dim=-1)

        t = torch.bmm(g, p.transpose(1, 2))
        t = t.view(theta_shape)

        out = self.out(t)
        out = self.bn(out)

        out = out + residual
        return out

    def zero_initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.zeros_(m.weight)

    def normal_initialize(self, std):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=std)  # conv_weight_std=0.01
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, std=std)  # conv_weight_std=0.01
