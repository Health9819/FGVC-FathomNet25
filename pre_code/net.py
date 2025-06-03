import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as func


EPSILON = 1e-12


class UniAttPooling(nn.Module):
    def __init__(self, in_dim, map_num, softmax_scale):
        super(UniAttPooling, self).__init__()
        self.map_num = map_num
        self.softmax_scale = softmax_scale
        self.atte = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Conv1d(in_dim, map_num + 1, 1, 1, 0, bias=False),
        )
        self.post_bn = nn.Sequential(
            nn.BatchNorm1d(map_num + 1),
            nn.ReLU(),
        )

    def spatial(self, x):
        spatial_maps = self.atte(x)
        if self.softmax_scale > 0.0:
            spatial_maps = torch.softmax(spatial_maps * self.softmax_scale, dim=-1)
        spatial_maps = self.post_bn(spatial_maps)
        return spatial_maps

    def pooling(self, spatial_maps, feat):
        b, n, d = spatial_maps.shape
        ff = torch.einsum('b m n, b c n -> b m c', spatial_maps, feat)
        ff = torch.sign(ff) * torch.sqrt(torch.add(torch.abs(ff), 1e-6))
        ff = func.normalize(ff.reshape(b, -1), dim=-1, p=2).reshape(b, n, -1)
        return ff[:, :self.map_num, :]  # remove the background maps from the features.

    def forward(self, x, f):
        assert x.dim() == 3, (
                'The input dimension of uni attention module should be (B, C, N), '
                'but current shape is %s' % x.dim())
        spatial_maps = self.spatial(x)
        f = self.pooling(spatial_maps, f)
        return f


class MLPs(nn.Module):
    def __init__(self, in_dim, cls_num, r=2):
        super(MLPs, self).__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(in_dim), nn.Linear(in_dim, in_dim//r), nn.ELU(),
            nn.BatchNorm1d(in_dim//r), nn.Linear(in_dim//r, cls_num),
        )

    def forward(self, x):
        return self.fc(x)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SwinPMG(nn.Module):
    def __init__(self, num_classes, ):
        super(SwinPMG, self).__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            'swin_large_patch4_window7_224',
            img_size=448, pretrained=True, features_only=True,
            out_dices=[1, 2, 3],
        )

        self.num_features = [384, 768, 1536]

        nf, fs = [0, 384, 768, 1536], 1024
        self.conv_block1 = nn.Sequential(
            BasicConv(nf[1], nf[1], kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(nf[1], fs, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(nf[2], nf[2] // 2, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(nf[2] // 2, fs, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(nf[3], nf[3] // 4, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(nf[3] // 4, fs, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.fc1 = MLPs(fs, cls_num=num_classes)
        self.fc2 = MLPs(fs, cls_num=num_classes)
        self.fc3 = MLPs(fs, cls_num=num_classes)
        self.fc4 = MLPs(fs * 3, cls_num=num_classes, r=4)

        self.max = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(1, -1),
        )

    def forward(self, x):
        _, f1, f2, f3 = self.backbone(x)
        f1, f2, f3 = f1.permute(0, 3, 1, 2), f2.permute(0, 3, 1, 2), f3.permute(0, 3, 1, 2)

        xf1, xf2, xf3 = self.conv_block1(f1), self.conv_block2(f2), self.conv_block3(f3)
        pf1, pf2, pf3 = self.max(xf1), self.max(xf2), self.max(xf3)
        p1, p2, p3, pc = self.fc1(pf1), self.fc2(pf2), self.fc3(pf3), self.fc4(torch.cat([pf1, pf2, pf3], dim=1))
        return p1, p2, p3, pc

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}
        super().load_state_dict(pretrained_dict, strict)