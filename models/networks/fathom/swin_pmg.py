import timm
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as func

from torch.optim import SGD
from libcore.libs.pl_model import FgICModel
from libcore.libs.registry import model_register
from libcore.libs.lr_schedulers import get_lr_scheduler

from models.utils import make_pair
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

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
            'swin_large_patch4_window7_224.ms_in22k',
            img_size=448, pretrained=True, features_only=True,
            out_dices=[1, 2, 3],
            pretrained_cfg_overlay=dict(file='/root/swin_large_patch4_window7_224_22k.pth'),
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
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
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


def gen_cost_matrix(file_path):
    cost_df = pd.read_csv(file_path, index_col=0)  # 替换为实际路径
    cost_matrix_np = cost_df.values.astype(float)  # 确保数值为浮点型
    cost_matrix = torch.from_numpy(cost_matrix_np).float()
    return cost_matrix


@model_register.register_module('swin_pmg')
class SwinPMGModel(FgICModel):
    def __init__(self, model_cfg, cls_num):
        super(SwinPMGModel, self).__init__(cls_num, model_cfg)
        self.automatic_optimization = True

        self.mixup_fn = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=0.1, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=cls_num
        )
        self.loss_fn = SoftTargetCrossEntropy()
        if cls_num == 79:
            self.register_buffer(
                'cost_matrix_device',  gen_cost_matrix("/root/FathomNet25/cost_metrix.csv"), persistent=True,
            )
        else:
            self.register_buffer(
                'cost_matrix_device',  torch.zeros((cls_num, cls_num)), persistent=True,
            )


    def build_network(self):
        net = SwinPMG(self.cls_num)
        return net, 5

    def backward_func(self, loss, optim):
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

    def training_step(self, batch_data, batch_idx):
        im, label           = batch_data
        im, label           = self.mixup_fn(im.clone(), label.clone())
        p1, p2, p3, pc      = self.network(im)

        loss_fn = self.loss_fn
        # loss_fn = func.cross_entropy

        pred = p1 + p2 + p3 + pc
        losses = [
            loss_fn(p1, label), loss_fn(p2, label), loss_fn(p3, label), loss_fn(pc, label),
        ]
        loss   = sum(losses)
        train_cost = self.cost_matrix_device[batch_data[1].long(), pred.argmax(dim=-1).reshape(-1).long()].mean()
        self.train_step_output = {'loss': loss.item(), 'losses': torch.stack(losses, dim=0), 'cost': train_cost}
        return loss

    def validation_step(self, batch_data, batch_idx):
        im, label           = batch_data
        p1, p2, p3, pc      = self.network(im)
        pred = p1 + p2 + p3 + pc

        loss_fn = func.cross_entropy
        losses = [
            loss_fn(p1, label), loss_fn(p2, label), loss_fn(p3, label), loss_fn(pc, label),
        ]
        loss   = sum(losses[1:])
        valid_cost = self.cost_matrix_device[batch_data[1].long(), pred.argmax(dim=-1).reshape(-1).long()].mean()
        return loss, torch.stack(losses, dim=0), [p1, p2, p3, pc, pred], valid_cost

    def configure_optimizers(self):
        optim_cfg           = self.model_cfg.OPTIM
        lr                  = make_pair(optim_cfg.LR)
        wd                  = make_pair(optim_cfg.WEIGHT_DECAY)
        mt                  = make_pair(optim_cfg.MOMENTUM)
        param1_to_update    = []
        param2_to_update    = []

        for name, param in self.network.named_parameters():
            param.requires_grad_(True)
            if not "backbone" in name:
                param1_to_update.append(param)
            else:
                param2_to_update.append(param)

        optimizer           = SGD(
            [
                {'params': param1_to_update, 'lr': lr[0], 'weight_decay': wd[0], 'momentum': mt[0]},
                {'params': param2_to_update, 'lr': lr[1], 'weight_decay': wd[1], 'momentum': mt[1]},
            ], lr=lr[1], weight_decay=wd[1], momentum=mt[1]
        )
        sched_cfg           = self.model_cfg.SCHED
        scheduler           = get_lr_scheduler(sched_cfg.NAME, optimizer, sched_cfg[sched_cfg.NAME], sched_cfg.WARMUP)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, "interval": "epoch", "frequency": 1,
            }
        }