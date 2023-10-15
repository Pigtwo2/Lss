import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SCSB(nn.Module):
    def __init__(self, in_channels):
        super(SCSB, self).__init__()

        self.conv_1xW = nn.Conv2d(in_channels, in_channels, (1, in_channels), bias=False)
        self.conv_Hx1 = nn.Conv2d(in_channels, in_channels, (in_channels, 1), bias=False)
        self.conv_1x1_gate = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        F_l = x
        E = self.conv_1xW(F_l)
        W = self.conv_Hx1(F_l)
        W = W.permute(0, 2, 1, 3)
        M = E + W
        G = self.conv_1x1_gate(M)
        G = self.sigmoid(G)
        F_y = G * F_l

        return F_y


class DPAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DPAM, self).__init__()
        self.conv_rate1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1)
        self.conv_rate3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=3, padding=3)
        self.conv_rate5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=5, padding=5)
        self.scsb = SCSB(in_channels)

    def forward(self, x):
        F_in = x

        F_l1 = self.conv_rate1(F_in)
        F_y1 = self.scsb(F_l1)
        F_z1 = F_l1 * F_y1

        F_l2 = self.conv_rate3(F_in)
        F_y2 = self.scsb(F_l2)
        F_z2 = F_l2 * F_y2

        F_l3 = self.conv_rate5(F_in)
        F_y3 = self.scsb(F_l3)
        F_z3 = F_l3 * F_y3

        F_z1_z2 = torch.sum([F_z1, F_z2], dim=1)
        F_z1_z3 = torch.sum([F_z1, F_z3], dim=1)
        F_z2_z3 = torch.sum([F_z2, F_z3], dim=1)

        F_out = torch.sum([F_z1_z2, F_z1_z3, F_z2_z3], dim=1)
        return F_out

class ShortRangeRefinement(nn.Module):
    def __init__(self, in_channels):
        super(ShortRangeRefinement, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)
        pool_out = torch.cat([max_out, avg_out], dim=1)
        conv_out = self.conv(pool_out)
        return self.sigmoid(conv_out)

class GlobalSemanticModule(nn.Module):
    def __init__(self, in_channels, kernel_sizes):
        super(GlobalSemanticModule, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, in_channels, k, padding=k//2) for k in kernel_sizes])
        self.sigmoid = nn.Sigmoid()

    def forward(self, Fd, Fu):
        Fd_avg = self.global_avg_pool(Fd)
        similarity = F.cosine_similarity(Fd_avg, Fu, dim=1, eps=1e-6)
        idx = torch.argmax(similarity)
        conv_out = self.convs[idx](Fd)
        return self.sigmoid(conv_out)

class EFM(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1, 3, 5]):
        super(EFM, self).__init__()

        self.short_range = ShortRangeRefinement(in_channels)
        self.global_semantic = GlobalSemanticModule(in_channels, kernel_sizes)

    def forward(self, x, Fu, Fd):
        sp = self.short_range(x)
        z = self.global_semantic(Fd, Fu)
        Fg = sp * z + x

        return Fg



class Network(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Network, self).__init__()
        resnet = models.resnet101(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.dpam = DPAM(512, 256)  # Sample channel sizes
        self.efm1 = EFM(256, 128)
        self.efm2 = EFM(128, 64)
        self.efm3 = EFM(64, 32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=3, dilation=3, padding=3)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        dpam_out = self.dpam(x4)
        fused_features=dpam_out + x3
        fused_features = self.relu(fused_features)
        efm1_out = self.efm1(fused_features)
        fused_features = efm1_out + x2
        fused_features = self.relu(fused_features)
        efm2_out = self.efm2(fused_features)
        fused_features = efm2_out + x1
        fused_features = self.relu(fused_features)
        efm3_out = self.efm3(fused_features)
        out = self.final_conv(efm3_out)
        out_upsampled = F.interpolate(out, size=4, mode='bilinear', align_corners=True)
        return out_upsampled