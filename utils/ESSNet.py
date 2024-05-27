import torch
import torch.nn as nn
import torch.nn.functional as F
from data_raw import get_patch_LoG


# 输入 N C H W   输出 N C H W
class ESSNet(nn.Module):
    def __init__(self, pca_component=50, num_classes=13, patch_size=15, att_deep=3, inter_size=50):
        super(ESSNet, self).__init__()
        self.patch_size = patch_size
        self.layer2 = ESS_Att(inter_size, reduction_ratio=16, att_deep=att_deep)  # ResSPA(inter_size, inter_size)
        self.bn = nn.BatchNorm2d(inter_size)
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(32 * (inter_size - 6), 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear((patch_size * patch_size * 64), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.4)  # 初始0.4

    def forward(self, x):
        n, c, h, w = x.size()  # 100 50 15 15
        x = self.layer2(x)  # att

        x = x.unsqueeze(1)  # N 1 C P P   # 100 50 15 15
        out = self.conv3d_1(x)  # 100 8 46 15 15
        out = self.conv3d_2(out)  # 100 16 144 14 14
        out = self.conv3d_3(out)  # 100 32 44 15 15
        out = self.conv2d(out.reshape(out.shape[0], -1, self.patch_size, self.patch_size))  # 由 N 1 C P P -> N C P P  100 32 41 13 13
        out = out.reshape(out.shape[0], -1)  # N c*p*p
        out = F.relu(self.dropout(self.fc1(out)))
        out = F.relu(self.dropout(self.fc2(out)))
        out = self.fc3(out)  # N cls_num

        return out  # out batch_size, cls_num


class ESS_Att(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, att_deep=3):
        super(ESS_Att, self).__init__()
        self.edge_att = EdgeAttention()
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(in_channels),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.spectral_att = SpectralAttention(in_channels, reduction_ratio)
        self.att_deep = att_deep
        self.spatial_att_1 = SpatAttn(in_channels, reduction_ratio, 1)
        self.spatial_att_3 = SpatAttn(in_channels, reduction_ratio, 3)


        # self.bn = nn.Sequential(
        #     nn.BatchNorm2d(in_channels),
        #     # nn.GELU()
        #     nn.ReLU(inplace=True)
        # )
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        identity = x
        x = self.edge_att(x)
        x = self.spectral_att(x)
        identity_1 = x

        x = self.conv3d_3(x.unsqueeze(1))
        x = x.reshape(x.shape[0], -1, h, w)

        identity_2 = x
        x = self.spatial_att_1(x)
        x = F.relu(self.bn(x))
        x = self.spatial_att_3(x)
        x = F.relu(self.bn(x))
        x = F.relu(x + identity_2)

        return x


class EdgeAttention(nn.Module):
    def __init__(self, data_sign="PaviaU"):
        super(EdgeAttention, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.data_sign = data_sign
        # self.gamma = nn.Parameter(torch.ones(1) / 10)
        self.gamma = nn.Parameter(torch.ones(1, 50)/10)

    def forward(self, patch_ori):
        b, c, h, w = patch_ori.shape
        ln = nn.LayerNorm([h, w], elementwise_affine=False)
        patch_log = get_patch_LoG(self.data_sign, patch_ori)
        patch_log = ln(patch_log)
        patch_att = self.sigmoid(patch_log)
        patch_proj = patch_ori * patch_att
        # patch_proj = self.gamma * patch_proj
        patch_proj = self.gamma.unsqueeze(2).unsqueeze(3).expand_as(patch_ori) * patch_proj
        return patch_proj + patch_ori


class SpectralAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        max_out = max_out.view(max_out.size(0), -1)
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        out = avg_out + max_out
        out = self.sigmoid(out)
        out = out.unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * out + x


class SpatAttn(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, ratio=8, kernel_size=1):
        super(SpatAttn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        # out = 0.1 * out
        out = self.gamma * out
        # print("gama", self.gamma)
        return out
        # return self.bn(out)
