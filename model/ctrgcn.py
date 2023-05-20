import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MS_CAM(nn.Module):
    '''
    单特征进行通道注意力加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=1):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei



class AFF(nn.Module):
    '''
    Only one input branch
    '''

    def __init__(self, in_channels, r=1):
        super(AFF, self).__init__()
        inter_channels = in_channels // r
        channels = in_channels
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = x * wei + residual * (1 - wei)

        return xo

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=1):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # inter_channels = in_channels // r

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次局部注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


# 具体流程可以参考图1，通道注意力机制
class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #

        return x

# class NAMAtt(nn.Module):
#
#     def __init__(self, channels, shape, out_channels=None, no_spatial=True):
#         super(NAMAtt, self).__init__()
#         self.Channel_Att = Channel_Att(channels)
#
#     def forward(self, x):
#         x_out1 = self.Channel_Att(x)
#
#         return x_out1



# class MultiScale_TemporalConv(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=3,
#                  stride=1,
#                  dilations=[1,2,3,4],
#                  residual=True,
#                  residual_kernel_size=1):
#
#         super().__init__()
#         assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'
#
#         # Multiple branches of temporal convolution
#         self.num_branches = len(dilations) + 2
#         branch_channels = out_channels // self.num_branches
#         if type(kernel_size) == list:
#             assert len(kernel_size) == len(dilations)
#         else:
#             kernel_size = [kernel_size]*len(dilations)
#         # Temporal Convolution branches
#         self.branches = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(
#                     in_channels,
#                     branch_channels,
#                     kernel_size=1,
#                     padding=0),
#                 nn.BatchNorm2d(branch_channels),
#                 nn.ReLU(inplace=True),
#                 TemporalConv(
#                     branch_channels,
#                     branch_channels,
#                     kernel_size=ks,
#                     stride=stride,
#                     dilation=dilation),
#             )
#             for ks, dilation in zip(kernel_size, dilations)
#         ])
#
#         # Additional Max & 1x1 branch
#         self.branches.append(nn.Sequential(
#             nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
#             nn.BatchNorm2d(branch_channels),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
#             nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
#         ))
#
#         self.branches.append(nn.Sequential(
#             nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
#             nn.BatchNorm2d(branch_channels)
#         ))
#
#         # Residual connection
#         if not residual:
#             self.residual = lambda x: 0
#         elif (in_channels == out_channels) and (stride == 1):
#             self.residual = lambda x: x
#         else:
#             self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
#
#         # initialize
#         self.apply(weights_init)
#
#         self.aff = AFF(out_channels)
#
#     def forward(self, x):
#         # Input dim: (N,C,T,V)
#         res = self.residual(x)
#         aff = self.aff
#         branch_outs = []
#         for tempconv in self.branches:
#             out = tempconv(x)
#             # out = aff(out, 0)
#             branch_outs.append(out)
#         # aff(branch_outs[0], 0)
#         out = torch.cat(branch_outs, dim=1)
#
#         # out += res
#
#         out = aff(res, out)
#         # out = aff(out, res)
#         return out


# class EfficientChannelAttention(nn.Module):           # Efficient Channel Attention module
#     def __init__(self, c, b=1, gamma=2):
#         super(EfficientChannelAttention, self).__init__()
#         t = int(abs((math.log(c, 2) + b) / gamma))
#         k = t if t % 2 else t + 1
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.avg_pool(x)
#         x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         out = self.sigmoid(x)
#         return out


class MultiScale_TemporalConv1(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 # newds=[1],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'



        t = int(abs((math.log(out_channels, 2) + 1) / 2))
        newk = t if t % 2 else t + 1

        t2 = int(abs((math.log(out_channels, 2))))
        newds = [1]
        if t2 == 6:
            # if 2 in newds:
            #     newds.remove(2)
            newds.append(2)
            # d = 1
        if t2 == 7:
            # if 3 in newds:
            #     newds.remove(3)
            newds.append(2)
            # d = 2
        if t2 == 8:
            # if 3 in newds:
            #     newds.remove(3)
            newds.append(3)
            # d = 3

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    # kernel_size=ks,
                    kernel_size=newk,
                    stride=stride,
                    # dilation=dilation),
                    dilation=newd),
            )
            for ks, newd in zip(kernel_size, newds)
            # for ks, newd in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

        self.aff = AFF(out_channels)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        aff = self.aff
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            # out = aff(out, 0)
            branch_outs.append(out)
        # aff(branch_outs[0], 0)
        out = torch.cat(branch_outs, dim=1)

        out += res

        out = aff(res, out)
        # out = aff(out, res)
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.aff = AFF(out_channels)

        # self.namAtt = Channel_Att(out_channels)

        if in_channels == 3 or in_channels == 6:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)    # φ   3,8
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)    # ψ   3,8
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)    # 3,64
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)   # ξ   8,64
        self.tanh = nn.Tanh()  # M(·)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):  # x(256,3,64,25)   (N * M, C, T, V)
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)  # mean() 在哪一维上取平均值  x1(256,8,25) x2(256,8,25) x3(256,64,64,25)
        # aff = self.aff
        # x3 = aff(x3, 0)

        # namatt = self.namAtt
        # x3 = namatt(x3)

        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))  # unsqueeze()函数起升维的作用  -1:在第最后一维进行扩展  x1(256,8,25,25)  x2(256,8,25)

        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V   0:通道方向扩  x1(256,64,25,25)

        # namatt = self.namAtt
        # x1 = namatt(x1)

        # x1 = aff(x1, 0)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=8, rel_reduction=8, mid_reduction=1, adaptive=True, residual=True, attention=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        # self.conv_a = nn.ModuleList()
        # self.conv_b = nn.ModuleList()


        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))
            # self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            # self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        # self.beta = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)
        # self.namAtt = Channel_Att(out_channels)
        # if attention:
        #     inner_channel = out_channels // inter_channels
        #
        #     self.fcn = nn.Sequential(
        #         nn.Conv2d(out_channels, inner_channel, kernel_size=1, bias=bias),
        #         nn.BatchNorm2d(inner_channel),
        #         nn.Hardswish(),
        #     )
        #     self.conv_t = nn.Conv2d(inner_channel, out_channels, kernel_size=1)
        #     self.conv_v = nn.Conv2d(inner_channel, out_channels, kernel_size=1)
        # self.attention = attention

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        N, C, T, V = x.size()
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            # A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            # A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            # A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            # A = A1 * self.beta + A[i]

            # namatt = self.namAtt
            # x = namatt(x)

            z = self.convs[i](x, A[i], self.alpha)

            y = z + y if y is not None else z


        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        # self.tcn1 = MultiScale_TemporalConv1(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
        #                                     residual=False)
        self.tcn1 = MultiScale_TemporalConv1(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                             dilations=dilations,
                                             residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, center=21, stream="body"):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        if stream == "limb":
            A = self.graph.limb_A
        elif center == 21:
            A = self.graph.A  # 3,25,25


        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64

        layers = []

        l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        if stream == "limb":
            layers = [l1, l4, l5, l6, l7, l8, l9, l10]
        elif center == 21:
            layers = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10]
        elif center == 2 or center == 1:
            layers = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10]

        self.layers = nn.ModuleList(layers)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for m in self.layers:
            x = m(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)

