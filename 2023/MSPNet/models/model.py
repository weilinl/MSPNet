import torch
import torch.nn as nn
import numpy as np

from SRM.srm_filter_kernel import all_normalized_hpf_list
from MPNCOV import MPNCOV

class TLU(nn.Module):
  def __init__(self, threshold):
    super(TLU, self).__init__()

    self.threshold = threshold

  def forward(self, input):
    output = torch.clamp(input, min=-self.threshold, max=self.threshold)

    return output


class HPF(nn.Module):
  def __init__(self):
    super(HPF, self).__init__()

    #Load 30 SRM Filters
    all_hpf_list_5x5 = []

    for hpf_item in all_normalized_hpf_list:
      if hpf_item.shape[0] == 3:
        hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

      all_hpf_list_5x5.append(hpf_item)

    hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)
    self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
    self.hpf.weight = hpf_weight

    #Truncation, threshold = 3
    self.tlu = TLU(3.0)

  def forward(self, input):

    out = self.hpf(input)
    out = self.tlu(out)

    return out


class CBR(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(CBR, self).__init__()

    self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
    self.bn = nn.BatchNorm2d(out_dim)
    self.act = nn.ReLU()

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.act(x)

    return x


class SFEBlock(nn.Module):
  def __init__(self, in_channels):
    super(SFEBlock, self).__init__()

    self.conv1 = CBR(in_channels, 2 * in_channels)
    self.conv1s = CBR(2 * in_channels, 2 * in_channels)
    self.conv2 = CBR(2 * in_channels, 2 * in_channels)
    self.conv2s = CBR(2 * in_channels, 2 * in_channels)
    self.conv3 = CBR(2 * in_channels,  4 * in_channels)
    self.conv3s = CBR(4 * in_channels, 4 * in_channels)
    self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)


  def forward(self, out):
    out = self.avg(out)
    out = self.conv1(out)
    out = self.conv1s(out)
    out = self.avg(out)
    out = self.conv2(out)
    out = self.conv2s(out)
    out = self.avg(out)
    out = self.conv3(out)
    out = self.conv3s(out)
    return out


class FC_BN_Relu(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(FC_BN_Relu, self).__init__()

    self.fc = nn.Linear(int(in_dim * (in_dim + 1) / 2), out_dim)
    self.bn = nn.BatchNorm1d(out_dim)
    self.act = nn.ReLU()

  def forward(self, x):
    x = self.fc(x)
    x = self.bn(x)
    x = self.act(x)

    return x


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()

    self.srm = HPF()

    self.conv_block_1 = CBR(30, 32)
    self.conv_block_2 = CBR(32, 32)
    self.conv_block_3 = CBR(32, 32)

    self.conv_block_4 = CBR(32, 32)
    self.scale_1 = SFEBlock(32)

    self.conv_block_5 = CBR(32, 32)
    self.scale_2 = SFEBlock(32)

    self.conv_block_6 = CBR(32, 32)
    self.scale_3 = SFEBlock(32)


    self.fc10 = FC_BN_Relu(128, 256)
    self.fc11 = nn.Linear(256, 2)

    self.fc20 = FC_BN_Relu(128, 256)
    self.fc21 = nn.Linear(256, 2)

    self.fc30 = FC_BN_Relu(128, 256)
    self.fc31 = nn.Linear(256, 2)


  def forward(self, input):
    x = input

    x0 = self.srm(x)
    x1 = self.conv_block_1(x0)
    x2 = self.conv_block_2(x1)
    x3 = self.conv_block_3(x2)

    x4 = self.conv_block_4(x3)
    scale_1 = self.scale_1(x4)

    x5 = x3 + x4
    x5 = self.conv_block_5(x5)
    scale_2 = self.scale_2(x5)

    x6 = x2 + x5
    x6 = self.conv_block_6(x6)
    scale_3 = self.scale_3(x6)

    s_1 = MPNCOV.CovpoolLayer(scale_1)
    s_1 = MPNCOV.SqrtmLayer(s_1, 5)
    s_1 = MPNCOV.TriuvecLayer(s_1)
    s_1 = s_1.view(s_1.size(0), -1)
    s_1_out = self.fc10(s_1)
    s_1 = self.fc21(s_1_out)

    s_2 = MPNCOV.CovpoolLayer(scale_2)
    s_2 = MPNCOV.SqrtmLayer(s_2, 5)
    s_2 = MPNCOV.TriuvecLayer(s_2)
    s_2 = s_2.view(s_2.size(0), -1)
    s_2_out = self.fc10(s_2)
    s_2 = self.fc21(s_2_out)

    s_3 = MPNCOV.CovpoolLayer(scale_3)
    s_3 = MPNCOV.SqrtmLayer(s_3, 5)
    s_3 = MPNCOV.TriuvecLayer(s_3)
    s_3 = s_3.view(s_3.size(0), -1)
    s_3_out = self.fc10(s_3)
    s_3 = self.fc21(s_3_out)

    return s_1, s_2, s_3, s_1_out, s_2_out, s_3_out
