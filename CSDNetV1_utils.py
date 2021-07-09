# -*- coding: utf-8 -*-
"""
======================================
@Create Time : 2021/5/24 8:22 
@Author : 弓长广文武
======================================
"""
import torch
from torch import nn

'''
======================================
@File    :   CSDNetV1_utils.py    
@Contact :   zhanggb1997@163.com
@Content :
======================================
'''

class DCBR_BB(nn.Module):
    def __init__(self, in_channel, out_channel, ks=3, str=1, pad=1, dil=1):
        """
        DCBR_BB (DilationConvolution_BatchNormal_ReLUActivation Basic Block)
        :param in_channel: the number of input feature map channel
        :param out_channel: the number of output feature map channel
        :param ks: kernel size
        :param str: stride
        :param pad: padding num
        :param dil: dilation rate
        """
        super(DCBR_BB, self).__init__()
        self.DCBR = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(ks, ks), stride=(str, str), padding=(pad, pad),
                      dilation=(dil, dil)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
        )

    def forward(self, x):
        out = self.DCBR(x)
        return out

class DCR_BB(nn.Module):
    def __init__(self, in_channel, out_channel, ks=3, str=1, pad=1, dil=1):
        """
        DCBR_BB (DilationConvolution_ReLUActivation Basic Block)
        :param in_channel: the number of input feature map channel
        :param out_channel: the number of output feature map channel
        :param ks: kernel size
        :param str: stride
        :param pad: padding num
        :param dil: dilation rate
        """
        super(DCR_BB, self).__init__()
        self.DCR = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(ks, ks), stride=(str, str), padding=(pad, pad),
                      dilation=(dil, dil)),
            nn.ReLU(True),
        )

    def forward(self, x):
        out = self.DCR(x)
        return out

class CBR_BB(nn.Module):
    def __init__(self, in_channel, out_channel, ks, pad):
        """
        CBR_BB (Convolution_BatchNormal_ReLUActivation Basic Block)
        :param in_channel: the number of input feature map channel
        :param out_channel: the number of output feature map channel
        :param ks: kernel size
        :param pad: padding num
        """
        super(CBR_BB, self).__init__()
        self.CBR = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(ks, ks), padding=(pad, pad)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
        )
    def forward(self, x):
        out = self.CBR(x)
        return out

class CR_BB(nn.Module):
    def __init__(self, in_channel, out_channel, ks, pad):
        """
        CBR_BB (Convolution_ReLUActivation Basic Block)
        :param in_channel: the number of input feature map channel
        :param out_channel: the number of output feature map channel
        :param ks: kernel size
        :param pad: padding num
        """
        super(CR_BB, self).__init__()
        self.CR = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(ks, ks), padding=(pad, pad)),
            nn.ReLU(True),
        )
    def forward(self, x):
        out = self.CR(x)
        return out

class CBR_Res_Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CBR_Res_Down, self).__init__()
        self.En1 = CBR_BB(in_channel, out_channel, ks=3, pad=1)
        self.En2 = nn.Sequential(
            CBR_BB(out_channel, out_channel, ks=3, pad=1),
            CBR_BB(out_channel, out_channel, ks=3, pad=1))
        self.MaxPool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        en1 = self.En1(x)
        en2 = self.En2(en1)
        ski = en2.add(en1)
        out = self.MaxPool(ski)
        return ski, out


class CR_Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CR_Down, self).__init__()
        self.En = nn.Sequential(
            CR_BB(in_channel, out_channel, ks=3, pad=1),
            CR_BB(out_channel, out_channel, ks=3, pad=1))
        self.MaxPool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        en = self.En(x)
        out = self.MaxPool(en)
        return en, out


class CR_Mid(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CR_Mid, self).__init__()
        self.En = nn.Sequential(
            CR_BB(in_channel, out_channel, ks=3, pad=1),
            CR_BB(out_channel, out_channel, ks=3, pad=1))
        self.De = nn.Sequential(
            CR_BB(out_channel, out_channel, ks=3, pad=1),
            CR_BB(out_channel, out_channel, ks=3, pad=1))

    def forward(self, x):
        en = self.En(x)
        de = self.De(en)
        return de


class CR_Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CR_Up, self).__init__()
        self.Deco = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.De = nn.Sequential(
            CR_BB(in_channel, out_channel, ks=3, pad=1),
            CR_BB(out_channel, out_channel, ks=3, pad=1))

    def forward(self, x, skip):
        deco = self.Deco(x)
        cat = torch.cat((skip, deco), dim=1)
        out = self.De(cat)
        return out


class CBR_Res_Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CBR_Res_Up, self).__init__()
        self.Deco = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.De1 = CBR_BB(in_channel, out_channel, ks=3, pad=1)
        self.De2 = nn.Sequential(
            CBR_BB(out_channel, out_channel, ks=3, pad=1),
            CBR_BB(out_channel, out_channel, ks=3, pad=1))
        self.MaxPool = nn.MaxPool2d((2, 2))

    def forward(self, x, ski):
        deco = self.Deco(x)
        cat = torch.cat((deco, ski), dim=1)
        de1 = self.De1(cat)
        de2 = self.De2(de1)
        out = de2.add(de1)
        return out


class MFRF_En(nn.Module):

    def __init__(self, in_channel, out_channel):
        """
        MFRF_En (Multi-scale Feature Recept and Fusion module in encoder) + Maxpool
        :param in_channel: the number of input feature map channel
        :param out_channel: the number of output feature map channel
        """
        super(MFRF_En, self).__init__()
        self.MFRF_En_03 = CBR_BB(in_channel, out_channel, ks=3, pad=1)
        self.MFRF_En_11 = CR_BB(out_channel, out_channel, ks=1, pad=0)
        self.MFRF_En_13 = CR_BB(out_channel, out_channel, ks=3, pad=1)
        self.MFRF_En_23 = nn.Sequential(
            CR_BB(out_channel, out_channel, ks=3, pad=1),
            CR_BB(out_channel, out_channel, ks=3, pad=1),
        )
        self.MFRF_En_01 = CBR_BB(out_channel * 3, out_channel, ks=1, pad=0)
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        MFRF03 = self.MFRF_En_03(x)
        MFRF11 = self.MFRF_En_11(MFRF03)
        MFRF13 = self.MFRF_En_13(MFRF03)
        MFRF23 = self.MFRF_En_23(MFRF03)
        cat = torch.cat((MFRF11, MFRF13, MFRF23),  dim=1)
        MFRF01 = self.MFRF_En_01(cat)
        out = MFRF01.add(MFRF03)
        out = self.MaxPool(out)
        return MFRF01, out


class MFRF_En_Dila(nn.Module):
    def __init__(self, in_channel, out_channel, dila_rate=2):
        """
        MFRF_En_Dila (Multi-scale Feature Recept and Fusion module with dilate rate in encoder)
        :param dila_rate: dialte rate
        :param in_channel: the number of input feature map channel
        :param out_channel: the number of output feature map channel
        """
        super(MFRF_En_Dila, self).__init__()
        self.MFRF_En_03 = DCBR_BB(in_channel, out_channel, ks=3, pad=dila_rate, dil=dila_rate)
        self.MFRF_En_11 = CR_BB(out_channel, out_channel, ks=1, pad=0)
        self.MFRF_En_13 = DCR_BB(out_channel, out_channel, ks=3, pad=dila_rate, dil=dila_rate)
        self.MFRF_En_23 = nn.Sequential(
            DCR_BB(out_channel, out_channel, ks=3, pad=dila_rate, dil=dila_rate),
            DCR_BB(out_channel, out_channel, ks=3, pad=dila_rate, dil=dila_rate),
        )
        self.MFRF_En_01 = CBR_BB(out_channel * 3, out_channel, ks=1, pad=0)

    def forward(self, x):
        MFRF03 = self.MFRF_En_03(x)
        MFRF11 = self.MFRF_En_11(MFRF03)
        MFRF13 = self.MFRF_En_13(MFRF03)
        MFRF23 = self.MFRF_En_23(MFRF03)
        cat = torch.cat((MFRF11, MFRF13, MFRF23),  dim=1)
        MFRF01 = self.MFRF_En_01(cat)
        out = MFRF01.add(MFRF03)
        return MFRF01, out


class MFRF_De(nn.Module):

    def __init__(self, in_channel, out_channel):
        """
        MFRF_De (Multi-scale Feature Recept and Fusion module in decoder)
        :param in_channel: the number of input feature map channel
        :param out_channel: the number of output feature map channel
        """
        super(MFRF_De, self).__init__()
        self.Deco = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.MFRF_De_03 = CBR_BB(in_channel, out_channel, ks=3, pad=1)
        self.MFRF_De_11 = CR_BB(out_channel, out_channel, ks=1, pad=0)
        self.MFRF_De_13 = CR_BB(out_channel, out_channel, ks=3, pad=1)
        self.MFRF_De_23 = nn.Sequential(
            CR_BB(out_channel, out_channel, ks=3, pad=1),
            CR_BB(out_channel, out_channel, ks=3, pad=1),
        )
        self.MFRF_De_01 = CBR_BB(out_channel * 3, out_channel, ks=1, pad=0)

    def forward(self, x, skip):
        Deco = self.Deco(x)
        cat1 = torch.cat((skip, Deco), dim=1)
        MFRF03 = self.MFRF_De_03(cat1)
        MFRF11 = self.MFRF_De_11(MFRF03)
        MFRF13 = self.MFRF_De_13(MFRF03)
        MFRF23 = self.MFRF_De_23(MFRF03)
        cat2 = torch.cat((MFRF11, MFRF13, MFRF23), dim=1)
        MFRF01 = self.MFRF_De_01(cat2)
        out = MFRF01.add(MFRF03)
        return out

class MFRF_De_Dila(nn.Module):

    def __init__(self, in_channel, out_channel, dila_rate=2):
        """
        MFRF_De (Multi-scale Feature Recept and Fusion module in decoder)
        :param in_channel: the number of input feature map channel
        :param out_channel: the number of output feature map channel
        """
        super(MFRF_De_Dila, self).__init__()
        self.Deco = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.MFRF_De_03 = DCBR_BB(in_channel, out_channel, ks=3, pad=dila_rate, dil=dila_rate)
        self.MFRF_De_11 = CR_BB(out_channel, out_channel, ks=1, pad=0)
        self.MFRF_De_13 = DCR_BB(out_channel, out_channel, ks=3, pad=dila_rate, dil=dila_rate)
        self.MFRF_De_23 = nn.Sequential(
            DCR_BB(out_channel, out_channel, ks=3, pad=dila_rate, dil=dila_rate),
            DCR_BB(out_channel, out_channel, ks=3, pad=dila_rate, dil=dila_rate),
        )
        self.MFRF_De_01 = CBR_BB(out_channel * 3, out_channel, ks=1, pad=0)

    def forward(self, x):
        MFRF03 = self.MFRF_De_03(x)
        MFRF11 = self.MFRF_De_11(MFRF03)
        MFRF13 = self.MFRF_De_13(MFRF03)
        MFRF23 = self.MFRF_De_23(MFRF03)
        cat2 = torch.cat((MFRF11, MFRF13, MFRF23), dim=1)
        MFRF01 = self.MFRF_De_01(cat2)
        out = MFRF01.add(MFRF03)
        return out

class MFRF_De_Dila_Skip(nn.Module):

    def __init__(self, in_channel, out_channel, dila_rate=2):
        """
        MFRF_De (Multi-scale Feature Recept and Fusion module in decoder)
        :param in_channel: the number of input feature map channel
        :param out_channel: the number of output feature map channel
        """
        super(MFRF_De_Dila_Skip, self).__init__()
        self.Deco = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.MFRF_De_03 = DCBR_BB(in_channel * 2, out_channel, ks=3, pad=dila_rate, dil=dila_rate)
        self.MFRF_De_11 = CR_BB(out_channel, out_channel, ks=1, pad=0)
        self.MFRF_De_13 = DCR_BB(out_channel, out_channel, ks=3, pad=dila_rate, dil=dila_rate)
        self.MFRF_De_23 = nn.Sequential(
            DCR_BB(out_channel, out_channel, ks=3, pad=dila_rate, dil=dila_rate),
            DCR_BB(out_channel, out_channel, ks=3, pad=dila_rate, dil=dila_rate),
        )
        self.MFRF_De_01 = CBR_BB(out_channel * 3, out_channel, ks=1, pad=0)

    def forward(self, x, skip):
        cat1 = torch.cat((skip, x), dim=1)
        MFRF03 = self.MFRF_De_03(cat1)
        MFRF11 = self.MFRF_De_11(MFRF03)
        MFRF13 = self.MFRF_De_13(MFRF03)
        MFRF23 = self.MFRF_De_23(MFRF03)
        cat2 = torch.cat((MFRF11, MFRF13, MFRF23), dim=1)
        MFRF01 = self.MFRF_De_01(cat2)
        out = MFRF01.add(MFRF03)
        return out


class MFRF_Dila(nn.Module):
    def __init__(self, in_channel, out_channel, dila_list):
        super(MFRF_Dila, self).__init__()
        self.encoder = MFRF_En_Dila(in_channel, out_channel, dila_list[0])
        self.decoder = MFRF_De_Dila(out_channel, out_channel, dila_list[1])

    def forward(self, x):
        skip, en = self.encoder(x)
        de = self.decoder(en)
        return de


class MFRF_Dila_Submeter(nn.Module):
    def __init__(self, in_channel, out_channel, dila_list):
        super(MFRF_Dila_Submeter, self).__init__()
        self.encoder = MFRF_En_Dila(in_channel, out_channel, dila_list[0])
        self.mid = MFRF_En_Dila(out_channel, out_channel, dila_list[1])
        self.decoder = MFRF_De_Dila_Skip(out_channel, out_channel, dila_list[2])

    def forward(self, x):
        skip1, en = self.encoder(x)
        skip2, mid = self.mid(en)
        de = self.decoder(mid, skip1)
        return de

class CCOFS(nn.Module):
    def __init__(self, in_channel, mid_channel, class_num, ks_fac, stri, pad):
        super(CCOFS, self).__init__()
        self.class_num = class_num

        self.CCOFS_Out1 = nn.Sequential(
            nn.ConvTranspose2d(in_channel, mid_channel, kernel_size=(ks_fac, ks_fac), stride=(stri, stri),
                               padding=(pad, pad)),
        )
        self.CCOFS_Out2 = nn.Conv2d(in_channels=mid_channel, out_channels=class_num-1, kernel_size=(3, 3),
                                       padding=(1, 1))

    def forward(self, x):
        skip = self.CCOFS_Out1(x)
        out = self.CCOFS_Out2(skip)

        return skip, out

class MainOut(nn.Module):
    def __init__(self, in_channel, class_num):
        super(MainOut, self).__init__()
        self.class_num = class_num
        self.Out1 = CBR_BB(in_channel, 2, ks=3, pad=1)
        self.Out2 = nn.Conv2d(8, class_num-1, (1, 1))

    def forward(self, o6, o7, o8, x):
        out = self.Out1(x)
        cat = torch. cat((o6, o7, o8, out), dim=1)
        out = self.Out2(cat)
        return out

class Out(nn.Module):
    def __init__(self, in_channel, class_num):
        super(Out, self).__init__()
        self.class_num = class_num
        # self.Out1 = nn.Conv2d(in_channel, class_num, (3, 3), padding=(1, 1))
        self.Out1 = nn.Conv2d(in_channel, class_num-1, (1, 1))

    def forward(self, x):
        out = self.Out1(x)
        return out