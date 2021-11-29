# -*- coding: utf-8 -*-
"""
======================================
@Create Time : 2021/5/24 8:22 
@Author : 弓长广文武
======================================
"""
from torch import nn
from CSDNetV1_utils import MFRF_Dila, CCOFS, \
    MFRF_En, MFRF_En_Dila, MFRF_De, MainOut, MFRF_Dila_Submeter, CBR_Res_Down, CBR_Res_Up, CR_Down, CR_Mid, CR_Up, Out

'''
======================================
@File    :   CSDNetV1.py    
@Contact :   zhanggb1997@163.com
@Content :
======================================
'''
class CSDNet(nn.Module):
    def __init__(self, in_channel=3, classes_num=3):
        super(CSDNet, self).__init__()

        # self.en1 = CR_Down(in_channel, 32)
        # self.en2 = CR_Down(32, 64)
        # self.en3 = CR_Down(64, 128)

        self.en1 = MFRF_En(in_channel, 32)
        self.en2 = MFRF_En(32, 64)
        self.en3 = MFRF_En(64, 128)

        # self.en_dila = CR_Mid(128, 256)
        self.en_dila = MFRF_Dila(128, 256, [1, 2])
        # self.en_dila = MFRF_Dila(128, 256, [2, 4])
        # self.en_dila_submeter = MFRF_Dila_Submeter(128, 256, [2, 4, 2])

        self.de1 = MFRF_De(256, 128)
        self.de2 = MFRF_De(128, 64)
        self.de3 = MFRF_De(64, 32)

        # self.de1 = CR_Up(256, 128)
        # self.de2 = CR_Up(128, 64)
        # self.de3 = CR_Up(64, 32)

        self.ccofs1 = CCOFS(256, 2, classes_num, 16, 8, 4)
        self.ccofs2 = CCOFS(128, 2, classes_num, 8, 4, 2)
        self.ccofs3 = CCOFS(64, 2, classes_num, 4, 2, 1)

        self.mainout = MainOut(32, classes_num)
        # self.out = Out(32, classes_num)


    def forward(self, x):
        # encoder
        skip_en1, en1 = self.en1(x)
        skip_en2, en2 = self.en2(en1)
        skip_en3, en3 = self.en3(en2)

        # mid dilation
        dila = self.en_dila(en3)
        # dila = self.en_dila_submeter(en3)

        # decoder
        de1 = self.de1(dila, skip_en3)
        de2 = self.de2(de1, skip_en2)
        de3 = self.de3(de2, skip_en1)


        # mid confus out
        s7, o7 = self.ccofs1(dila)
        s8, o8 = self.ccofs2(de1)
        s9, o9 = self.ccofs3(de2)

        # out
        o10 = self.mainout(s7, s8, s9, de3)
        # o10 = self.out(de3)

        return (o7, o8, o9, o10)
        # return o10




