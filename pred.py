# -*- coding: utf-8 -*-
"""
======================================
@Create Time : 2021/3/27 10:29 
@Author : 弓长广文武
======================================
"""
import os
import time
from glob import glob
from tqdm import tqdm
import torch
from torch.cuda import device
from torch.utils.data import DataLoader
import numpy as np
from pytorch_code.code_main.data import PredLoad, PredOut

'''
======================================
@File    :   pred.py    
@Contact :   zhanggb1997@163.com
@Content :
======================================
'''
SNOW = [255, 102, 36]
CLOUD = [66, 97, 255]
BACKGROUND = [214, 217, 212]

COLOR_DICT = np.array([BACKGROUND, CLOUD, SNOW])

def pred_model(data_path, save_path, batch_size, model=None, mod_wit_path=None, classes_num=3, flag='', is_multi_loss=False):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model and mod_wit_path:
        model.to(device)
        model.load_state_dict(torch.load(mod_wit_path, map_location=device))
    elif model:
        model.to(device)
    else:
        print('请传入模型')
        return
    data = PredLoad(data_path, 'image', classes_num)
    load = DataLoader(data, batch_size, shuffle=False)
    # 不训练
    # model.train(False)
    model.eval()
    # 放入数据
    with torch.no_grad():
        time_start = time.time()
        pred_result = []
        for image, img_name in tqdm(load, desc='Pred'):
            image = image.to(device=device, dtype=torch.float32)
            if is_multi_loss:
                pred = model(image)[-1]
            else:
                pred = model(image)
            pred_result.append(pred.detach().cpu().numpy())

        time_end = time.time()
        time_cost = time_end - time_start
        print(flag + 'Pred complete in {:d}m {:.4f}s'.format(int(time_cost // 60), time_cost % 60))

        for image, img_name in tqdm(load, desc='Pred'):
            image = image.to(device=device, dtype=torch.float32)
            if is_multi_loss:
                pred = model(image)[-1]
            else:
                pred = model(image)
            pred = pred.detach().cpu().numpy()
            pred_out = PredOut(pred, save_path, img_name, COLOR_DICT, classes_num=classes_num, flag=flag)
            pred_out.predprocess()
