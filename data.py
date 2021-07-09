# -*- coding: utf-8 -*-
"""
======================================
@Create Time : 2021/3/26 19:52 
@Author : 弓长广文武
======================================
"""
import os
import time

'''
======================================
@File    :   data.py    
@Contact :   zhanggb1997@163.com
@Content :
======================================
'''
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2

class DatasetLoad(Dataset):
    def __init__(self, data_path, image_file, label_file, image_mode, label_mode, classes_num=3):
        self.data_path = data_path
        # 获取所有图像
        self.img_path = os.path.join(self.data_path, image_file)
        self.img_list = glob(os.path.join(self.img_path, '*.tif*'))
        self.lab_path = os.path.join(self.data_path, label_file)
        self.lab_list = glob(os.path.join(self.lab_path, '*.tif*'))

        assert len(self.lab_list) == len(self.img_list), 'label与image数量不一致'

        self.image_mode = image_mode
        self.label_mode = label_mode
        self.classes_num = classes_num

    def augment(self, image, label, flipCode): # cv2总进行数据增强 1水平翻转 0垂直翻转 -1水平垂直翻转
        img_flip= cv2.flip(image, flipCode)
        lab_flip= cv2.flip(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.img_list)

    def __getitem__(self, index):
        image_path = self.img_list[index]
        label_path = self.lab_list[index]
        # 读取影像
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), self.image_mode)
        lab = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), self.label_mode)
        # 处理图像
        assert lab.shape[:2] == img.shape[:2], 'label与image图像大小不一致！'
        # 图像数值压缩
        img = img / 255.
        # 判断多分类还是二分类
        if self.classes_num < 3:
            lab = lab / 255.
            new_lab = np.zeros(lab.shape)
            # 标签数据转换为二值图像
            new_lab[lab > 0.5] = 1
            # new_lab[lab <= 0.5] = 0
            # new_lab = torch.from_numpy(new_lab).long()
        else:
            # 不用做one_hot
            # new_lab = np.zeros((self.classes_num, lab.shape[0], lab.shape[1]))
            # for i in range(self.classes_num):
            #     new_lab[i, lab == i] = 1
            # new_lab = torch.zeros(self.classes_num, lab.shape[0], lab.shape[1]).scatter_(2, lab, 1)
            # new_lab = torch.from_numpy(new_lab).long()
            new_lab = lab

        img = np.transpose(img, (2, 0, 1))
        return img, new_lab

class PredLoad(Dataset):
    def __init__(self, data_path, image_file, classess_num=3):
        self.data_path = data_path
        # 获取所有图像
        self.img_path = os.path.join(self.data_path, image_file)
        self.img_list = glob(os.path.join(self.img_path, '*.*'))
        self.classes_num = classess_num

    def __len__(self):
        # 返回训练集大小
        return len(self.img_list)

    def __getitem__(self, index):
        image_path = self.img_list[index]
        image_name = os.path.split(image_path)[1]
        # 读取影像
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 处理图像
        # 图像数值压缩
        img = img / 255.
        img = np.transpose(img, (2, 0, 1))
        return img, image_name

class PredOut(object):
    def __init__(self, pred_map, pred_save_path, pred_save_name, color_dict=None, classes_num=3, flag=''):
        self.pred_map = pred_map
        self.pred_save_path = pred_save_path
        self.pred_save_name = pred_save_name
        self.color_dict = color_dict
        self.classes_num = classes_num
        self.flag = flag

    def predprocess(self):
        for i, pred in enumerate(self.pred_map):
            pred_res = np.zeros(pred.shape, dtype=np.uint8)

            # 判断是多分类还是二分类
            if self.classes_num < 3:
                # 标签数据转换为二值图像
                pred_res[pred > 0.5] = 255
                pred_res[pred <= 0.5] = 0

            else:
                for row in range(pred_res.shape[1]):
                    for col in range(pred_res.shape[2]):
                        index_of_class = np.argmax(pred[:, row, col])
                        pred_res[:, row, col] = self.color_dict[index_of_class]
                # pred_res = torch.argmax(pred, 1)

            name = self.pred_save_name[i]
            self.predsave(self.pred_save_path, name, pred_res)

    def predsave(self, path, name, result):
        now_time = time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='年', m='月', d='日', h='时', M='分', s='秒')
        name = os.path.splitext(name)[0] + self.flag + now_time + os.path.splitext(name)[1]
        if not self.classes_num < 3:
            pred_res = np.transpose(result, (1, 2, 0))
        else:
            pred_res = result[0, :, :]
        cv2.imencode('.tif', pred_res)[1].tofile(os.path.join(path, name))





if __name__ == "__main__":
    isbi_dataset = DatasetLoad(r"E:\a学生文件\张广斌\data\my_data\CSD_S5\512\last_5000\no_en\train", 'image', 'label', 1, 0)
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)








#
# class BasicDataset(Dataset):
#     def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
#         self.imgs_dir = imgs_dir
#         self.masks_dir = masks_dir
#         self.scale = scale
#         self.mask_suffix = mask_suffix
#         assert 0 < scale <= 1, 'Scale must be between 0 and 1'
#
#         self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
#                     if not file.startswith('.')]
#         logging.info(f'Creating dataset with {len(self.ids)} examples')
#
#     def __len__(self):
#         return len(self.ids)
#
#     @classmethod
#     def preprocess(cls, pil_img, scale):
#         w, h = pil_img.size
#         newW, newH = int(scale * w), int(scale * h)
#         assert newW > 0 and newH > 0, 'Scale is too small'
#         pil_img = pil_img.resize((newW, newH))
#
#         img_nd = np.array(pil_img)
#
#         if len(img_nd.shape) == 2:
#             img_nd = np.expand_dims(img_nd, axis=2)
#
#         # HWC to CHW
#         img_trans = img_nd.transpose((2, 0, 1))
#         if img_trans.max() > 1:
#             img_trans = img_trans / 255
#
#         return img_trans
#
#     def __getitem__(self, i):
#         idx = self.ids[i]
#         mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
#         img_file = glob(self.imgs_dir + idx + '.*')
#
#         assert len(mask_file) == 1, \
#             f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
#         assert len(img_file) == 1, \
#             f'Either no image or multiple images found for the ID {idx}: {img_file}'
#         mask = Image.open(mask_file[0])
#         img = Image.open(img_file[0])
#
#         assert img.size == mask.size, \
#             f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
#
#         img = self.preprocess(img, self.scale)
#         mask = self.preprocess(mask, self.scale)
#
#         return {
#             'image': torch.from_numpy(img).type(torch.FloatTensor),
#             'mask': torch.from_numpy(mask).type(torch.FloatTensor)
#         }
#
#
# class CarvanaDataset(BasicDataset):
#     def __init__(self, imgs_dir, masks_dir, scale=1):
#         super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')