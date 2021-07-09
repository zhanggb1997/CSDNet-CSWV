# -*- coding: utf-8 -*-
"""
======================================
@Create Time : 2021/3/27 10:26 
@Author : 弓长广文武
======================================
"""
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from pytorch_code.code_main.data import DatasetLoad
from pytorch_code.code_net.UNet.zgb_UNet import UNet
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from colorama import Fore
'''
======================================
@File    :   trian.py    
@Contact :   zhanggb1997@163.com
@Content :
======================================
'''
def train_model(model, device, tra_path, val_path, epochs, batch_size, lr, val_percent=0.1, save_cp=True, img_scale=0.5,
              model_save_path='best_model.pth', log_save_path='/log', is_multi_loss=False, classes_num=3):
    # 数据的加载
    tra_data = DatasetLoad(tra_path, 'image', 'label', image_mode=1, label_mode=0, classes_num=classes_num)
    tra_load = DataLoader(tra_data, batch_size, shuffle=False, pin_memory=True, drop_last=False)
    val_data = DatasetLoad(val_path, 'image', 'label', image_mode=1, label_mode=0, classes_num=classes_num)
    val_load = DataLoader(val_data, batch_size, shuffle=False, pin_memory=True, drop_last=False)

    # 优化器定义
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    # 损失函数定义
    if classes_num > 2:
        criterion = nn.CrossEntropyLoss()  # 多分类
    else:
        criterion = torch.nn.BCEWithLogitsLoss()  # 二分类

    # 初始损失正无穷化
    loss_best = float('inf')
    tra_acc, tra_loss, val_acc, val_loss = [], [], [], []
    # 构建可视化损失和精确度图
    writer = SummaryWriter(log_save_path, comment='Acc_Loss_Show')

    # 清楚缓存
    # torch.cuda.empty_cache()

    # # 创建GradScaler对象
    scaler = GradScaler()

    # 开始训练
    start = time.time()
    for epoch in range(epochs):
        # model.train()
        print('\n' + '*' * 60)
        print('=================== Epoch {} / {}=================='.format(epoch+1, epochs))

        for phase in ['Train', 'Valid']:
            # 预先定义一个轮次中整体的损失、精度、步数
            run_loss = 0.0
            run_acc = 0.0
            step = 0

            # 判断模式
            if phase == 'Train':
                data_load = tra_load
                model.train(True)  # 调整为训练模式
                # 进度条
                tqdm_loader = tqdm(data_load)  # # 训练迭代器 创建显示进度条的对象
                tqdm_loader.bar_format = '{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET)  # 设置进度条样式属性
                tqdm_loader.unit = 'iterate'  # 每一次迭代为单位计算时间
                # 不断馈送数据
                for image, label in tqdm_loader:
                    step += 1  # 当前批次数
                    tqdm_loader.set_description('epoch:{}-trian:{}'.format(epoch + 1, step))  # 设置tqdm左边显示内容

                    # 优化器梯度清空
                    optimizer.zero_grad()  # 清除梯度值

                    # 数据拷贝到device中
                    image = image.to(device=device, dtype=torch.float32, non_blocking=True)
                    label = label.to(device=device, dtype=torch.float32, non_blocking=True)

                    # 启用AMP自动混合精度模式
                    with autocast():
                        pred = model(image)  # 预测
                        if not is_multi_loss:  # 不是多损失
                            loss = criterion(pred, label.long())  # 多分类
                            acc = acc_metric(pred, label.long())  # 多分类
                            # loss = criterion(np.squeeze(pred), label)  # 二分类
                            # acc = acc_metric(np.squeeze(pred), label)  # 二分类
                        else:  # 多损失情况
                            loss = multi_loss(pred, label.long(), epoch)  # 多分类
                            acc = multi_acc_metric(pred, label.long())  # 多分类
                            # loss = multi_loss(np.squeeze(pred), label, epoch)  # 二分类
                            # acc = multi_acc_metric(np.squeeze(pred), label)  # 二分类

                        loss = loss.requires_grad_()  # 使其能具有梯度属性


                    # 进度条右侧更新损失和精度
                    tqdm_loader.set_postfix(
                        loss_acc='----Loss:{:.5f}----Acc:{:.5f}'.format(loss.item(), acc.item()))

                    # 计算当前epoch中的整体acc\loss
                    run_acc += float(acc) * data_load.batch_size
                    run_loss += float(loss) * data_load.batch_size

                    # # # 更新参数梯度下降反向传播
                    # loss.backward()
                    # optimizer.step()
                    # # 使用amp后对损失放大进行梯度下降
                    scaler.scale(loss).backward()
                    # optimizer.step()
                    # 如果出现了inf或者NaN，scaler.step(optimizer)会忽略此次的权重更新
                    scaler.step(optimizer)
                    # 准备着，看是否要增大scaler  如果没有出现inf或者NaN，那么权重正常更新，
                    # 当连续多次没有出现inf或者NaN，则scaler.update()会将scaler的大小增加
                    scaler.update()
            else:
                data_load = val_load
                model.train(False)
                model.eval()
                # 进度条
                tqdm_loader = tqdm(data_load)  # # 验证迭代器  创建显示进度条的对象
                tqdm_loader.bar_format = '{l_bar}%s{bar}%s{r_bar}' % (Fore.GREEN, Fore.RESET)  # 设置进度条样式属性
                tqdm_loader.unit = 'iterate'  # 每一次迭代为单位计算时间

                # 不断馈送数据
                for image, label in tqdm_loader:
                    step += 1  # 当前批次数
                    tqdm_loader.set_description('epoch:{}-trian:{}'.format(epoch + 1, step))  # 设置tqdm左边显示内容

                    # 数据拷贝到device中
                    image = image.to(device=device, dtype=torch.float32)
                    label = label.to(device=device, dtype=torch.float32)

                    # # 启用AMP自动混合精度模式
                    with autocast():
                        with torch.no_grad():
                            pred = model(image)  # 预测
                            if is_multi_loss:  # 多输出
                                if classes_num > 2:
                                    loss = multi_loss_val(pred, label.long())  # 多分类
                                    acc = multi_acc_metric_val(pred, label.long())  # 多分类
                                else:
                                    loss = multi_loss_val(pred, label)  # 二分类
                                    acc = multi_acc_metric_val(pred, label)  # 二分类
                            else:
                                if classes_num > 2:
                                    loss = criterion(pred, label.long())  # 多分类
                                    acc = acc_metric(pred, label.long())  # 多分类
                                else:
                                    loss = criterion(np.squeeze(pred), label)  # 二分类
                                    acc = acc_metric(np.squeeze(pred), label)  # 二分类

                    # 进度条右侧更新损失和精度
                    tqdm_loader.set_postfix(
                        loss_acc='----Loss:{:.5f}----Acc:{:.5f}'.format(loss.item(), acc.item()))
                    # 计算当前epoch中的整体acc\loss
                    run_acc += float(acc) * data_load.batch_size
                    run_loss += float(loss) * data_load.batch_size


            epoch_loss = run_loss / len(data_load.dataset)
            epoch_acc = run_acc / len(data_load.dataset)

                # print('{} Loss: {:.5f} Acc: {:.5f}'.format(phase, epoch_loss, epoch_acc))
            # 建立损失精度图
            if phase == 'Train':
                tra_loss.append(epoch_loss), tra_acc.append(epoch_acc)
                writer.add_scalar('train_loss', epoch_loss, epoch + 1)
                writer.add_scalar('train_acc', epoch_acc, epoch + 1)
                # # 画出层中权值的分布情况
                # for name, param in model.named_parameters():
                #     writer.add_histogram(
                #         name, param.clone().data.numpy(), epoch_index)
            elif phase == 'Valid':
                val_loss.append(epoch_loss), val_acc.append(epoch_acc)
                writer.add_scalar('val_loss', epoch_loss, epoch + 1)
                writer.add_scalar('val_acc', epoch_acc, epoch + 1)
                # 监控验证集，保存val_loss最小的网络参数
                if epoch_loss < loss_best:
                    loss_last = loss_best
                    loss_best = epoch_loss

                    torch.save(model.state_dict(), model_save_path)
                    print('\nVal_loss improved from {:.5f} to {:.5f} and save to {}'.format(loss_last, loss_best, model_save_path))
                else:
                    print('\nVal_loss didn\'t improved from {:.5f}'.format(loss_best))

        if epoch == 0:
            time_all = 0
        time_epoch = time.time() - time_all - start
        print('Epoch:{}  train_loss:{:.5f}  train_acc:{:.5f}  val_loss:{:.5f}  val_acc:{:.5f}    Training and Validation 1 epoch in {:.0f}m {:.2f}s'
              .format(epoch+1, tra_loss[epoch], tra_acc[epoch], val_loss[epoch], val_acc[epoch], time_epoch // 60, time_epoch % 60))
        time_all += time_epoch
    time_elapsed = time.time() - start
    print('Training complete in {:d}m {:.2f}s'.format(int(time_elapsed // 60), time_elapsed % 60))
    writer.close()
    return tra_acc, tra_loss, val_acc, val_loss

def acc_metric(pred, label):
    return (pred.argmax(dim=1) == label.cuda()).float().mean()  # 多分类
    # return (label.cuda() - pred).float().mean()  # 二分类

def multi_acc_metric(pred, label):
    # acc5 = 0.05 * (pred[0].argmax(dim=1) == label.cuda()).float().mean()
    # acc1 = 0.1 * (pred[0].argmax(dim=1) == label.cuda()).float().mean()
    # acc2 = 0.15 * (pred[1].argmax(dim=1) == label.cuda()).float().mean()
    # acc3 = 0.25 * (pred[2].argmax(dim=1) == label.cuda()).float().mean()
    acc = (pred[-1].argmax(dim=1) == label.cuda()).float().mean()
    # return acc1 + acc2 + acc3 + acc
    return acc

def multi_acc_metric_val(pred, label):
    acc1 = (pred[-1].argmax(dim=1) == label.cuda()).float().mean()
    return acc1

def multi_loss_val(pred, label):
    criterion = nn.CrossEntropyLoss()  # 多分类
    loss1 = criterion(pred[-1], label)

    # criterion = torch.nn.BCEWithLogitsLoss() # 二分类
    # loss1 = criterion(np.squeeze(pred[-1]), label)
    return loss1

def multi_loss(pred, label, epoch):
    criterion = nn.CrossEntropyLoss()  # 多分类
    loss = criterion(pred[-1], label)
    loss0 = criterion(pred[0], label)
    loss1 = criterion(pred[1], label)
    loss2 = criterion(pred[2], label)

    # criterion = torch.nn.BCEWithLogitsLoss()  # 二分类
    # loss = criterion(np.squeeze(pred[-1]), label)
    # loss0 = criterion(np.squeeze(pred[0]), label)
    # loss1 = criterion(np.squeeze(pred[1]), label)
    # loss2 = criterion(np.squeeze(pred[2]), label)
    # if epoch < 49:
    # if epoch < 30:
    # if epoch < 75:
    #     loss0 = criterion(pred[0], label)
    #     loss1 = criterion(pred[1], label)
    #     loss2 = criterion(pred[2], label)
    #     # loss3 = criterion(pred[3], label)
    #     ratio = 1 / (2 * (1 + (int((epoch + 1) / 10))))
    #     ratio = 1 / (2 * (1 + (int((epoch + 1) / 15))))
    #     # ratio = 1 / (2 * (1 + (int((epoch + 1) / 6))))
    #     # ratio = 1 / (2 + (0.2 * epoch))
    #     # if epoch == 2:
    #     #     i = 1
    #     # print('loss0:{:.5f}   loss1:{:.5f}   loss2::{:.5f}   loss:{:.5f}   lossall:{:.5f}'.format(loss0, loss1, loss2, loss, loss0 * 0.2 + loss1 * 0.3 + loss2 * 0.5))
    #     return (loss0 * 0.2 + loss1 * 0.3 + loss2 * 0.5) * ratio + loss * (1 - ratio)
    #     # return (loss0 * 0.1 + loss1 * 0.2 + loss2 * 0.3 + loss3 * 0.4) * ratio + loss * (1 - ratio)


    # loss3 = 0.2 * criterion(pred[2], label)
    loss_all = loss0 * 0.2 + loss1 * 0.3 + loss2 * 0.5
    #
    if loss_all > 0.074:
        return loss_all * 0.5 + loss * 0.5
    else:
        return loss

            # + 0.25 * criterion(pred[2], label) + 0.15 * criterion(pred[1], label) + 0.1 * criterion(pred[0], label)
    # return loss1

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用  cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    model = UNet(3, 3)
    # 将网络拷贝到deivce中
    model.to(device=device)
    # 指定训练集地址，开始训练
    data_path = r'E:\a学生文件\张广斌\data\my_data\CSD_S5\512\last_5000\rota_en\train'
    train_model(model, device, data_path, 10, 1)