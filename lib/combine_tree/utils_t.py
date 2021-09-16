import nn as mynn

import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import time

from PIL import Image 
from collections import Counter

import sys
import json
import shutil
import os.path as osp
import os
"""

reference to https://github.com/yikang-li/FactorizableNet

"""



################### model modif ###################

def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr        
        
def np_to_tensor(x, is_cuda=True, dtype=torch.FloatTensor):
    v = torch.from_numpy(x).type(dtype)
    if is_cuda:
        v = v.cuda()
    return v
    
def set_trainable(model, requires_grad):
    set_trainable_param(model.parameters(), requires_grad)

def set_trainable_param(parameters, requires_grad):
    for param in parameters:
        param.requires_grad = requires_grad

def weight_init_mynn_conv_MSRAFill(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        mynn.init.MSRAFill(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)      
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
def weight_init_zeros(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
def weight_init_mynn_Xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        mynn.init.XavierFill(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
def weight_init_kaiming_uniform(m, fan_mode='fan_in', nonlinearity='relu'):    ### fan_mode='fan_in', nonlinearity='leaky_relu'
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, mode = fan_mode, nonlinearity = nonlinearity)
        if m.bias is not None:
            m.bias.data.fill_(0.)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.)

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

def weight_init_kaiming_norm(m, fan_mode='fan_in', nonlinearity='relu'):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        #nn.init.kaiming_normal_(m.weight.data, mode = fan_mode, nonlinearity = nonlinearity)
        if m.bias is not None:
            m.bias.data.fill_(0.)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.)

def weights_uniform_init(m, a=-1, b=1):    #math.sqrt(3)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight.data, a, b)
        if m.bias is not None:
            m.bias.data.fill_(0.)

def weights_orthogonal_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.)            

################### model save and load ###################      

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())
        #print '[Saved]: {}'.format(k)

def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        try:
            if k in h5f:
                param = torch.from_numpy(np.asarray(h5f[k]))
                v.copy_(param)
            else:
                print('[Not in pretrained]: {}'.format(k))
        except Exception as e:
            print('[Loaded net not complete] Parameter[{}] Size Mismatch...'.format(k))
            pdb.set_trace()            

def load_faster_rcnn(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        try:
            if k in h5f:
                param = torch.from_numpy(np.asarray(h5f[k]))
                v.copy_(param)
                print('[in faster_cnn]: {}'.format(k))
        except Exception as e:
            print('[Loaded net not complete] Parameter[{}] Size Mismatch...'.format(k))
            pdb.set_trace()   

def save_checkpoint(info, model, optim, dir_logs, scheduler, is_best=True):
    '''
    Example:

        save_checkpoint({
                    'epoch': epoch,
                    'best_ac': best_ac[0],
                },
                model.module, #model.module.state_dict(),
                optimizer.state_dict(),
                'output/FN_VRD_1_iters_SGD',
                scheduler.state_dict(),
                is_best)
    '''   
    os.system('mkdir -p ' + dir_logs)

    path_ckpt_info  = os.path.join(dir_logs, 'ckpt_info.pth.tar')
    path_ckpt_model = os.path.join(dir_logs, 'ckpt_model.h5')
    path_ckpt_optim = os.path.join(dir_logs, 'ckpt_optim.pth')
    
    path_ckpt_scheduler = os.path.join(dir_logs, 'ckpt_scheduler.pth')
    
    path_best_info  = os.path.join(dir_logs, 'best_info.pth.tar')
    path_best_model = os.path.join(dir_logs, 'best_model.h5')
    path_best_optim = os.path.join(dir_logs, 'best_optim.pth')
    
    path_best_scheduler = os.path.join(dir_logs, 'best_scheduler.pth')
    #print('Files created.')
    # save info
    torch.save(info, path_ckpt_info)
    #print('save path_ckpt_info.')
    if is_best:
        shutil.copyfile(path_ckpt_info, path_best_info)
        #print('save path_best_info.')
    # save model state & optim state
    save_net(path_ckpt_model, model)
    #print('save path_ckpt_model.')
    torch.save(optim, path_ckpt_optim)
    #print('save path_ckpt_optim.')
    torch.save(scheduler, path_ckpt_scheduler)
    #print('save path_ckpt_scheduler.')
    if is_best:
        shutil.copyfile(path_ckpt_model, path_best_model)
        #print('save path_best_model.')
        shutil.copyfile(path_ckpt_optim, path_best_optim)
        #print('save path_best_optim.')
        shutil.copyfile(path_ckpt_scheduler, path_best_scheduler)
        #print('save path_best_scheduler.')


def load_checkpoint(model, path_ckpt, optimizer=None, scheduler=None):
    '''
    Example:

        load_checkpoint(model, optimizer, scheduler, 'best')
    '''  
    path_ckpt_info  = path_ckpt + '_info.pth.tar'
    path_ckpt_model = path_ckpt + '_model.h5'
    path_ckpt_optim = path_ckpt + '_optim.pth'
    path_ckpt_scheduler = path_ckpt + '_scheduler.pth'
    start_epoch = 0
    best_ac = 0
    if os.path.isfile(path_ckpt_info):
        info = torch.load(path_ckpt_info)
        if 'epoch' in info:
            start_epoch = info['epoch'] + 1
        else:
            print('Warning train.py: no epoch to resume')
        if 'best_ac' in info:
            best_ac = info['best_ac']
        else:
            print('Warning train.py: no best_ac to resume')
    else:
        print("Warning train.py: no info checkpoint found at '{}'".format(path_ckpt_info))
    if os.path.isfile(path_ckpt_model):
        load_net(path_ckpt_model, model)
    else:
        print("Warning train.py: no model checkpoint found at '{}'".format(path_ckpt_model))
    
    # if os.path.isfile(path_ckpt_optim): 
    #     model.cuda()
    #     optim_state = torch.load(path_ckpt_optim)
    #     optimizer.load_state_dict(optim_state)
    # else:
    #     print("Warning train.py: no optim checkpoint found at '{}'".format(path_ckpt_optim))
    # if os.path.isfile(path_ckpt_scheduler):
    #     scheduler_state = torch.load(path_ckpt_scheduler)
    #     scheduler.load_state_dict(scheduler_state)
    # else:
    #     print("Warning train.py: no scheduler checkpoint found at '{}'".format(path_ckpt_scheduler))
    
    print("=> loaded checkpoint '{}' (epoch {}, best_ac {})"
              .format(path_ckpt, start_epoch, best_ac * 100))
    return start_epoch, best_ac



################### print ###################     
def print_log(file_path,*args):
    print(*args)
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args,file=f)

def show_config(cfg):
    print_log(cfg.log_path, '=====================Config=====================')
    for k,v in cfg.__dict__.items():
        print_log(cfg.log_path, k,': ',v)
    print_log(cfg.log_path, '======================End=======================')
    
def show_epoch_info(phase, log_path, info, lr=None):
    print_log(log_path, '')
    if phase=='Test':
        print_log(log_path, '====> %s at epoch #%d'%(phase, info['epoch']))
    else:
        if lr is None:
            print_log(log_path, '%s at epoch #%d'%(phase, info['epoch']))
        else:
            print_log(log_path, '%s at epoch #%d lr: %f'%(phase, info['epoch'], lr))
        
    print_log(log_path, 'Group Activity Accuracy: %.2f%%, Loss: %.5f, Action Loss: %.5f, Activity Loss: %.5f, Using %.1f seconds'%(
                info['activities_acc'], info['loss'], info['action_loss_meter'], info['activity_loss_meter'], info['time']))

                
################### optimizer ###################              

def list_features(net_, has_train_bbox=False):
    if has_train_bbox:
        raise Exception('To-do-bbox_training!')
    else:
        backbone_features = list(net_.backbone.parameters())
        backbone_features_len = len(backbone_features)
        net_features = list(net_.parameters())[backbone_features_len:]
        print('backbone feature length:', backbone_features_len)
        print('net main feature length:', len(net_features))
        return backbone_features, net_features

def get_optimizer(mode, opts, backbone_features, net_features, optim_class='adam'):
    if mode == 1:
        set_trainable_param(backbone_features, True)
        set_trainable_param(net_features, True)
        if optim_class == 'adam':
            optimizer = torch.optim.Adam([
                    {'params': backbone_features},
                    {'params': net_features},
                    ], lr=opts.train_learning_rate, weight_decay=opts.weight_decay)
        else:
            optimizer = torch.optim.SGD([
                    {'params': backbone_features},
                    {'params': net_features},
                    ], lr=opts.train_learning_rate, momentum=opts.momentum, weight_decay=opts.weight_decay, nesterov=opts.nesterov)    
                
    elif mode == 2:
        set_trainable_param(backbone_features, False)
        set_trainable_param(net_features, True)
        if optim_class == 'adam':
            optimizer = torch.optim.Adam([
                    {'params': net_features},
                    ], lr=opts.train_learning_rate, weight_decay=opts.weight_decay)
        else:
            optimizer = torch.optim.SGD([
                    {'params': net_features},
                    ], lr=opts.train_learning_rate, momentum=opts.momentum, weight_decay=opts.weight_decay, nesterov=opts.nesterov)    
    else:
        raise Exception('Unrecognized optimization mode specified!')
    return optimizer
    
def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)
    return totalnorm

def avg_gradient(model, iter_size):
    """Computes a gradient clipping coefficient based on gradient norm."""
    if iter_size >1:
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.div_(iter_size)
                
################### loss ###################       
    
def build_loss_cls(cls_score, labels, loss_weight=None):
    s_labels = labels.squeeze().to(device=cls_score.device)
    cross_entropy = F.cross_entropy(cls_score, s_labels, weight=loss_weight)
    maxv, predict = cls_score.data.max(1)
    cnt_ac = torch.sum(s_labels.eq(predict).float())
    return cross_entropy, cnt_ac

def calcu_fg(cls_score, labels):
    s_labels = labels.squeeze()
    fg_cnt = torch.sum(s_labels.data.ne(0))
    bg_cnt = s_labels.data.numel() - fg_cnt
    maxv, predict = cls_score.data.max(1)
    if fg_cnt == 0:
        tp = torch.zeros_like(fg_cnt)
    else:
        tp = torch.sum(predict[:fg_cnt].eq(s_labels.data[:fg_cnt]))
    tf = torch.sum(predict[fg_cnt:].eq(s_labels.data[fg_cnt:]))
    return tp, tf, fg_cnt, bg_cnt
    
def build_loss_bbox(bbox_pred, roi_data, fg_cnt):
    bbox_targets, bbox_inside_weights = roi_data[2:4]
    bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)    ### find the most overlap gt_labels
    bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)
    ###loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-5)
    ###!!!!!!!! use Bounding Box Regression with Uncertainty for Accurate Object Detection   !!!!!!!
    loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-5)
    return loss_box
    
################### Meter ################### 
    
class AverageMeter(object):
    """
    Computes the average value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class Timer(object):
    """
    class to do timekeeping
    """
    def __init__(self):
        self.last_time=time.time()
        
    def timeit(self):
        old_time=self.last_time
        self.last_time=time.time()
        return self.last_time-old_time
        
    def reset(self):
        self.last_time=time.time()
        
        
         