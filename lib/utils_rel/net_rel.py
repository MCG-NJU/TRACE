# Adapted by Ji Zhang in 2019
#
# Based on Detectron.pytorch/lib/utils/net.py written by Roy Tseng

import logging
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from utils.net import _get_lr_change_ratio
from utils.net import _CorrectMomentum

logger = logging.getLogger(__name__)

def class_coco2ag():
    a = [0, 1, -1, 60, -1, 74, -1, -1, 57, -1, -1, 40, -1, -1, -1, -1, -1, -1, -1, 64, -1, -1, -1, -1, 68, -1, -1, 73, 49, -1, -1, 58, 61, -1, -1, -1, -1]
    a = np.array(a, dtype=np.int)
    idx = np.where(a >= 0)[0]
    a = a[idx] - 1
    return a, idx

# Detectron ---> Detectron2
def make_res101_map(m_name):
    s = ''
    if m_name.find('Conv_Body.res1.conv1') >= 0:
        s = 'backbone.stem.conv1.weight'
    elif m_name.find('Conv_Body.res1.bn') >= 0:
        m_tail = m_name.split('.')[-1]
        s = 'backbone.stem.conv1.norm.' + m_tail
    elif m_name.find('Conv_Body.res') >= 0:
        m_name_list = m_name.split('.')
        m_tail = m_name_list[-1]
        
        res_id_id = m_name.find('res') + 3
        res_id = m_name[res_id_id]
        res_m_id = m_name[res_id_id + 2]
        
        s = 'backbone.res' + res_id + '.' + res_m_id + '.'
        if m_name.find('conv') >= 0:
        
            conv_id_id = m_name.find('conv') + 4
            conv_id = m_name[conv_id_id]
            
            s = s + 'conv' + conv_id + '.' + m_tail
        elif m_name.find('bn') >= 0:
            
            bn_id_id = m_name.find('bn') + 2
            bn_id = m_name[bn_id_id]
            
            s = s + 'conv' + bn_id + '.norm.' + m_tail
        elif m_name.find('downsample') >= 0:
            
            downsample_m_id_id = m_name.find('downsample') + 10 + 1
            downsample_m_id = m_name[downsample_m_id_id]
            
            if downsample_m_id == '0':
                s = s + 'shortcut.' + m_tail
            elif downsample_m_id == '1':
                s = s + 'shortcut.norm.' + m_tail
    elif m_name.find('RPN.RPN_conv') >= 0:    
        m_name_list = m_name.split('.')
        s = 'proposal_generator.rpn_head.conv.' + m_name_list[-1]
    elif m_name.find('RPN.RPN_cls_score') >= 0:    
        m_name_list = m_name.split('.')
        s = 'proposal_generator.rpn_head.objectness_logits.' + m_name_list[-1]
    elif m_name.find('RPN.RPN_bbox_pred') >= 0:    
        m_name_list = m_name.split('.')
        s = 'proposal_generator.rpn_head.anchor_deltas.' + m_name_list[-1]
    elif m_name.find('Box_Head.res') >= 0:    
        m_name_list = m_name.split('.')
        if m_name.find('bn') >= 0:
            bn_id_id = m_name.find('bn') + 2
            bn_id = m_name[bn_id_id]
            s = 'roi_heads.res5.' + m_name_list[-3] + '.conv' + bn_id + '.norm.' + m_name_list[-1]
        elif m_name.find('downsample') >= 0:
            downsample_m_id_id = m_name.find('downsample') + 10 + 1
            downsample_m_id = m_name[downsample_m_id_id]
            if downsample_m_id == '0':
                s = 'roi_heads.res5.' + m_name_list[-4] + '.shortcut.' + m_name_list[-1]
            elif downsample_m_id == '1':
                s = 'roi_heads.res5.' + m_name_list[-4] + '.shortcut.norm.' + m_name_list[-1]
        else:
            s = 'roi_heads.res5.' + m_name_list[-3] + '.' + m_name_list[-2] + '.' + m_name_list[-1]
    elif m_name.find('Box_Head') >= 0:    
        m_name_list = m_name.split('.')
        s = 'roi_heads.box_head.' + m_name_list[-2] + '.' + m_name_list[-1]
    elif m_name.find('Box_Outs') >= 0:    
        m_name_list = m_name.split('.')
        s = 'roi_heads.box_predictor.' + m_name_list[-2] + '.' + m_name_list[-1]
    return s
    
def make_res50_fpn_map(m_name):
    s = ''
    m_name_list = m_name.split('.')
    if m_name.find('Conv_Body.conv_top') >= 0:
        s = 'backbone.fpn_lateral5.' + m_name_list[-1]
    elif m_name.find('Conv_Body.topdown_lateral_modules') >= 0:
        id = 4 - int(m_name_list[-3])
        s = 'backbone.fpn_lateral' + str(id) + '.' + m_name_list[-1]
    elif m_name.find('Conv_Body.posthoc_modules') >= 0:
        ###!!!id = int(m_name_list[-2]) + 2
        id = 5 - int(m_name_list[-2])
        s = 'backbone.fpn_output' + str(id) + '.' + m_name_list[-1]
    elif m_name.find('conv_body.res1.conv1') >= 0:
        s = 'backbone.bottom_up.stem.conv1.weight'
    elif m_name.find('conv_body.res1.bn') >= 0:
        m_tail = m_name.split('.')[-1]
        s = 'backbone.bottom_up.stem.conv1.norm.' + m_tail
    elif m_name.find('conv_body.res') >= 0:
        m_tail = m_name_list[-1]
        res_id_id = m_name.find('res') + 3
        res_id = m_name[res_id_id]
        res_m_id = m_name[res_id_id + 2]
        s = 'backbone.bottom_up.res' + res_id + '.' + res_m_id + '.'
        if m_name_list[-2].find('conv') >= 0:
            conv_id = m_name_list[-2][-1]
            s = s + 'conv' + conv_id + '.' + m_tail
        elif m_name.find('bn') >= 0:
            bn_id_id = m_name.find('bn') + 2
            bn_id = m_name[bn_id_id]
            s = s + 'conv' + bn_id + '.norm.' + m_tail
        elif m_name.find('downsample') >= 0:
            downsample_m_id_id = m_name.find('downsample') + 10 + 1
            downsample_m_id = m_name[downsample_m_id_id]
            if downsample_m_id == '0':
                s = s + 'shortcut.' + m_tail
            elif downsample_m_id == '1':
                s = s + 'shortcut.norm.' + m_tail
    elif m_name.find('RPN.FPN_RPN_conv') >= 0:    
        m_name_list = m_name.split('.')
        s = 'proposal_generator.rpn_head.conv.' + m_name_list[-1]
    elif m_name.find('RPN.FPN_RPN_cls_score') >= 0:    
        m_name_list = m_name.split('.')
        s = 'proposal_generator.rpn_head.objectness_logits.' + m_name_list[-1]
    elif m_name.find('RPN.FPN_RPN_bbox_pred') >= 0:    
        m_name_list = m_name.split('.')
        s = 'proposal_generator.rpn_head.anchor_deltas.' + m_name_list[-1]
    elif m_name.find('Box_Head') >= 0:    
        m_name_list = m_name.split('.')
        s = 'roi_heads.box_head.' + m_name_list[-2] + '.' + m_name_list[-1]
    elif m_name.find('Box_Outs') >= 0:    
        m_name_list = m_name.split('.')
        s = 'roi_heads.box_predictor.' + m_name_list[-2] + '.' + m_name_list[-1]
    return s

def make_backbone_map(m_name):
    return 'backbone.' + m_name
    
def update_learning_rate_att(optimizer, cur_lr, new_lr):
    """Update learning rate"""
    if cur_lr != new_lr:
        ratio = _get_lr_change_ratio(cur_lr, new_lr)
        if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
            logger.info('Changing learning rate %.6f -> %.6f', cur_lr, new_lr)
        # Update learning rate, note that different parameter may have different learning rate
        param_keys = []
        for ind, param_group in enumerate(optimizer.param_groups):
            if (ind == 1 or ind == 3) and cfg.SOLVER.BIAS_DOUBLE_LR:  # bias params
                param_group['lr'] = new_lr * 2
            else:
                param_group['lr'] = new_lr
            if ind <= 1:  # backbone params
                param_group['lr'] = cfg.SOLVER.BACKBONE_LR_SCALAR * param_group['lr']  # 0.1 * param_group['lr']
            param_keys += param_group['params']
        if cfg.SOLVER.TYPE in ['SGD'] and cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
            _CorrectMomentum(optimizer, param_keys, new_lr / cur_lr)
            

def update_learning_rate_rel(optimizer, cur_lr, new_lr):
    """Update learning rate"""
    if cur_lr != new_lr:
        ratio = _get_lr_change_ratio(cur_lr, new_lr)
        if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
            logger.info('Changing learning rate %.6f -> %.6f', cur_lr, new_lr)
        # Update learning rate, note that different parameter may have different learning rate
        param_keys = []
        for ind, param_group in enumerate(optimizer.param_groups):
            if (ind == 1 or ind == 3) and cfg.SOLVER.BIAS_DOUBLE_LR:  # bias params
                param_group['lr'] = new_lr * 2
            else:
                param_group['lr'] = new_lr
            if ind <= 1:  # backbone params
                param_group['lr'] = cfg.SOLVER.BACKBONE_LR_SCALAR * param_group['lr']  # 0.1 * param_group['lr']
            param_keys += param_group['params']
        if cfg.SOLVER.TYPE in ['SGD'] and cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
            _CorrectMomentum(optimizer, param_keys, new_lr / cur_lr)


def trans_prd(ckpt, find_s='Prd_RCNN.'):
    flg = False
    for k,v in ckpt.items():
        if k.find(find_s) >= 0:
            flg = True
            break
    if flg:
        new_ckpt = dict()
        for k,v in ckpt.items():
            if k.find(find_s) >= 0:
                param = np.asarray(ckpt[k])
                new_ckpt[k[k.find(find_s)+len(find_s):]] = param
        return new_ckpt
    else:
        return ckpt
            
def load_ckpt_rel(model, ckpt):
    """Load checkpoint"""
    c2a, c2a_idx = class_coco2ag()
    for k, v in model.state_dict().items():
        kk = k
        if kk not in ckpt:
            kk = make_res101_map(k)
        if kk not in ckpt:
            kk = make_res50_fpn_map(k)
        if kk not in ckpt:
            kk = make_backbone_map(k)
        
        if kk in ckpt:
            param = torch.from_numpy(np.asarray(ckpt[kk]))
            if v.shape == param.shape:
                v.copy_(param)
                #print('Parameter[{}] match Parameter[{}]. Param size: {} --> {}'.format(k, kk, v.shape, param.shape))
            else:
                if kk.find('roi_heads.box_predictor') >=0 or kk.find('Box_Outs.cls_score.weight') >= 0:
                    v[c2a_idx].copy_(param[c2a])
                else:
                    print('[Loaded net not complete] Parameter[{}] Size Mismatch. Param size: {} -x-> {}'.format(kk, v.shape, param.shape)) 
        #else:
        #    print('[Loaded net not complete] Parameter[{}] -x-> [{}]'.format(kk, k))     
    ## model.load_state_dict(ckpt, strict=False)
    #for k, v in ckpt.items():
    #    print(k)
    #assert False
