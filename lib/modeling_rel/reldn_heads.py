# Written by Ji Zhang in 2019

import numpy as np
from numpy import linalg as la
import math
import logging
import json

import nn as mynn
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


from core.config import cfg
from modeling_rel.sparse_targets_rel import FrequencyBias

logger = logging.getLogger(__name__)


class reldn_head(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        dim_in_final = dim_in // 3
        self.dim_in_final = dim_in_final
        
        
        if cfg.MODEL.MULTI_RELATION:
            num_prd_classes = cfg.MODEL.NUM_PRD_CLASSES
        else:
            if cfg.MODEL.USE_BG:
                num_prd_classes = cfg.MODEL.NUM_PRD_CLASSES + 1
            else:
                num_prd_classes = cfg.MODEL.NUM_PRD_CLASSES
        
            
        if cfg.MODEL.RUN_BASELINE:
            # only run it on testing mode
            self.freq_bias = FrequencyBias(cfg.TEST.DATASETS[0])
            return
        
        self.prd_cls_feats = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(dim_in // 2, dim_in_final),
            nn.LeakyReLU(0.1))
        self.prd_cls_scores = nn.Linear(dim_in_final, num_prd_classes)
        
        if cfg.MODEL.USE_FREQ_BIAS:
            # Assume we are training/testing on only one dataset
            if len(cfg.TRAIN.DATASETS):
                self.freq_bias = FrequencyBias(cfg.TRAIN.DATASETS[0])
            else:
                self.freq_bias = FrequencyBias(cfg.TEST.DATASETS[0])

        if cfg.MODEL.USE_SPATIAL_FEAT:
            self.spt_cls_feats = nn.Sequential(
                nn.Linear(28, 64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, 64),
                nn.LeakyReLU(0.1))
            self.spt_cls_scores = nn.Linear(64, num_prd_classes)
        
        if cfg.MODEL.ADD_SO_SCORES:
            self.prd_sbj_scores = nn.Linear(dim_in_final, num_prd_classes)
            self.prd_obj_scores = nn.Linear(dim_in_final, num_prd_classes)
        
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                mynn.init.XavierFill(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # spo_feat will be concatenation of SPO
    def forward(self, spo_feat, spt_feat=None, sbj_labels=None, obj_labels=None, sbj_feat=None, obj_feat=None):

        device_id = spo_feat.get_device()
        if sbj_labels is not None:
            sbj_labels = Variable(torch.from_numpy(sbj_labels.astype('int64'))).cuda(device_id)
        if obj_labels is not None:
            obj_labels = Variable(torch.from_numpy(obj_labels.astype('int64'))).cuda(device_id)

        if cfg.MODEL.RUN_BASELINE:
            assert sbj_labels is not None and obj_labels is not None
            prd_cls_scores = self.freq_bias.rel_index_with_labels(torch.stack((sbj_labels, obj_labels), 1))
            prd_cls_scores = F.softmax(prd_cls_scores, dim=1)
            return prd_cls_scores, prd_cls_scores, None, prd_cls_scores, None, None
        
        if spo_feat.dim() == 4:
            spo_feat = spo_feat.squeeze(3).squeeze(2)
        prd_cls_feats = self.prd_cls_feats(spo_feat)
        prd_vis_scores = self.prd_cls_scores(prd_cls_feats)
        sbj_cls_scores = None
        obj_cls_scores = None
            
        if cfg.MODEL.USE_FREQ_BIAS:
            assert sbj_labels is not None and obj_labels is not None
            prd_bias_scores = self.freq_bias.rel_index_with_labels(torch.stack((sbj_labels, obj_labels), 1))
        
        if cfg.MODEL.USE_SPATIAL_FEAT:
            assert spt_feat is not None
            device_id = spo_feat.get_device()
            spt_feat = Variable(torch.from_numpy(spt_feat.astype('float32'))).cuda(device_id)
            spt_cls_feats = self.spt_cls_feats(spt_feat)
            prd_spt_scores = self.spt_cls_scores(spt_cls_feats)
        else:
            prd_spt_scores = None
            
        if cfg.MODEL.ADD_SO_SCORES:
            prd_sbj_scores = self.prd_sbj_scores(sbj_feat)
            prd_obj_scores = self.prd_obj_scores(obj_feat)
            
        if cfg.MODEL.ADD_SCORES_ALL:
            ttl_cls_scores = torch.tensor(prd_vis_scores)
            if cfg.MODEL.USE_FREQ_BIAS:
                ttl_cls_scores += prd_bias_scores
            if cfg.MODEL.USE_SPATIAL_FEAT:
                ttl_cls_scores += prd_spt_scores
            if cfg.MODEL.ADD_SO_SCORES:
                ttl_cls_scores += prd_sbj_scores + prd_obj_scores
        else:
            ttl_cls_scores = None
            
        if not self.training:
            if cfg.MODEL.MULTI_RELATION:
                prd_vis_scores = self.sigmoid(prd_vis_scores)
                if cfg.MODEL.USE_FREQ_BIAS:
                    prd_bias_scores = self.sigmoid(prd_bias_scores)
                if cfg.MODEL.USE_SPATIAL_FEAT:
                    prd_spt_scores = self.sigmoid(prd_spt_scores)
                if cfg.MODEL.ADD_SCORES_ALL:
                    ttl_cls_scores = self.sigmoid(ttl_cls_scores)
            else:
                prd_vis_scores = F.softmax(prd_vis_scores, dim=1)
                if cfg.MODEL.USE_FREQ_BIAS:
                    prd_bias_scores = F.softmax(prd_bias_scores, dim=1)
                if cfg.MODEL.USE_SPATIAL_FEAT:
                    prd_spt_scores = F.softmax(prd_spt_scores, dim=1)
                if cfg.MODEL.ADD_SCORES_ALL:
                    ttl_cls_scores = F.softmax(ttl_cls_scores, dim=1)
            
            
        return prd_vis_scores, prd_bias_scores, prd_spt_scores, ttl_cls_scores, sbj_cls_scores, obj_cls_scores


def binary_prd_losses(ttl_cls_scores, multi_prd_labels_int32, keep_pair_class, pos_weight=None):
    device_id = ttl_cls_scores.get_device()
    #prd_cls_scores = ttl_cls_scores[np.arange(0, ttl_cls_scores.shape[0]), keep_pair_class]
    #print(keep_pair_class)
    #print(keep_pair_class.shape)
    #keep_pair_class_id = torch.from_numpy(keep_pair_class.astype(np.int64)).cuda(device_id)
    
    #print(keep_pair_class.max(), keep_pair_class.min())
    
    keep_pair_class_id = torch.tensor(keep_pair_class).long().cuda(device_id)
    keep_pair_class_id = keep_pair_class_id.view(-1, 1)
    prd_cls_scores = torch.gather(ttl_cls_scores, dim=1, index=keep_pair_class_id)
    prd_cls_scores = prd_cls_scores.view(-1)
    if pos_weight is None:
        binary_loss_function = nn.BCEWithLogitsLoss()
    else:
        binary_loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight.cuda(device_id))
    
    #zz = np.where(multi_prd_labels_int32 > 0)[0]
    #ff = np.where(multi_prd_labels_int32 <= 0)[0]
    #print(len(zz), len(ff), (float(len(zz)) / (float(len(multi_prd_labels_int32)) + 1e-12)))
    #print(keep_pair_class)
    #print(multi_prd_labels_int32)
    #assert False
    
    prd_labels = Variable(torch.from_numpy(multi_prd_labels_int32.astype(np.float32))).cuda(device_id)
    loss_cls_prd = binary_loss_function(prd_cls_scores, prd_labels)
    # class accuracy
    #prd_cls_preds = (prd_cls_scores >= 0).type_as(prd_labels)
    #accuracy_cls_prd = prd_cls_preds.eq(prd_labels).float().mean(dim=0)
    accuracy_cls_prd = None
    
    return loss_cls_prd, accuracy_cls_prd

def triplet_prd_loss(ttl_cls_scores, multi_prd_labels_int32, keep_pair_class, s_score, o_score, pos_weight=None):
    device_id = ttl_cls_scores.get_device()
    
    #prd_cls_scores = ttl_cls_scores[np.arange(0, ttl_cls_scores.shape[0]), keep_pair_class]
    keep_pair_class_id = torch.from_numpy(keep_pair_class.astype(np.int64)).cuda(device_id)
    keep_pair_class_id = keep_pair_class_id.view(-1, 1)
    prd_cls_scores = torch.gather(ttl_cls_scores, dim=1, index=keep_pair_class_id)
    prd_cls_scores = prd_cls_scores.view(-1)
    
    prd_cls_scores = prd_cls_scores * s_score * o_score
    
    if pos_weight is None:
        binary_loss_function = nn.BCEWithLogitsLoss()
    else:
        binary_loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight.cuda(device_id))
    prd_labels = Variable(torch.from_numpy(multi_prd_labels_int32.astype(np.float32))).cuda(device_id)
    loss_cls_prd = binary_loss_function(prd_cls_scores, prd_labels)
    return loss_cls_prd

def smooth_l1(x, sigma=1):
    beta = 1. / (sigma ** 2)
    y = torch.where(x < beta, 0.5 * x ** 2 / beta, torch.abs(x) - 0.5 * beta)
    return y
    
def rel_compet_loss(ttl_cls_scores, multi_prd_labels_int32, keep_pair_class, rel_mat, rel_mat_cou):
    push_away_pos_loss, push_away_neg_loss, push_close_pos_loss, push_close_neg_loss = 0, 0, 0, 0
    normd_ttl_cls_scores = nn.Sigmoid()(ttl_cls_scores)
    keep_pair_n_id, keep_pair_n_class = np.where(
                (rel_mat[keep_pair_class] < 0.1) & 
                (rel_mat[keep_pair_class] >= -1e-4) & 
                (rel_mat_cou[keep_pair_class] > 0)
                )
    push_away_pr_o = normd_ttl_cls_scores[keep_pair_n_id, keep_pair_n_class]
    push_away_pr_s = normd_ttl_cls_scores[keep_pair_n_id, keep_pair_class[keep_pair_n_id]]
    push_away_y = multi_prd_labels_int32[keep_pair_n_id]
    pos_id = np.where(push_away_y > 0)[0]
    neg_id = np.where(push_away_y <= 0)[0]
    if len(pos_id) > 0:
        #push_away_pos_loss = \
        #    torch.mean(smooth_l1(push_away_pr_s[pos_id] - (1 - push_away_pr_o[pos_id])))
        #push_away_pos_loss = -torch.mean(torch.log(1 - push_away_pr_o[pos_id]))
        push_away_pos_loss = torch.mean(smooth_l1(push_away_pr_o[pos_id]))
    if len(neg_id) > 0:
        #push_away_neg_loss = \
        #    torch.mean(smooth_l1((1 - push_away_pr_s[neg_id]) - push_away_pr_o[neg_id]))
        #push_away_neg_loss = -torch.mean(torch.log(push_away_pr_o[neg_id]))
        push_away_neg_loss = torch.mean(smooth_l1(1 - push_away_pr_o[neg_id]))
            
    keep_pair_p_id, keep_pair_p_class = np.where(
                (rel_mat[keep_pair_class] > 0.9) & 
                (rel_mat_cou[keep_pair_class] > 0)
                )
    push_close_pr_o = normd_ttl_cls_scores[keep_pair_p_id, keep_pair_p_class]
    push_close_pr_s = normd_ttl_cls_scores[keep_pair_p_id, keep_pair_class[keep_pair_p_id]]
    push_close_y = multi_prd_labels_int32[keep_pair_p_id]
    push_close_pos_id = np.where(push_close_y > 0)[0]
    push_close_neg_id = np.where(push_close_y <= 0)[0]
    if len(push_close_pos_id) > 0:
        #push_close_pos_loss = \
        #    torch.mean(smooth_l1(push_close_pr_s[push_close_pos_id] - push_close_pr_o[push_close_pos_id]))
        #push_close_pos_loss = -torch.mean(torch.log(push_close_pr_o[push_close_pos_id]))
        push_close_pos_loss = torch.mean(smooth_l1(1 - push_close_pr_o[push_close_pos_id]))
    if len(push_close_neg_id) > 0:
        #push_close_neg_loss = \
        #    torch.mean(smooth_l1((1 - push_close_pr_s[push_close_neg_id]) - (1 - push_close_pr_o[push_close_neg_id])))
        #push_close_neg_loss = -torch.mean(torch.log(1 - push_close_pr_o[push_close_neg_id]))
        push_close_neg_loss = torch.mean(smooth_l1(push_close_pr_o[push_close_neg_id]))
        
    loss_rel_push = push_away_pos_loss + push_away_neg_loss + push_close_pos_loss + push_close_neg_loss
    
    #print(keep_pair_n_id.shape) # (18716,) (20088,)
    #print(push_close_pr_o.shape) # torch.Size([3800]) torch.Size([1849])
    #print(push_close_pr_s.shape) # torch.Size([3800]) torch.Size([1849])
    #print(push_away_pr_o.shape) #   torch.Size([20088])
    #print(push_away_pr_s.shape) #   torch.Size([20088])
    #print(pos_id.shape) #   (5644,)
    #print(neg_id.shape) #   (14444,)
    #print(push_close_pos_id.shape) #   (0,)
    #print(push_close_neg_id.shape) #   (1849,)
    #print(push_away_pos_loss) # tensor(2.5041, device='cuda:0', grad_fn=<MeanBackward0>)
    #print(push_away_neg_loss) # tensor(3.1201, device='cuda:0', grad_fn=<MeanBackward0>)
    #print(push_close_pos_loss) # tensor(1.8889, device='cuda:0', grad_fn=<MeanBackward0>) tensor(nan, device='cuda:0', grad_fn=<MeanBackward0>)
    #print(push_close_neg_loss) # tensor(0.7219, device='cuda:0', grad_fn=<MeanBackward0>)
    #print()
    #print(loss_rel_push)
    return loss_rel_push
    
    
    
def reldn_losses(prd_cls_scores, prd_labels_int32, fg_only=False):
    #print(prd_labels_int32.shape) #(1571,)
    #print(prd_labels_int32) #[29 29 29 ...  0  0  0]
    device_id = prd_cls_scores.get_device()
    prd_labels = Variable(torch.from_numpy(prd_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_prd = F.cross_entropy(prd_cls_scores, prd_labels)
    # class accuracy
    prd_cls_preds = prd_cls_scores.max(dim=1)[1].type_as(prd_labels)
    accuracy_cls_prd = prd_cls_preds.eq(prd_labels).float().mean(dim=0)

    return loss_cls_prd, accuracy_cls_prd


def reldn_contrastive_losses(prd_scores_sbj_pos, prd_scores_obj_pos, rel_ret):
    # sbj
    if cfg.MODEL.MULTI_RELATION:
        prd_probs_sbj_pos = F.sigmoid(prd_scores_sbj_pos)
    else:
        prd_probs_sbj_pos = F.softmax(prd_scores_sbj_pos, dim=1)
    sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target = split_pos_neg_spo_agnostic(
        prd_probs_sbj_pos, rel_ret['binary_labels_sbj_pos_int32'], rel_ret['inds_unique_sbj_pos'], rel_ret['inds_reverse_sbj_pos'])
    sbj_contrastive_loss = F.margin_ranking_loss(sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target, margin=cfg.MODEL.NODE_CONTRASTIVE_MARGIN)
    # obj
    if cfg.MODEL.MULTI_RELATION:
        prd_probs_obj_pos = F.sigmoid(prd_scores_obj_pos)
    else:
        prd_probs_obj_pos = F.softmax(prd_scores_obj_pos, dim=1)
    obj_pair_pos_batch, obj_pair_neg_batch, obj_target = split_pos_neg_spo_agnostic(
        prd_probs_obj_pos, rel_ret['binary_labels_obj_pos_int32'], rel_ret['inds_unique_obj_pos'], rel_ret['inds_reverse_obj_pos'])
    obj_contrastive_loss = F.margin_ranking_loss(obj_pair_pos_batch, obj_pair_neg_batch, obj_target, margin=cfg.MODEL.NODE_CONTRASTIVE_MARGIN)
    
    return sbj_contrastive_loss, obj_contrastive_loss


def reldn_so_contrastive_losses(prd_scores_sbj_pos, prd_scores_obj_pos, rel_ret):
    # sbj
    if cfg.MODEL.MULTI_RELATION:
        prd_probs_sbj_pos = F.sigmoid(prd_scores_sbj_pos)
    else:
        prd_probs_sbj_pos = F.softmax(prd_scores_sbj_pos, dim=1)
    sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target = split_pos_neg_so_aware(
        prd_probs_sbj_pos,
        rel_ret['binary_labels_sbj_pos_int32'], rel_ret['inds_unique_sbj_pos'], rel_ret['inds_reverse_sbj_pos'],
        rel_ret['sbj_labels_sbj_pos_int32'], rel_ret['obj_labels_sbj_pos_int32'], 's')
    sbj_so_contrastive_loss = F.margin_ranking_loss(sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target, margin=cfg.MODEL.NODE_CONTRASTIVE_SO_AWARE_MARGIN)
    # obj
    if cfg.MODEL.MULTI_RELATION:
        prd_probs_obj_pos = F.sigmoid(prd_scores_obj_pos)
    else:
        prd_probs_obj_pos = F.softmax(prd_scores_obj_pos, dim=1)
    obj_pair_pos_batch, obj_pair_neg_batch, obj_target = split_pos_neg_so_aware(
        prd_probs_obj_pos,
        rel_ret['binary_labels_obj_pos_int32'], rel_ret['inds_unique_obj_pos'], rel_ret['inds_reverse_obj_pos'],
        rel_ret['sbj_labels_obj_pos_int32'], rel_ret['obj_labels_obj_pos_int32'], 'o')
    obj_so_contrastive_loss = F.margin_ranking_loss(obj_pair_pos_batch, obj_pair_neg_batch, obj_target, margin=cfg.MODEL.NODE_CONTRASTIVE_SO_AWARE_MARGIN)
    
    return sbj_so_contrastive_loss, obj_so_contrastive_loss


def reldn_p_contrastive_losses(prd_scores_sbj_pos, prd_scores_obj_pos, prd_bias_scores_sbj_pos, prd_bias_scores_obj_pos, rel_ret):
    # sbj
    if cfg.MODEL.MULTI_RELATION:
        prd_probs_sbj_pos = F.sigmoid(prd_scores_sbj_pos)
    else:
        prd_probs_sbj_pos = F.softmax(prd_scores_sbj_pos, dim=1)
    if cfg.MODEL.MULTI_RELATION:
        prd_bias_probs_sbj_pos = F.sigmoid(prd_bias_scores_sbj_pos)
    else:
        prd_bias_probs_sbj_pos = F.softmax(prd_bias_scores_sbj_pos, dim=1)
    sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target = split_pos_neg_p_aware(
        prd_probs_sbj_pos,
        prd_bias_probs_sbj_pos,
        rel_ret['binary_labels_sbj_pos_int32'], rel_ret['inds_unique_sbj_pos'], rel_ret['inds_reverse_sbj_pos'],
        rel_ret['prd_labels_sbj_pos_int32'])
    sbj_p_contrastive_loss = F.margin_ranking_loss(sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target, margin=cfg.MODEL.NODE_CONTRASTIVE_P_AWARE_MARGIN)
    # obj
    if cfg.MODEL.MULTI_RELATION:
        prd_probs_obj_pos = F.sigmoid(prd_scores_obj_pos)
    else:
        prd_probs_obj_pos = F.softmax(prd_scores_obj_pos, dim=1)
    if cfg.MODEL.MULTI_RELATION:
        prd_bias_probs_obj_pos = F.sigmoid(prd_bias_scores_obj_pos)
    else:
        prd_bias_probs_obj_pos = F.softmax(prd_bias_scores_obj_pos, dim=1)
    obj_pair_pos_batch, obj_pair_neg_batch, obj_target = split_pos_neg_p_aware(
        prd_probs_obj_pos,
        prd_bias_probs_obj_pos,
        rel_ret['binary_labels_obj_pos_int32'], rel_ret['inds_unique_obj_pos'], rel_ret['inds_reverse_obj_pos'],
        rel_ret['prd_labels_obj_pos_int32'])
    obj_p_contrastive_loss = F.margin_ranking_loss(obj_pair_pos_batch, obj_pair_neg_batch, obj_target, margin=cfg.MODEL.NODE_CONTRASTIVE_P_AWARE_MARGIN)
    
    return sbj_p_contrastive_loss, obj_p_contrastive_loss


def split_pos_neg_spo_agnostic(prd_probs, binary_labels_pos, inds_unique_pos, inds_reverse_pos):
    device_id = prd_probs.get_device()
    if cfg.MODEL.MULTI_RELATION:
        prd_pos_probs = 1 - torch.prod(1 - prd_probs, dim=1)
    else:
        prd_pos_probs = 1 - prd_probs[:, 0]  # shape is (#rels,)
    # loop over each group
    pair_pos_batch = torch.ones(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    pair_neg_batch = torch.zeros(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    for i in range(inds_unique_pos.shape[0]):
        inds = np.where(inds_reverse_pos == i)[0]
        prd_pos_probs_i = prd_pos_probs[inds]
        binary_labels_pos_i = binary_labels_pos[inds]
        pair_pos_inds = np.where(binary_labels_pos_i > 0)[0]
        pair_neg_inds = np.where(binary_labels_pos_i == 0)[0]
        if pair_pos_inds.size == 0 or pair_neg_inds.size == 0:  # ignore this node if either pos or neg does not exist
            continue
        prd_pos_probs_i_pair_pos = prd_pos_probs_i[pair_pos_inds]
        prd_pos_probs_i_pair_neg = prd_pos_probs_i[pair_neg_inds]
        min_prd_pos_probs_i_pair_pos = torch.min(prd_pos_probs_i_pair_pos)
        max_prd_pos_probs_i_pair_neg = torch.max(prd_pos_probs_i_pair_neg)
        pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pair_pos.unsqueeze(0)))
        pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))

    target = torch.ones_like(pair_pos_batch).cuda(device_id)
        
    return pair_pos_batch, pair_neg_batch, target


def split_pos_neg_so_aware(prd_probs, binary_labels_pos, inds_unique_pos, inds_reverse_pos, sbj_labels_pos, obj_labels_pos, s_or_o):
    device_id = prd_probs.get_device()
    if cfg.MODEL.MULTI_RELATION:
        prd_pos_probs = 1 - torch.prod(1 - prd_probs, dim=1)
    else:
        prd_pos_probs = 1 - prd_probs[:, 0]  # shape is (#rels,)
    # loop over each group
    pair_pos_batch = torch.ones(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    pair_neg_batch = torch.zeros(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    for i in range(inds_unique_pos.shape[0]):
        inds = np.where(inds_reverse_pos == i)[0]
        prd_pos_probs_i = prd_pos_probs[inds]
        binary_labels_pos_i = binary_labels_pos[inds]
        sbj_labels_pos_i = sbj_labels_pos[inds]
        obj_labels_pos_i = obj_labels_pos[inds]
        pair_pos_inds = np.where(binary_labels_pos_i > 0)[0]
        pair_neg_inds = np.where(binary_labels_pos_i == 0)[0]
        if pair_pos_inds.size == 0 or pair_neg_inds.size == 0:  # ignore this node if either pos or neg does not exist
            continue
        prd_pos_probs_i_pair_pos = prd_pos_probs_i[pair_pos_inds]
        prd_pos_probs_i_pair_neg = prd_pos_probs_i[pair_neg_inds]
        sbj_labels_i_pair_pos = sbj_labels_pos_i[pair_pos_inds]
        obj_labels_i_pair_pos = obj_labels_pos_i[pair_pos_inds]
        sbj_labels_i_pair_neg = sbj_labels_pos_i[pair_neg_inds]
        obj_labels_i_pair_neg = obj_labels_pos_i[pair_neg_inds]
        max_prd_pos_probs_i_pair_neg = torch.max(prd_pos_probs_i_pair_neg)  # this is fixed for a given i
        if s_or_o == 's':
            # get all unique object labels
            unique_obj_labels, inds_unique_obj_labels, inds_reverse_obj_labels = np.unique(
                obj_labels_i_pair_pos, return_index=True, return_inverse=True, axis=0)
            for j in range(inds_unique_obj_labels.shape[0]):
                # get min pos
                inds_j = np.where(inds_reverse_obj_labels == j)[0]
                prd_pos_probs_i_pos_j = prd_pos_probs_i_pair_pos[inds_j]
                min_prd_pos_probs_i_pos_j = torch.min(prd_pos_probs_i_pos_j)
                # get max neg
                neg_j_inds = np.where(obj_labels_i_pair_neg == unique_obj_labels[j])[0]
                if neg_j_inds.size == 0:
                    if cfg.MODEL.USE_SPO_AGNOSTIC_COMPENSATION:
                        pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                        pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))
                    continue
                prd_pos_probs_i_neg_j = prd_pos_probs_i_pair_neg[neg_j_inds]
                max_prd_pos_probs_i_neg_j = torch.max(prd_pos_probs_i_neg_j)
                pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_neg_j.unsqueeze(0)))
        else:
            # get all unique subject labels
            unique_sbj_labels, inds_unique_sbj_labels, inds_reverse_sbj_labels = np.unique(
                sbj_labels_i_pair_pos, return_index=True, return_inverse=True, axis=0)
            for j in range(inds_unique_sbj_labels.shape[0]):
                # get min pos
                inds_j = np.where(inds_reverse_sbj_labels == j)[0]
                prd_pos_probs_i_pos_j = prd_pos_probs_i_pair_pos[inds_j]
                min_prd_pos_probs_i_pos_j = torch.min(prd_pos_probs_i_pos_j)
                # get max neg
                neg_j_inds = np.where(sbj_labels_i_pair_neg == unique_sbj_labels[j])[0]
                if neg_j_inds.size == 0:
                    if cfg.MODEL.USE_SPO_AGNOSTIC_COMPENSATION:
                        pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                        pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))
                    continue
                prd_pos_probs_i_neg_j = prd_pos_probs_i_pair_neg[neg_j_inds]
                max_prd_pos_probs_i_neg_j = torch.max(prd_pos_probs_i_neg_j)
                pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_neg_j.unsqueeze(0)))

    target = torch.ones_like(pair_pos_batch).cuda(device_id)

    return pair_pos_batch, pair_neg_batch, target


def split_pos_neg_p_aware(prd_probs, prd_bias_probs, binary_labels_pos, inds_unique_pos, inds_reverse_pos, prd_labels_pos):
    device_id = prd_probs.get_device()
    if cfg.MODEL.MULTI_RELATION:
        prd_pos_probs = 1 - torch.prod(1 - prd_probs, dim=1)
    else:
        prd_pos_probs = 1 - prd_probs[:, 0]  # shape is (#rels,)
    prd_labels_det = prd_probs[:, 1:].argmax(dim=1).data.cpu().numpy() + 1  # prd_probs is a torch.tensor, exlucding background
    # loop over each group
    pair_pos_batch = torch.ones(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    pair_neg_batch = torch.zeros(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    for i in range(inds_unique_pos.shape[0]):
        inds = np.where(inds_reverse_pos == i)[0]
        prd_pos_probs_i = prd_pos_probs[inds]
        prd_labels_pos_i = prd_labels_pos[inds]
        prd_labels_det_i = prd_labels_det[inds]
        binary_labels_pos_i = binary_labels_pos[inds]
        pair_pos_inds = np.where(binary_labels_pos_i > 0)[0]
        pair_neg_inds = np.where(binary_labels_pos_i == 0)[0]
        if pair_pos_inds.size == 0 or pair_neg_inds.size == 0:  # ignore this node if either pos or neg does not exist
            continue
        prd_pos_probs_i_pair_pos = prd_pos_probs_i[pair_pos_inds]
        prd_pos_probs_i_pair_neg = prd_pos_probs_i[pair_neg_inds]
        prd_labels_i_pair_pos = prd_labels_pos_i[pair_pos_inds]
        prd_labels_i_pair_neg = prd_labels_det_i[pair_neg_inds]
        max_prd_pos_probs_i_pair_neg = torch.max(prd_pos_probs_i_pair_neg)  # this is fixed for a given i
        unique_prd_labels, inds_unique_prd_labels, inds_reverse_prd_labels = np.unique(
            prd_labels_i_pair_pos, return_index=True, return_inverse=True, axis=0)
        for j in range(inds_unique_prd_labels.shape[0]):
            # get min pos
            inds_j = np.where(inds_reverse_prd_labels == j)[0]
            prd_pos_probs_i_pos_j = prd_pos_probs_i_pair_pos[inds_j]
            min_prd_pos_probs_i_pos_j = torch.min(prd_pos_probs_i_pos_j)
            # get max neg
            neg_j_inds = np.where(prd_labels_i_pair_neg == unique_prd_labels[j])[0]
            if neg_j_inds.size == 0:
                if cfg.MODEL.USE_SPO_AGNOSTIC_COMPENSATION:
                    pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                    pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))
                continue
            prd_pos_probs_i_neg_j = prd_pos_probs_i_pair_neg[neg_j_inds]
            max_prd_pos_probs_i_neg_j = torch.max(prd_pos_probs_i_neg_j)
            pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
            pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_neg_j.unsqueeze(0)))

    target = torch.ones_like(pair_pos_batch).cuda(device_id)
        
    return pair_pos_batch, pair_neg_batch, target
