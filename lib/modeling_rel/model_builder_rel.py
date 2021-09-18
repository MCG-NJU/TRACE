# Adapted from Detectron.pytorch/lib/modeling/model_builder.py
# for this project by Ji Zhang, 2019

from functools import wraps
import importlib
import logging
import numpy as np
import copy
import json
import math
import os
import pickle

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_layers import ROIPool, ROIAlign
# from model.roi_pooling.functions.roi_pool import RoIPoolFunction
# from model.roi_crop.functions.roi_crop import RoICropFunction
# from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling_rel.fast_rcnn_heads as fast_rcnn_heads
import modeling_rel.relpn_heads as relpn_heads
import modeling_rel.reldn_heads as reldn_heads
import modeling_rel.rel_pyramid_module as rel_pyramid_module

from modeling_rel.sparse_targets_rel import FrequencyBias
from modeling_rel.get_glob_rel_mat import get_glob_rel_mat

import utils_rel.boxes_rel as box_utils_rel
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils_rel.net_rel as net_utils_rel
from utils.timer import Timer
import utils.resnet_weights_helper as resnet_utils
import utils.fpn as fpn_utils

from combine_tree.word_vectors import obj_edge_vectors
from combine_tree.cluster_tree3d import cluster_tree3d
import combine_tree.utils_t as utils_t



import mmcv
import i3d.i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb as i3d_cfg
from i3d.resnet3d import ResNet3d as resnet3d
from i3d.box_head3d import Tail_Layer_3D as box_head3d

from deep_sort.tracking_by_deep_sort import track_frames_from_featmap


import modeling_rel.focalloss as focalloss





logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        # these two keywords means we need to use the functions from the modeling_rel directory
        if func_name.find('VGG') >= 0 or func_name.find('roi_2mlp_head') >= 0:
            dir_name = 'modeling_rel.'
        else:
            dir_name = 'modeling.'
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = dir_name + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()
        #print(cfg.MODEL.CONV_BODY)    #VGG16.VGG16_conv_body

        # Region Proposal Network
        if cfg.RPN.RPN_ON:
            self.RPN = rpn_heads.generic_rpn_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]
            # [0.03125, 0.0625, 0.125, 0.25]: [1/32, 1/16, 1/8, 1/4]
        
        #print(cfg.FAST_RCNN.ROI_BOX_HEAD)    #VGG16.VGG16_roi_conv5_head
        
        print('self.Conv_Body.spatial_scale')
        print(self.Conv_Body.spatial_scale)
        print('self.RPN.dim_out')
        print(self.RPN.dim_out)
        
        # BBOX Branch
        self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
            self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
        self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
            self.Box_Head.dim_out)
        
        # RelPN
        self.RelPN = relpn_heads.generic_relpn_outputs()
        
        
        
        if cfg.MODEL.MULTI_RELATION:
            n_relation = cfg.MODEL.NUM_PRD_CLASSES
        else:
            if cfg.MODEL.USE_BG:
                n_relation = cfg.MODEL.NUM_PRD_CLASSES + 1
            else:
                n_relation = cfg.MODEL.NUM_PRD_CLASSES
                
        
        if 'frozen_stages' in i3d_cfg.model['backbone'] and (not cfg.FREEZE_PART_I3D):
            i3d_cfg.model['backbone']['frozen_stages'] = 0
        elif 'frozen_stages' in i3d_cfg.model['backbone']:
            i3d_cfg.model['backbone']['frozen_stages'] = 2
        
        if 'num_stages' in i3d_cfg.model['backbone'] and cfg.I3D_DC5:
            i3d_cfg.model['backbone']['num_stages'] = 4
            i3d_cfg.model['backbone']['dilations']=(1, 1, 1, 2)
            if i3d_cfg.model['backbone']['dilations'][-1] == 2:
                i3d_cfg.model['backbone']['spatial_strides']=(1, 2, 2, 1)
            else:
                i3d_cfg.model['backbone']['spatial_strides']=(1, 2, 2, 2)
            
        elif 'num_stages' in i3d_cfg.model['backbone']:
            i3d_cfg.model['backbone']['num_stages'] = 3
        i3d_st = i3d_cfg.model['backbone']['num_stages']
        print('{} frozen_stages in I3D'.format(i3d_cfg.model['backbone']['frozen_stages']))
        self.Prd_RCNN = resnet3d(**i3d_cfg.model['backbone'])
        
        
        
        C = self.Prd_RCNN.feat_dim
        self.Prd_RCNN2D = None
        if cfg.MODEL.REL_CONV_BODY != '':
            self.Prd_RCNN2D = get_func(cfg.MODEL.REL_CONV_BODY)()
        
        low_feat_vec_dim = 256
        self.low_feat_vec_dim = low_feat_vec_dim
        group_coeffi = cfg.MSP_GROUP #4
        max_tree_height = cfg.MSP_MAX_TREE_HEIGHT #7
        pool_size = 7
        self.pool_size = pool_size
        obj_embed_dim = 200 #300
        cluster_num_partition = cfg.CLUSTER_NUM_PARTITION #2
        
        feat_vec_dim = self.Box_Head.dim_out
        feat_combine = self.Box_Head.dim_out // cfg.RESIZE_HIDDEN_DIM
        
        feat_region_vec_dim = self.Box_Head.dim_out
        hidden_dim = self.Box_Head.dim_out // cfg.RESIZE_HIDDEN_DIM
        
        print('Box_Head.dim_out:')
        print(self.Box_Head.dim_out)
        
        if cfg.USE_BASELINE.FULL_GRAPH_MSP:
            msp_mode = 'fully_connected_graph'
        else:
            msp_mode = 'tree-gru'
        print('msp mode: {}'.format(msp_mode))
        print('cluster_num_partition: {}'.format(cfg.CLUSTER_NUM_PARTITION))
        print('max_tree_height: {}'.format(cfg.MSP_MAX_TREE_HEIGHT))
        print('group_coeffi: {}'.format(cfg.MSP_GROUP))    
        
        if cfg.FPN.REL_FPN_ON:
            spatial_scale = [0.03125, 0.0625, 0.125, 0.25]
        elif self.Prd_RCNN2D is not None:
            spatial_scale = self.Prd_RCNN2D.spatial_scale
        else:
            spatial_scale = 1.0/16
        self.rel_spatial_scale = spatial_scale
        
        print('spatial_scale')
        print(spatial_scale)
        
        
        rel_box_head = box_head3d(
            self.Prd_RCNN, 
            self.Prd_RCNN.res_layers,
            self.roi_feature_transform, 
            spatial_scale, 
            num_stages = i3d_cfg.model['backbone']['num_stages'])
        
        self.cluster_tree3d = cluster_tree3d((C, 0, 0), 
                                            feat_combine, 
                                            feat_vec_dim, 
                                            feat_region_vec_dim, 
                                            hidden_dim, 
                                            low_feat_vec_dim, 
                                            ROIAlign, 
                                            spatial_scale, 
                                            group_coeffi=group_coeffi, 
                                            max_tree_height=max_tree_height, 
                                            pool_size=pool_size, 
                                            BoxHead=rel_box_head, 
                                            cluster_num_partition=cluster_num_partition, 
                                            msp_mode=msp_mode)
                                            
    
        self.obj_embed_dim = obj_embed_dim
        obj_cats = json.load(open(cfg.MODEL.OBJ_CLASS_FILE_NAME))
        self.obj_class = tuple(['__background__'] + obj_cats)
        embed_vecs = obj_edge_vectors(self.obj_class, cfg.MODEL.OBJ_EMBED_FILE_NAME, wv_dim=self.obj_embed_dim)
        n_obj_class = len(self.obj_class)
        self.embed_obj_det = nn.Linear(n_obj_class, self.obj_embed_dim, bias=False)
        self.embed_obj_det.weight.data = embed_vecs.clone().t()
        self.embed_s = nn.Linear(n_obj_class, self.obj_embed_dim, bias=False)
        self.embed_s.weight.data = embed_vecs.clone().t()
        self.embed_o = nn.Linear(n_obj_class, self.obj_embed_dim, bias=False)
        self.embed_o.weight.data = embed_vecs.clone().t()
    
        self.after_obj_det_embed = nn.Sequential(
                                nn.LeakyReLU(0.1),
                                nn.Linear(self.obj_embed_dim, self.Box_Head.dim_out))
                                            
        self.obj_tracking = track_frames_from_featmap

    
        self.restore_obj_feat = nn.Sequential(
                            nn.LeakyReLU(0.1),
                            nn.Linear(feat_combine, self.Box_Head.dim_out, bias=True))
    
    
        infer_dim_out = self.Box_Head.dim_out // cfg.RESIZE_HIDDEN_DIM

    
        self.spatio_atten_s = nn.Sequential(
                            nn.LeakyReLU(0.1),
                            nn.Linear(self.Box_Head.dim_out, low_feat_vec_dim, bias=True))
        self.spatio_atten_o = nn.Sequential(
                            nn.LeakyReLU(0.1),
                            nn.Linear(self.Box_Head.dim_out, low_feat_vec_dim, bias=True))
        self.spatio_atten_map = nn.Conv2d(C, low_feat_vec_dim, 1, stride=1, padding=0, bias=True)
        
        self.atten_q_map = nn.Conv2d(C, low_feat_vec_dim, 1, stride=1, padding=0, bias=True)
    
        self.sqrt_low_dim_embed_dim = math.sqrt(low_feat_vec_dim)
    
        self.spatio_inferA = nn.Sequential(
                    nn.Conv2d(low_feat_vec_dim, low_feat_vec_dim, 1, stride=1, padding=0, bias=True),)
        self.spatio_inferB = nn.Sequential(
                    nn.Conv2d(low_feat_vec_dim, low_feat_vec_dim, 1, stride=1, padding=0, bias=True),)
        self.spatio_infer_fc = nn.Sequential(
                    nn.Linear(low_feat_vec_dim*pool_size*pool_size, infer_dim_out, bias=True),)
                            
        self.sem_s = nn.Sequential(
                    nn.Linear(self.Box_Head.dim_out, n_relation, bias=True))
        self.sem_s_e = nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Linear(self.obj_embed_dim, infer_dim_out, bias=True))
        self.sem_o = nn.Sequential(
                    nn.Linear(self.Box_Head.dim_out, n_relation, bias=True))
        self.sem_o_e = nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Linear(self.obj_embed_dim, infer_dim_out, bias=True))
        self.sem_r = nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Linear(feat_combine, infer_dim_out, bias=True),)
        
        
        if cfg.MODEL.USE_SPATIAL_FEAT:
            self.spt_cls_feats = nn.Sequential(
                nn.Linear(28, 64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(64, 64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(64, n_relation))
    
    
        self.relation_infer_sem = nn.Sequential(
                            nn.Dropout(p=0.5),
                            nn.LeakyReLU(0.1, inplace=True),
                            nn.Linear(infer_dim_out, n_relation, bias=True),)
        self.relation_infer_spa = nn.Sequential(
                            nn.Dropout(p=0.5),
                            nn.LeakyReLU(0.1, inplace=True),
                            nn.Linear(infer_dim_out, n_relation, bias=True),)
                            
        self.sigmoid = nn.Sigmoid()
        
        if cfg.MODEL.USE_FREQ_BIAS or cfg.MODEL.RUN_BASELINE:
            # Assume we are training/testing on only one dataset
            if len(cfg.TRAIN.DATASETS):
                self.freq_bias = FrequencyBias(cfg.TRAIN.DATASETS[0])
            else:
                #assert False, 'No training set for FrequencyBias!'
                self.freq_bias = FrequencyBias(cfg.TEST.DATASETS[0])
        
        self.reset()
        
        self._init_modules()
        
        
        # initialize S/O branches AFTER init_weigths so that weights can be automatically copied
        self.S_Head = copy.deepcopy(self.Box_Head)
        self.O_Head = copy.deepcopy(self.S_Head)
        for p in self.S_Head.parameters():
            p.requires_grad = True
        for p in self.O_Head.parameters():
            p.requires_grad = True
            
        
        self.pos_weight = None
        
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        
    def reset(self):
        self.after_obj_det_embed.apply(utils_t.weight_init_mynn_Xavier)
        
        self.restore_obj_feat.apply(utils_t.weight_init_mynn_Xavier)
        
        self.spatio_atten_s.apply(utils_t.weight_init_mynn_Xavier)
        self.spatio_atten_o.apply(utils_t.weight_init_mynn_Xavier)
        
        utils_t.weight_init_mynn_Xavier(self.spatio_atten_map)
        
        utils_t.weight_init_mynn_Xavier(self.atten_q_map)
        
        self.spatio_inferA.apply(utils_t.weight_init_mynn_conv_MSRAFill)
        self.spatio_inferB.apply(utils_t.weight_init_mynn_conv_MSRAFill)
        
        self.spatio_infer_fc.apply(utils_t.weight_init_mynn_Xavier)
        
        self.sem_s.apply(utils_t.weight_init_mynn_Xavier)
        
        self.sem_s_e.apply(utils_t.weight_init_mynn_Xavier)
        
        self.sem_o.apply(utils_t.weight_init_mynn_Xavier)
        
        self.sem_o_e.apply(utils_t.weight_init_mynn_Xavier)
        
        self.sem_r.apply(utils_t.weight_init_mynn_Xavier)


        self.relation_infer_sem.apply(utils_t.weight_init_mynn_Xavier)
        self.relation_infer_spa.apply(utils_t.weight_init_mynn_Xavier)
        
        
    
    def _init_modules(self):
        # VGG16 imagenet pretrained model is initialized in VGG16.py
        if cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS != '':
            logger.info("Loading pretrained weights from %s", cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS)
            resnet_utils.load_pretrained_imagenet_weights(self)
            for p in self.Conv_Body.parameters():
                p.requires_grad = False
                
        if cfg.RESNETS.VRD_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.RESNETS.VRD_PRETRAINED_WEIGHTS)
        if cfg.VGG16.VRD_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.VGG16.VRD_PRETRAINED_WEIGHTS)
            
        if cfg.RESNETS.VG_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.RESNETS.VG_PRETRAINED_WEIGHTS)
        if cfg.VGG16.VG_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.VGG16.VG_PRETRAINED_WEIGHTS)
            
        if cfg.RESNETS.OI_REL_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.RESNETS.OI_REL_PRETRAINED_WEIGHTS)
        if cfg.VGG16.OI_REL_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.VGG16.OI_REL_PRETRAINED_WEIGHTS)
        
        
        self.cem_Box_Outs = copy.deepcopy(self.Box_Outs)
        del self.cem_Box_Outs.bbox_pred
        for p in self.cem_Box_Outs.parameters():
            p.requires_grad = True
        
        
        if cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS != '' or cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS != '' or \
            cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS != '' or cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS != '' or \
            cfg.RESNETS.OI_REL_PRD_PRETRAINED_WEIGHTS != '' or cfg.VGG16.OI_REL_PRD_PRETRAINED_WEIGHTS != '':
            if cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS)
                if cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS.split('.')[-1] == 'pkl':
                    with open(cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS, 'rb') as f:
                        checkpoint = pickle.load(f)
                        f.close()
                else:
                    checkpoint = torch.load(cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            if cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS)
                if cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS.split('.')[-1] == 'pkl':
                    with open(cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS, 'rb') as f:
                        checkpoint = pickle.load(f)
                        f.close()
                else:
                    checkpoint = torch.load(cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            if cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS)
                if cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS.split('.')[-1] == 'pkl':
                    with open(cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS, 'rb') as f:
                        checkpoint = pickle.load(f)
                        f.close()
                else:
                    checkpoint = torch.load(cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            if cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS)
                if cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS.split('.')[-1] == 'pkl':
                    with open(cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS, 'rb') as f:
                        checkpoint = pickle.load(f)
                        f.close()
                else:
                    checkpoint = torch.load(cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            if cfg.RESNETS.OI_REL_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.RESNETS.OI_REL_PRD_PRETRAINED_WEIGHTS)
                if cfg.RESNETS.OI_REL_PRD_PRETRAINED_WEIGHTS.split('.')[-1] == 'pkl':
                    with open(cfg.RESNETS.OI_REL_PRD_PRETRAINED_WEIGHTS, 'rb') as f:
                        checkpoint = pickle.load(f)
                        f.close()
                else:
                    checkpoint = torch.load(cfg.RESNETS.OI_REL_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            if cfg.VGG16.OI_REL_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.VGG16.OI_REL_PRD_PRETRAINED_WEIGHTS)
                if cfg.VGG16.OI_REL_PRD_PRETRAINED_WEIGHTS.split('.')[-1] == 'pkl':
                    with open(cfg.VGG16.OI_REL_PRD_PRETRAINED_WEIGHTS, 'rb') as f:
                        checkpoint = pickle.load(f)
                        f.close()
                else:
                    checkpoint = torch.load(cfg.VGG16.OI_REL_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            
            # not using the last softmax layers
            if 'model' in checkpoint:
                if 'roi_heads.box_predictor.cls_score.weight' not in checkpoint['model']:
                    del checkpoint['model']['Box_Outs.cls_score.weight']
                    del checkpoint['model']['Box_Outs.cls_score.bias']
                else:
                    del checkpoint['model']['roi_heads.box_predictor.cls_score.weight']
                    del checkpoint['model']['roi_heads.box_predictor.cls_score.bias']
                    
                if 'roi_heads.box_predictor.bbox_pred.weight' not in checkpoint['model']:
                    del checkpoint['model']['Box_Outs.bbox_pred.weight']
                    del checkpoint['model']['Box_Outs.bbox_pred.bias']
                else:
                    del checkpoint['model']['roi_heads.box_predictor.bbox_pred.weight']
                    del checkpoint['model']['roi_heads.box_predictor.bbox_pred.bias']
            
            if 'model' in checkpoint:
                ckpt = checkpoint['model']
            elif 'state_dict' in checkpoint:
                ckpt = checkpoint['state_dict']
            else:
                ckpt = checkpoint
            ckpt_prd = net_utils_rel.trans_prd(ckpt, find_s='Prd_RCNN.')
            net_utils_rel.load_ckpt_rel(self.Prd_RCNN, ckpt_prd)
                
            if cfg.TRAIN.FREEZE_PRD_CONV_BODY:
                for p in self.Prd_RCNN.Conv_Body.parameters():
                    p.requires_grad = False
            if cfg.TRAIN.FREEZE_PRD_BOX_HEAD:
                for p in self.Prd_RCNN.Box_Head.parameters():
                    p.requires_grad = False
            
            
            if cfg.MODEL.REL_CONV_BODY != '' and cfg.RESNETS.PRD_2D_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.RESNETS.PRD_2D_PRETRAINED_WEIGHTS)
                if cfg.RESNETS.PRD_2D_PRETRAINED_WEIGHTS.split('.')[-1] == 'pkl':
                    with open(cfg.RESNETS.PRD_2D_PRETRAINED_WEIGHTS, 'rb') as f:
                        checkpoint = pickle.load(f)
                        f.close()
                else:
                    checkpoint = torch.load(cfg.RESNETS.PRD_2D_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
                if 'model' in checkpoint:
                    ckpt = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    ckpt = checkpoint['state_dict']
                else:
                    ckpt = checkpoint
                ckpt_prd = net_utils_rel.trans_prd(ckpt, find_s='Prd_RCNN.')
                net_utils_rel.load_ckpt_rel(self.Prd_RCNN2D, ckpt_prd)
                for p in self.Prd_RCNN2D.parameters(): 
                    p.requires_grad=True
                
        if cfg.RESNETS.TO_BE_FINETUNED_WEIGHTS != '' or cfg.VGG16.TO_BE_FINETUNED_WEIGHTS != '':
            is_weight_exist = False
            if cfg.RESNETS.TO_BE_FINETUNED_WEIGHTS != '':
                logger.info("loading trained and to be finetuned weights from %s", cfg.RESNETS.TO_BE_FINETUNED_WEIGHTS)
                if os.path.exists(cfg.RESNETS.TO_BE_FINETUNED_WEIGHTS):
                    is_weight_exist = True
                    checkpoint = torch.load(cfg.RESNETS.TO_BE_FINETUNED_WEIGHTS, map_location=lambda storage, loc: storage)
                    logger.info("loaded to be finetuned weights.")
                else:
                    logger.info("Not load to be finetuned weights!!!")
            if cfg.VGG16.TO_BE_FINETUNED_WEIGHTS != '':
                logger.info("loading trained and to be finetuned weights from %s", cfg.VGG16.TO_BE_FINETUNED_WEIGHTS)
                if os.path.exists(cfg.RESNETS.TO_BE_FINETUNED_WEIGHTS):
                    is_weight_exist = True
                    checkpoint = torch.load(cfg.VGG16.TO_BE_FINETUNED_WEIGHTS, map_location=lambda storage, loc: storage)
                    logger.info("loaded to be finetuned weights.")
                else:
                    logger.info("Not load to be finetuned weights!!!")
            if is_weight_exist:
                net_utils_rel.load_ckpt_rel(self, checkpoint['model'])
                for p in self.Conv_Body.parameters():
                    p.requires_grad = False
                for p in self.RPN.parameters():
                    p.requires_grad = False
                if not cfg.MODEL.UNFREEZE_DET:
                    for p in self.Box_Head.parameters():
                        p.requires_grad = False
                    for p in self.Box_Outs.parameters():
                        p.requires_grad = False
                    
        if cfg.RESNETS.REL_PRETRAINED_WEIGHTS != '':
            logger.info("loading rel pretrained weights from %s", cfg.RESNETS.REL_PRETRAINED_WEIGHTS)
            checkpoint = torch.load(cfg.RESNETS.REL_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            prd_rcnn_state_dict = {}
            reldn_state_dict = {}
            for name in checkpoint['model']:
                if name.find('Prd_RCNN') >= 0:
                    prd_rcnn_state_dict[name] = checkpoint['model'][name]
                if name.find('RelDN') >= 0:
                    reldn_state_dict[name] = checkpoint['model'][name]
            net_utils_rel.load_ckpt_rel(self.Prd_RCNN, prd_rcnn_state_dict)
            if cfg.TRAIN.FREEZE_PRD_CONV_BODY:
                for p in self.Prd_RCNN.Conv_Body.parameters():
                    p.requires_grad = False
            if cfg.TRAIN.FREEZE_PRD_BOX_HEAD:
                for p in self.Prd_RCNN.Box_Head.parameters():
                    p.requires_grad = False
            del reldn_state_dict['RelDN.prd_cls_scores.weight']
            del reldn_state_dict['RelDN.prd_cls_scores.bias']
            if 'RelDN.prd_sbj_scores.weight' in reldn_state_dict:
                del reldn_state_dict['RelDN.prd_sbj_scores.weight']
            if 'RelDN.prd_sbj_scores.bias' in reldn_state_dict:
                del reldn_state_dict['RelDN.prd_sbj_scores.bias']
            if 'RelDN.prd_obj_scores.weight' in reldn_state_dict:
                del reldn_state_dict['RelDN.prd_obj_scores.weight']
            if 'RelDN.prd_obj_scores.bias' in reldn_state_dict:
                del reldn_state_dict['RelDN.prd_obj_scores.bias']
            if 'RelDN.spt_cls_scores.weight' in reldn_state_dict:
                del reldn_state_dict['RelDN.spt_cls_scores.weight']
            if 'RelDN.spt_cls_scores.bias' in reldn_state_dict:
                del reldn_state_dict['RelDN.spt_cls_scores.bias']
            net_utils_rel.load_ckpt_rel(self.RelDN, reldn_state_dict)
            
        # By Ji on 05/11/2019
        if cfg.RESNETS.REL_RCNN_PRETRAINED_WEIGHTS != '':
            logger.info("loading rel_rcnn pretrained weights from %s", cfg.RESNETS.REL_RCNN_PRETRAINED_WEIGHTS)
            checkpoint = torch.load(cfg.RESNETS.REL_RCNN_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            to_be_deleted = []
            for p, _ in checkpoint['model'].items():
                if p.find('Prd_RCNN') < 0 or p.find('Box_Outs') >= 0:
                    to_be_deleted.append(p)
            for p in to_be_deleted:
                del checkpoint['model'][p]
            net_utils_rel.load_ckpt_rel(self.Prd_RCNN, checkpoint['model'])
        
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1 or classname.find('bn') != -1:
              m.eval()
        
        if cfg.TRAIN.EVA_BATCHNORM:
            self.Conv_Body.apply(set_bn_eval)
        if cfg.FREEZE_ALL_I3D:
            print('Freeze all i3d.')
            for p in self.Prd_RCNN.parameters(): 
                p.requires_grad=False

        
    def load_detector_weights(self, weight_name):
        logger.info("loading pretrained weights from %s", weight_name)
        if weight_name.split('.')[-1] == 'pkl':
            with open(weight_name, 'rb') as f:
                checkpoint = pickle.load(f)
                f.close()
        else:    
            checkpoint = torch.load(weight_name, map_location=lambda storage, loc: storage)
        net_utils_rel.load_ckpt_rel(self, checkpoint['model'])
        
        # freeze everything above the rel module
        for p in self.Conv_Body.parameters():
            p.requires_grad = False
            
        if not cfg.FPN.FREEZE_FPN:
            for p in self.Conv_Body.conv_top.parameters():
                p.requires_grad = True
            for p in self.Conv_Body.topdown_lateral_modules.parameters():
                p.requires_grad = True
            for p in self.Conv_Body.posthoc_modules.parameters():
                p.requires_grad = True
                
        if cfg.RPN.FREEZE_RPN:
            for p in self.RPN.parameters():
                p.requires_grad = False
        if not cfg.MODEL.UNFREEZE_DET:
            for p in self.Box_Head.parameters():
                p.requires_grad = False
            for p in self.Box_Outs.parameters():
                p.requires_grad = False

    def forward(self, data, all_frames, im_info, pre_processed_frames_rpn_ret=None, pre_processed_temporal_roi=None, 
                do_vis=False, dataset_name=None, file_name=None, roidb=None, use_gt_labels=False, bf_cur_len=None, 
                f_scale=None, **rpn_kwargs):
        
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, all_frames, im_info, \
                    pre_processed_frames_rpn_ret, pre_processed_temporal_roi, \
                    do_vis, dataset_name, file_name, roidb, use_gt_labels, \
                    bf_cur_len, f_scale, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, all_frames, im_info, \
                    pre_processed_frames_rpn_ret, pre_processed_temporal_roi, \
                    do_vis, dataset_name, file_name, roidb, use_gt_labels, \
                    bf_cur_len, f_scale, **rpn_kwargs)

    def _forward(self, data, all_frames, im_info, pre_processed_frames_rpn_ret=None, pre_processed_temporal_roi=None, 
                do_vis=False, dataset_name=None, file_name=None, roidb=None, use_gt_labels=False, bf_cur_len=None, 
                f_scale=None, **rpn_kwargs):
        
        im_data = data
        device_id = im_data.get_device()
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))
        if dataset_name is not None:
            dataset_name = blob_utils.deserialize(dataset_name)
        else:
            dataset_name = cfg.TRAIN.DATASETS[0] if self.training else cfg.TEST.DATASETS[0]  # assuming only one dataset per run
        
        if cfg.TEST.GET_FRAME_ROIS and roidb is not None:
            file_name = int(roidb[0]['file_name'].split('.')[0])
        
        if self.training and cfg.ENABLE_FRAME_PRE_PROCESSING:
            if roidb[0]['pre_processed_frames_rpn_ret'] is not None:
                pre_processed_frames_rpn_ret = roidb[0]['pre_processed_frames_rpn_ret']
            
            if roidb[0]['pre_processed_temporal_roi'] is not None:
                pre_processed_temporal_roi = roidb[0]['pre_processed_temporal_roi']
        
        
        temporal_imgs = all_frames # (N, T, C, H, W)
        if isinstance(bf_cur_len, torch.Tensor):
            bf_cur_len = bf_cur_len.item()
            
        if isinstance(f_scale, torch.Tensor):
            f_scale = f_scale.item()

        
        img_list_f = [i for i in range(0, bf_cur_len+1)]
        img_list_b = [i for i in range(bf_cur_len, 2*cfg.HALF_NUMBER_OF_FRAMES+1)]
        batch_size = temporal_imgs.shape[0]
        
        return_dict = {}  # A dict to collect return variables

        blob_conv = self.Conv_Body(im_data)
        
        if (cfg.ENABLE_FRAME_PRE_PROCESSING and pre_processed_frames_rpn_ret is not None):
            if dataset_name.find('vidvrd') >= 0:
                #pre_processed_boxes = 0
                pre_processed_boxes = pre_processed_frames_rpn_ret['rois'].copy()
                #print(pre_processed_boxes)
                #assert False
            else:
                pre_processed_boxes = None
            if self.training:
                rpn_ret = self.RPN(blob_conv, im_info, roidb)
                if dataset_name.find('vid') >= 0:
                    pre_processed_frames_rpn_ret['rois'][:, 1:] *= float(im_info[0, -1].item())
                pre_processed_fg_inds = \
                    np.where(pre_processed_frames_rpn_ret['labels_int32'][0] >= 0)[0]
                rpn_ret['rois'] = np.vstack((rpn_ret['rois'], \
                    pre_processed_frames_rpn_ret['rois'][pre_processed_fg_inds]))
                rpn_ret['labels_int32'] = np.concatenate((rpn_ret['labels_int32'], \
                    pre_processed_frames_rpn_ret['labels_int32'][0][pre_processed_fg_inds]), 0)
            else:
                if dataset_name.find('vid') >= 0:
                    pre_processed_frames_rpn_ret['rois'][:, 1:] *= float(im_info[0, -1].item())
                rpn_ret = {'rois': pre_processed_frames_rpn_ret['rois'], 
                        'labels_int32': pre_processed_frames_rpn_ret['labels_int32'][0]}
                if 'prd_category' in pre_processed_frames_rpn_ret:
                    rpn_ret['prd_category'] = pre_processed_frames_rpn_ret['prd_category']
                    rpn_ret['score'] = pre_processed_frames_rpn_ret['score']
            if cfg.FPN.FPN_ON:
                self._add_rel_multilevel_rois(rpn_ret)
        else:
            rpn_ret = self.RPN(blob_conv, im_info, roidb)

        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]

        cls_score = None
        if not self.training:
            box_feat = self.Box_Head(blob_conv, rpn_ret, use_relu=True)
            cls_score, bbox_pred = self.Box_Outs(box_feat)
        
        # now go through the predicate branch
        use_relu = False if cfg.MODEL.NO_FC7_RELU else True
        if self.training:
            fg_inds = np.where(rpn_ret['labels_int32'] > 0)[0]
            det_rois = rpn_ret['rois'][fg_inds]
            det_labels = rpn_ret['labels_int32'][fg_inds]
            det_scores = None
            
            rel_ret = self.RelPN(det_rois, det_labels, det_scores, im_info, dataset_name, roidb)
            
            ori_det_feat_vec = self.Box_Head(blob_conv, rel_ret, rois_name='det_rois', use_relu=False)
            det_cls_score = self.Box_Outs.cls_score(self.relu(ori_det_feat_vec))
            det_cls_soft = F.softmax(det_cls_score, dim=1)
            if cfg.DISABLE_EMBED:
                det_feat_vec = ori_det_feat_vec
            else:
                det_embed = self.embed_obj_det(det_cls_soft)
                det_feat_vec = ori_det_feat_vec + self.after_obj_det_embed(det_embed)
            
            sbj_feat = self.S_Head(blob_conv, rel_ret, rois_name='sbj_rois', use_relu=use_relu)
            obj_feat = self.O_Head(blob_conv, rel_ret, rois_name='obj_rois', use_relu=use_relu)
        else:
            bad_gt = False
            if roidb is not None:
                all_boxes = np.concatenate((roidb['sbj_gt_boxes'], roidb['obj_gt_boxes']))
                if len(all_boxes) <= 0:
                    bad_gt = True
            if roidb is not None and (not bad_gt):
                im_scale = im_info.data.numpy()[:, 2][0]
                im_w = im_info.data.numpy()[:, 1][0]
                im_h = im_info.data.numpy()[:, 0][0]
                sbj_boxes = roidb['sbj_gt_boxes'].copy()
                obj_boxes = roidb['obj_gt_boxes'].copy()
                
                
                pair_gt_boxes = np.concatenate((sbj_boxes, obj_boxes), axis=1).astype(np.int32)
                pair_gt_boxes, pair_gt_index = np.unique(pair_gt_boxes, return_index=True, axis=0)
                sbj_boxes = pair_gt_boxes[:, :4].astype(np.float32)
                obj_boxes = pair_gt_boxes[:, 4:].astype(np.float32)
                
                
                
                all_boxes = np.concatenate((sbj_boxes, obj_boxes)).astype(np.int32)
                unique_all_boxes, gt_boxes_id = np.unique(all_boxes, return_inverse=True, axis=0)
                sbj_gt_boxes_id = gt_boxes_id[:sbj_boxes.shape[0]]
                obj_gt_boxes_id = gt_boxes_id[sbj_boxes.shape[0]:]
                unique_all_boxes = unique_all_boxes.astype(np.float32)
                
                
                sbj_rois = sbj_boxes * im_scale
                obj_rois = obj_boxes * im_scale
                all_rois = unique_all_boxes * im_scale
                repeated_batch_idx = 0 * blob_utils.ones((sbj_rois.shape[0], 1))
                det_rois_repeated_batch_idx = 0 * blob_utils.ones((all_rois.shape[0], 1))
                sbj_rois = np.hstack((repeated_batch_idx, sbj_rois))
                obj_rois = np.hstack((repeated_batch_idx, obj_rois))
                all_rois = np.hstack((det_rois_repeated_batch_idx, all_rois))
                rel_rois = box_utils_rel.rois_union(sbj_rois, obj_rois)
                rel_ret = {}
                rel_ret['sbj_rois'] = sbj_rois
                rel_ret['obj_rois'] = obj_rois
                rel_ret['rel_rois'] = rel_rois
                rel_ret['det_rois'] = all_rois
                
                rel_ret['sbj_inds'] = sbj_gt_boxes_id
                rel_ret['obj_inds'] = obj_gt_boxes_id
                
                
                if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
                    lvl_min = cfg.FPN.ROI_MIN_LEVEL
                    lvl_max = cfg.FPN.ROI_MAX_LEVEL
                    rois_blob_names = ['sbj_rois', 'obj_rois', 'rel_rois', 'det_rois']
                    for rois_blob_name in rois_blob_names:
                        # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
                        target_lvls = fpn_utils.map_rois_to_fpn_levels(
                            rel_ret[rois_blob_name][:, 1:5], lvl_min, lvl_max)
                        fpn_utils.add_multilevel_roi_blobs(
                            rel_ret, rois_blob_name, rel_ret[rois_blob_name], target_lvls,
                            lvl_min, lvl_max)
                sbj_det_feat = self.Box_Head(blob_conv, rel_ret, rois_name='sbj_rois', use_relu=True)
                sbj_cls_scores, _ = self.Box_Outs(sbj_det_feat)
                sbj_cls_scores = sbj_cls_scores.data.cpu().numpy()
                obj_det_feat = self.Box_Head(blob_conv, rel_ret, rois_name='obj_rois', use_relu=True)
                obj_cls_scores, _ = self.Box_Outs(obj_det_feat)
                obj_cls_scores = obj_cls_scores.data.cpu().numpy()
                
                
                ori_det_feat_vec = self.Box_Head(blob_conv, rel_ret, rois_name='det_rois', use_relu=False)
                det_cls_score, det_bbox_pred = self.Box_Outs(self.relu(ori_det_feat_vec))
                det_cls_scores_np = det_cls_score.data.cpu().numpy()
                det_cls_soft = det_cls_score
                
                if cfg.DISABLE_EMBED:
                    det_feat_vec = ori_det_feat_vec
                else:
                    det_embed = self.embed_obj_det(det_cls_soft)
                    det_feat_vec = ori_det_feat_vec + self.after_obj_det_embed(det_embed)

                
                if use_gt_labels:
                    sbj_labels = roidb['sbj_gt_classes']  # start from 0
                    obj_labels = roidb['obj_gt_classes']  # start from 0
                    
                    sbj_labels = sbj_labels[pair_gt_index]
                    obj_labels = obj_labels[pair_gt_index]
                    
                    sbj_scores = np.ones_like(sbj_labels, dtype=np.float32)
                    obj_scores = np.ones_like(obj_labels, dtype=np.float32)
                    
                    
                    all_labels = np.concatenate((sbj_labels, obj_labels))
                    _, gt_labels_id = np.unique(gt_boxes_id, return_index=True)
                    det_labels = all_labels[gt_labels_id]
                    det_scores = np.ones_like(det_labels, dtype=np.float32)
                    
                else:
                    sbj_labels = np.argmax(sbj_cls_scores[:, 1:], axis=1)
                    obj_labels = np.argmax(obj_cls_scores[:, 1:], axis=1)
                    sbj_scores = np.amax(sbj_cls_scores[:, 1:], axis=1)
                    obj_scores = np.amax(obj_cls_scores[:, 1:], axis=1)
                    
                    
                    det_labels = np.argmax(det_cls_scores_np[:, 1:], axis=1)
                    det_scores = np.amax(det_cls_scores_np[:, 1:], axis=1)
                    
                    
                rel_ret['sbj_scores'] = sbj_scores.astype(np.float32, copy=False)
                rel_ret['obj_scores'] = obj_scores.astype(np.float32, copy=False)
                rel_ret['sbj_labels'] = sbj_labels.astype(np.int32, copy=False) + 1  # need to start from 1
                rel_ret['obj_labels'] = obj_labels.astype(np.int32, copy=False) + 1  # need to start from 1
                rel_ret['all_sbj_labels_int32'] = sbj_labels.astype(np.int32, copy=False)
                rel_ret['all_obj_labels_int32'] = obj_labels.astype(np.int32, copy=False)
                
                
                rel_ret['det_scores'] = det_scores.astype(np.float32, copy=False)
                rel_ret['det_labels'] = det_labels.astype(np.int32, copy=False) + 1  # need to start from 1
                rel_ret['all_det_labels_int32'] = det_labels.astype(np.int32, copy=False)

                if cfg.MODEL.USE_SPATIAL_FEAT:
                    spt_feat = box_utils_rel.get_spt_features(sbj_boxes, obj_boxes, im_w, im_h)
                    rel_ret['spt_feat'] = spt_feat

                sbj_feat = self.S_Head(blob_conv, rel_ret, rois_name='sbj_rois', use_relu=use_relu)
                obj_feat = self.O_Head(blob_conv, rel_ret, rois_name='obj_rois', use_relu=use_relu)
            else:
                score_thresh = cfg.TEST.SCORE_THRESH
                iter_num = 0
                if cfg.ENABLE_FRAME_PRE_PROCESSING and pre_processed_frames_rpn_ret is not None and \
                  dataset_name.find('vidvrd') >= 0 and (not cfg.USE_PRD_BOXES):
                    det_rois = rpn_ret['rois']
                    if 'prd_category' in rpn_ret:
                        det_scores = rpn_ret['score']
                        det_labels = rpn_ret['prd_category']
                        det_scores = np.amax(det_scores[:, 1:], axis=1)
                    else:
                        if cls_score is not None:
                            det_scores = cls_score
                            det_cls_scores_np = det_scores.data.cpu().numpy()
                            det_labels = np.argmax(det_cls_scores_np[:, 1:], axis=1).astype(np.int32, copy=False) + 1
                            det_scores = np.amax(det_cls_scores_np[:, 1:], axis=1)
                    
                    rel_ret = self.RelPN(det_rois, det_labels, det_scores, 
                                im_info, dataset_name, roidb=None, lim_score=score_thresh)
                else:
                    pre_boxes = pre_processed_boxes if (cfg.ENABLE_FRAME_PRE_PROCESSING and \
                        pre_processed_frames_rpn_ret is not None and dataset_name.find('vidvrd') >= 0) else None
                    while score_thresh >= -1e-05:  # a negative value very close to 0.0
                        det_rois, det_labels, det_scores, _ = \
                            self.prepare_det_rois(rpn_ret['rois'], cls_score, bbox_pred, 
                                im_info, dataset_name, score_thresh, cfg.TRAIN.IMS_PER_BATCH, 
                                pre_boxes=pre_boxes)
                        rel_ret = self.RelPN(det_rois, det_labels, det_scores, 
                                        im_info, dataset_name, roidb=None, lim_score=score_thresh)
                        valid_len = len(rel_ret['rel_rois'])
                        if valid_len > 0:
                            break
                        
                        if score_thresh >= 0.02 - 1e-04:
                            logger.info('Got {} rel_rois when score_thresh={}, changing to {}'.format(
                                valid_len, score_thresh, score_thresh - 0.01))    
                            score_thresh -= 0.01
                        elif iter_num <= 50:
                            logger.info('Got {} rel_rois when score_thresh={}, changing to {}'.format(
                                valid_len, score_thresh, score_thresh / 2.0))
                            score_thresh /= 2.0
                            iter_num += 1
                        else:
                            logger.info('Got {} rel_rois when score_thresh={}, changing to {}'.format(
                                valid_len, score_thresh, -1e-06))
                            score_thresh = -1e-06
                            score_thresh *= 2
                
                
                ori_det_feat_vec = self.Box_Head(blob_conv, rel_ret, rois_name='det_rois', use_relu=False)
                det_cls_score, det_bbox_pred = self.Box_Outs(self.relu(ori_det_feat_vec))
                det_cls_soft = det_cls_score
                
                if cfg.DISABLE_EMBED:
                    det_feat_vec = ori_det_feat_vec
                else:
                    det_embed = self.embed_obj_det(det_cls_soft)
                    det_feat_vec = ori_det_feat_vec + self.after_obj_det_embed(det_embed)

                
                det_s_feat = self.S_Head(blob_conv, rel_ret, rois_name='det_rois', use_relu=use_relu)
                det_o_feat = self.O_Head(blob_conv, rel_ret, rois_name='det_rois', use_relu=use_relu)
                sbj_feat = det_s_feat[rel_ret['sbj_inds']]
                obj_feat = det_o_feat[rel_ret['obj_inds']]
                
        
        
        if cfg.USE_BASELINE.FULL_GRAPH_MSP and self.training:
            X_list_id = rel_ret['sbj_inds'].astype(np.int)[:,-1]
            Y_list_id = rel_ret['obj_inds'].astype(np.int)[:,-1]
        elif cfg.USE_BASELINE.FULL_GRAPH_MSP:
            X_list_id = rel_ret['sbj_inds'].astype(np.int)
            Y_list_id = rel_ret['obj_inds'].astype(np.int)
        else:
            X_list_id = None
            Y_list_id = None
    
        temporal_roi = self.get_tracklet(img_list_f, img_list_b, rel_ret['det_rois'], 
                    post_prdcnn_T=len(img_list_f)+len(img_list_b)-1)
        
        sampled_temporal_imgs = temporal_imgs
    
        roi = rel_ret['det_rois']
        
        roi, temporal_roi, sampled_temporal_imgs, f_resized_scale = \
                self.downsample_for_3d(roi, temporal_roi, 
                    sampled_temporal_imgs, float(1.0/im_info[0, -1].item()), f_scale)
                                          
                                
        sampled_temporal_imgs = sampled_temporal_imgs.permute(0, 2, 1, 3, 4)
        temporal_blob_conv_prd = \
            self.Prd_RCNN(sampled_temporal_imgs) # n, T, C, P, P -> n, C, T, P, P
        temporal_blob_conv_prd = temporal_blob_conv_prd.permute(0, 2, 1, 3, 4) # n, C, T, P, P -> n, T, C, P, P
        cur_id = int(math.ceil(bf_cur_len*1.0 / (sampled_temporal_imgs.shape[2]*1.0) * temporal_blob_conv_prd.shape[1]))-1
        sampled_temporal_roi = temporal_roi[:, :temporal_blob_conv_prd.shape[1], :]
        
        
        if cfg.FPN.REL_FPN_ON:
            temporal_blob_conv_prd = temporal_blob_conv_prd[-self.num_roi_levels:]
            feat_map = []
            for ilvl in range(len(temporal_blob_conv_prd)):
                if len(temporal_blob_conv_prd[ilvl].shape) != 5:
                    temporal_blob_conv_prd[ilvl] = \
                        temporal_blob_conv_prd[ilvl].view(batch_size, -1, 
                                temporal_blob_conv_prd[ilvl].shape[1], 
                                temporal_blob_conv_prd[ilvl].shape[2], 
                                temporal_blob_conv_prd[ilvl].shape[3])
                feat_map.append(temporal_blob_conv_prd[ilvl][:, cur_id, :, :, :])
        else:
            if len(temporal_blob_conv_prd.shape) != 5:
                temporal_blob_conv_prd = \
                    temporal_blob_conv_prd.view(batch_size, -1, 
                                temporal_blob_conv_prd.shape[1], 
                                temporal_blob_conv_prd.shape[2], 
                                temporal_blob_conv_prd.shape[3])
            feat_map = temporal_blob_conv_prd[:, cur_id, :, :, :]
        
        if self.Prd_RCNN2D is not None:
            feat_map = self.Prd_RCNN2D(temporal_imgs[:, bf_cur_len])
        
        
        pooled_features, pooled_region_features_map, batch_nonleaf_node_layer_list, \
        batch_tree_bboxnum_list, batch_tree_bboxnum_sum_list, \
        batch_bboxnum_list, batch_bboxnum_sum_list, \
        batch_nonleaf_bboxnum, batch_nonleaf_bboxnum_sum, \
        batch_A, batch_A_T, batch_link, batch_non_leaf_bbox = \
            self.cluster_tree3d(roi, feat_map, det_feat_vec, sampled_temporal_roi, \
                temporal_blob_conv_prd, temporal_blob_conv_prd, sbj_id=X_list_id, obj_id=Y_list_id)
        
        obj_feature, region_feature = \
            self.feat_divide(pooled_features, batch_bboxnum_list, 
                            batch_tree_bboxnum_sum_list, batch_size)
        obj_feature = det_feat_vec + self.restore_obj_feat(obj_feature)
        
        
        obj_score = self.cem_Box_Outs.cls_score(self.relu(obj_feature))
    
        X_list = rel_ret['sbj_inds'].astype(np.int)
        Y_list = rel_ret['obj_inds'].astype(np.int)
        
        spatio_feat1, spatio_feat2, atten_map_lca, lca_list = \
            self.get_spatio_atten_feat(
                    roi, feat_map, obj_feature, 
                    pooled_region_features_map, 
                    batch_bboxnum_list, 
                    batch_bboxnum_sum_list, 
                    batch_link, batch_size, 
                    batch_nonleaf_bboxnum_sum, 
                    X_list, Y_list,)

        spatio_infer_feat_p = self.leakyrelu(self.spatio_inferA(spatio_feat1) + self.spatio_inferB(spatio_feat2))
        spatio_infer_feat_p = spatio_infer_feat_p + atten_map_lca
        spatio_infer_feat_p = spatio_infer_feat_p.view(spatio_infer_feat_p.shape[0], -1)
        spatio_infer_feat = self.spatio_infer_fc(spatio_infer_feat_p)
    
        sem_infer_feat = self.get_sem_feat(obj_feature, region_feature, 
            lca_list, X_list, Y_list, det_cls_soft, batch_size, dataset_name)
        if cfg.MODEL.USE_SPATIAL_FEAT:
            spt_feat = rel_ret['spt_feat']
        else:
            spt_feat = None
        if cfg.MODEL.USE_FREQ_BIAS or cfg.MODEL.RUN_BASELINE:
            sbj_labels = rel_ret['all_sbj_labels_int32']
            obj_labels = rel_ret['all_obj_labels_int32']
        else:
            sbj_labels = None
            obj_labels = None
            
        device_id = spatio_infer_feat.get_device()
        if sbj_labels is not None:
            sbj_labels = Variable(torch.from_numpy(sbj_labels.astype('int64'))).cuda(device_id)
        if obj_labels is not None:
            obj_labels = Variable(torch.from_numpy(obj_labels.astype('int64'))).cuda(device_id)
        if cfg.MODEL.USE_FREQ_BIAS or cfg.MODEL.RUN_BASELINE:
            assert sbj_labels is not None and obj_labels is not None
            prd_bias_scores = self.freq_bias.rel_index_with_labels(torch.stack((sbj_labels, obj_labels), 1))
        
        relation_binary_score = \
                self.relation_infer_spa(spatio_infer_feat) + \
                self.relation_infer_sem(sem_infer_feat) + \
                self.sem_s(sbj_feat) + self.sem_o(obj_feat)
        if cfg.MODEL.RUN_BASELINE:
            ttl_cls_scores = prd_bias_scores
        else:
            ttl_cls_scores = relation_binary_score
        if cfg.MODEL.USE_FREQ_BIAS and not cfg.MODEL.RUN_BASELINE:
            ttl_cls_scores += prd_bias_scores
        if cfg.MODEL.USE_SPATIAL_FEAT:
            spt_feat = Variable(torch.from_numpy(rel_ret['spt_feat'].astype('float32'))).cuda(device_id)
            prd_spt_scores = self.spt_cls_feats(spt_feat)
            ttl_cls_scores += prd_spt_scores    
    
        if not self.training:
            if cfg.MODEL.MULTI_RELATION:
                ttl_cls_scores = self.sigmoid(ttl_cls_scores)
                obj_score = F.softmax(obj_score, dim=1)
                if cfg.MODEL.USE_FREQ_BIAS or cfg.MODEL.RUN_BASELINE:
                    prd_bias_scores = self.sigmoid(prd_bias_scores)
                if cfg.MODEL.USE_SPATIAL_FEAT:
                    prd_spt_scores = self.sigmoid(prd_spt_scores)
            else:
                ttl_cls_scores = F.softmax(ttl_cls_scores, dim=1)
                obj_score = F.softmax(obj_score, dim=1)
                if cfg.MODEL.USE_FREQ_BIAS or cfg.MODEL.RUN_BASELINE:
                    prd_bias_scores = F.softmax(prd_bias_scores, dim=1)
                if cfg.MODEL.USE_SPATIAL_FEAT:
                    prd_spt_scores = F.softmax(prd_spt_scores, dim=1)

        if self.training:
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            if not (cfg.ENABLE_FRAME_PRE_PROCESSING and \
              pre_processed_frames_rpn_ret is not None) and cls_score is not None:
                # rpn loss
                rpn_kwargs.update(dict(
                    (k, rpn_ret[k]) for k in rpn_ret.keys()
                    if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
                ))
                loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)
                if cfg.FPN.FPN_ON:
                    for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                        return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                        return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
                else:
                    return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
                    return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox
                # bbox loss
                loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                    cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                    rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])
                    
                return_dict['losses']['loss_cls'] = loss_cls
                return_dict['losses']['loss_bbox'] = loss_bbox
                return_dict['metrics']['accuracy_cls'] = accuracy_cls
            
            
            
            if cfg.MODEL.USE_OBJ_LOSS_WEIGHT > 0:
                combine_loss_cls, _ = fast_rcnn_heads.fast_rcnn_cls_losses(
                    obj_score, rel_ret['det_labels'])
                return_dict['losses']['combine_loss_cls'] = \
                    cfg.MODEL.USE_OBJ_LOSS_WEIGHT * combine_loss_cls
            
            if cfg.MODEL.ADD_SCORES_ALL:
                if cfg.MODEL.MULTI_RELATION:
                    loss_cls_ttl, _ = reldn_heads.binary_prd_losses(
                        ttl_cls_scores, rel_ret['multi_prd_labels_int32'], 
                        rel_ret['keep_pair_class_int32'], pos_weight=self.pos_weight)
                else:
                    loss_cls_ttl, _ = reldn_heads.reldn_losses(
                        ttl_cls_scores, rel_ret['all_prd_labels_int32'])
                return_dict['losses']['loss_cls_ttl'] = loss_cls_ttl
            else:
                assert False, 'Only support add all scores loss!'
            
            if cfg.MODEL.USE_TRIPLET_PENALTY_LOSS_WEIGHT > 0:
                max_obj_score, _ = torch.max(obj_score[:, 1:], dim=1)
                max_obj_score = max_obj_score.view(-1)
                loss_cls_triplet_prd = reldn_heads.triplet_prd_loss(
                        ttl_cls_scores, rel_ret['multi_prd_labels_int32'], 
                        rel_ret['keep_pair_class_int32'], 
                        max_obj_score[X_list[:, 1]], max_obj_score[Y_list[:, 1]])
                return_dict['losses']['loss_cls_triplet_prd'] = \
                        cfg.MODEL.USE_TRIPLET_PENALTY_LOSS_WEIGHT * loss_cls_triplet_prd
            
            
            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)
        else:
            # Testing
            return_dict['sbj_rois'] = rel_ret['sbj_rois']
            return_dict['obj_rois'] = rel_ret['obj_rois']
            return_dict['sbj_labels'] = rel_ret['sbj_labels']
            return_dict['obj_labels'] = rel_ret['obj_labels']
            return_dict['sbj_scores'] = rel_ret['sbj_scores']
            return_dict['obj_scores'] = rel_ret['obj_scores']
            
            return_dict['det_rois'] = rel_ret['det_rois']
            return_dict['det_labels'] = rel_ret['det_labels']
            return_dict['msp_det_scores'] = obj_score
            return_dict['sbj_inds'] = X_list.reshape(-1)
            return_dict['obj_inds'] = Y_list.reshape(-1)
            return_dict['prd_scores'] = ttl_cls_scores
        
            
            if cfg.MODEL.USE_FREQ_BIAS or cfg.MODEL.RUN_BASELINE:
                return_dict['prd_scores_bias'] = prd_bias_scores
            if cfg.MODEL.USE_SPATIAL_FEAT:
                return_dict['prd_scores_spt'] = prd_spt_scores
            if cfg.MODEL.ADD_SCORES_ALL:
                return_dict['prd_ttl_scores'] = ttl_cls_scores
            if do_vis:
                return_dict['blob_conv'] = blob_conv
                
                return_dict['feat_map'] = feat_map
                return_dict['temporal_blob_conv_prd'] = temporal_blob_conv_prd
                
                return_dict['batch_A'] = batch_A
                return_dict['batch_non_leaf_bbox'] = batch_non_leaf_bbox / f_scale
                return_dict['roi'] = roi / f_scale    
                return_dict['spatio_feat1'] = spatio_feat1
                return_dict['spatio_feat2'] = spatio_feat2
            
        return_dict['pre_processed_frames_rpn_ret'] = None
        return_dict['pre_processed_temporal_roi'] = None
        if cfg.TEST.GET_FRAME_ROIS: # must use one gpu!
            return_dict['pre_processed_frames_rpn_ret'] = [{'rois': rpn_ret['rois'], \
                'labels_int32': rpn_ret['labels_int32']}, frames_rpn_ret['rois'], None]
            return_dict['pre_processed_temporal_roi'] = temporal_roi
            return_dict['file_name'] = file_name
        
        
        return return_dict

    
    def get_roi_inds(self, det_labels, lbls):
        lbl_set = np.array(lbls)
        inds = np.where(np.isin(det_labels, lbl_set))[0]
        return inds
    
    def prepare_det_rois(self, rois, cls_scores, bbox_pred, im_info, dataset_name, 
                        score_thresh=cfg.TEST.SCORE_THRESH, batch_size=cfg.TRAIN.IMS_PER_BATCH, pre_boxes=None):
        im_info = im_info.data.cpu().numpy()
        # NOTE: 'rois' is numpy array while
        # 'cls_scores' and 'bbox_pred' are pytorch tensors
        scores = cls_scores.data.cpu().numpy().reshape(cls_scores.shape[0], -1)
        
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy().reshape(bbox_pred.shape[0], -1)
        
        assert rois.shape[0] == scores.shape[0] == box_deltas.shape[0], \
            '{}, {}, {}'.format(rois.shape[0], scores.shape[0], box_deltas.shape[0])
        
        det_rois = np.empty((0, 5), dtype=np.float32)
        det_labels = np.empty((0), dtype=np.float32)
        det_scores = np.empty((0), dtype=np.float32)
        det_inds_mapping = np.empty((0), dtype=np.int32)
        for im_i in range(batch_size):
            # get all boxes that belong to this image
            inds = np.where(abs(rois[:, 0] - im_i) < 1e-06)[0]
            # unscale back to raw image space
            im_boxes = rois[inds, 1:5] / im_info[im_i, 2]
            im_scores = scores[inds]
            # In case there is 1 proposal
            im_scores = im_scores.reshape([-1, im_scores.shape[-1]])
            # In case there is 1 proposal
            im_box_deltas = box_deltas[inds]
            im_box_deltas = im_box_deltas.reshape([-1, im_box_deltas.shape[-1]])
            
            if dataset_name.find('vidvrd') >= 0 and cfg.ENABLE_FRAME_PRE_PROCESSING and pre_boxes is not None:
                im_boxes = np.tile(pre_boxes[inds, 1:5], (1, im_scores.shape[1]))
            else:
                im_scores, im_boxes = self.get_det_boxes(im_boxes, im_scores, \
                            im_box_deltas, im_info[im_i][:2] / im_info[im_i][2])
                
            im_scores, im_boxes, im_labels, im_inds = \
                self.box_results_with_nms_and_limit(im_scores, im_boxes, score_thresh)
            
            batch_inds = im_i * np.ones(
                (im_boxes.shape[0], 1), dtype=np.float32)
            
            im_det_rois = np.hstack((batch_inds, im_boxes * im_info[im_i, 2]))
            det_rois = np.append(det_rois, im_det_rois, axis=0)
            det_labels = np.append(det_labels, im_labels, axis=0)
            det_scores = np.append(det_scores, im_scores, axis=0)
            det_inds_mapping = np.append(det_inds_mapping, im_inds, axis=0)
        
        return det_rois, det_labels, det_scores, det_inds_mapping

    def get_det_boxes(self, boxes, scores, box_deltas, h_and_w):
        if cfg.TEST.BBOX_REG:
            if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
                # Remove predictions for bg class (compat with MSRA code)
                box_deltas = box_deltas[:, -4:]
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # (legacy) Optionally normalize targets by a precomputed mean and stdev
                box_deltas = box_deltas.view(-1, 4) * cfg.TRAIN.BBOX_NORMALIZE_STDS \
                             + cfg.TRAIN.BBOX_NORMALIZE_MEANS
            pred_boxes = box_utils.bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
            pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, h_and_w)
            if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
                pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        return scores, pred_boxes
    
    def box_results_with_nms_and_limit(self, scores, boxes, score_thresh=cfg.TEST.SCORE_THRESH):
        num_classes = cfg.MODEL.NUM_CLASSES
        cls_boxes = [[] for _ in range(num_classes)]
        im_inds = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > score_thresh)[0]
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4:(j + 1) * 4]
            dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
            if cfg.TEST.SOFT_NMS.ENABLED:
                nms_dets, keep_dets = box_utils.soft_nms(
                    dets_j,
                    sigma=cfg.TEST.SOFT_NMS.SIGMA,
                    overlap_thresh=cfg.TEST.NMS,
                    score_thresh=0.0001,
                    method=cfg.TEST.SOFT_NMS.METHOD
                )
                
                im_inds.append(inds[keep_dets])
                
            else:
                keep = box_utils.nms(dets_j, cfg.TEST.NMS)
                nms_dets = dets_j[keep, :]
                
                im_inds.append(inds[keep])
                
            # add labels
            label_j = np.ones((nms_dets.shape[0], 1), dtype=np.float32) * j
            nms_dets = np.hstack((nms_dets, label_j))
            # Refine the post-NMS boxes using bounding-box voting
            ## Bbox Voting changes the bbox x,y,w,h !!!!!!!!!!!!!
            if cfg.TEST.BBOX_VOTE.ENABLED:
                nms_dets = box_utils.box_voting(
                    nms_dets,
                    dets_j,
                    cfg.TEST.BBOX_VOTE.VOTE_TH,
                    scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
                )
            cls_boxes[j] = nms_dets
        
        # Limit to max_per_image detections **over all classes**
        if cfg.TEST.DETECTIONS_PER_IM > 0:
            image_scores = np.hstack(
                [cls_boxes[j][:, -2] for j in range(1, num_classes)]
            )
            if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
                image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
                for j in range(1, num_classes):
                    keep = np.where(cls_boxes[j][:, -2] >= image_thresh)[0]
                    im_inds[j-1] = im_inds[j-1][keep]
                    cls_boxes[j] = cls_boxes[j][keep, :]
        im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
        boxes = im_results[:, :-2]
        scores = im_results[:, -2]
        labels = im_results[:, -1]
        
        im_inds = np.concatenate(im_inds, 0)

        return scores, boxes, labels, im_inds

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIAlign',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = ROIPool((resolution, resolution), sc)(bl_in, rois)
                    elif method == 'RoIAlign':
                        xform_out = ROIAlign(
                            (resolution, resolution), sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)
            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)
            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = ROIPool((resolution, resolution), spatial_scale)(blobs_in, rois)
            elif method == 'RoIAlign':
                xform_out = ROIAlign(
                    (resolution, resolution), spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
        return blob_conv

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
    
    
    def get_tracklet(self, img_list_f, img_list_b, roi, batch_size=None, frames_rois_wo_cur=None, 
      torch_frames_box_feat_wo_cur=None, torch_box_feat=None, device_id=None, post_prdcnn_T=1):
        """
            img_list_f: front current frame
            img_list_b: behind current frame
            roi: torch.tensor: (batch_id, xmin, ymin, xmax, ymax, class, score). size:(bbox_num, 6)
            
            return:
                # track_list: temporal_roi: torch.tensor: (batch_id, xmin, ymin, xmax, ymax). size:(bbox_num, T, 5)
                track_list: temporal_roi: np.array: (batch_id, xmin, ymin, xmax, ymax). size:(bbox_num, T, 5)
        """
        if frames_rois_wo_cur is None:
            T = post_prdcnn_T
            t_rois = np.expand_dims(roi, axis=1).repeat(T, axis=1)
            return t_rois
            
        if len(img_list_f) <= 1 and len(img_list_b) <= 1:
            return roi.unsqueeze(1)
        
        
        frames_box_feat_wo_cur = torch_frames_box_feat_wo_cur.clone().detach()
        box_feat = torch_box_feat.clone().detach()
        frames_box_feat_wo_cur.requires_grad = False
        box_feat.requires_grad = False
        frames_box_feat_wo_cur = F.normalize(frames_box_feat_wo_cur, p=2, dim=1)
        box_feat = F.normalize(box_feat, p=2, dim=1)
        frames_box_feat_wo_cur = frames_box_feat_wo_cur.data.cpu().numpy()
        box_feat = box_feat.data.cpu().numpy()
        
        track_list = []
        wh_frames_rois_wo_cur = frames_rois_wo_cur.copy()
        wh_roi = roi.copy()
        wh_roi[:, 3] = wh_roi[:, 3] - wh_roi[:, 1] + 1
        wh_roi[:, 4] = wh_roi[:, 4] - wh_roi[:, 2] + 1
        
        wh_frames_rois_wo_cur[:, 3] = wh_frames_rois_wo_cur[:, 3] - wh_frames_rois_wo_cur[:, 1] + 1
        wh_frames_rois_wo_cur[:, 4] = wh_frames_rois_wo_cur[:, 4] - wh_frames_rois_wo_cur[:, 2] + 1
        
        wh_frames_rois_wo_cur = np.hstack([wh_frames_rois_wo_cur[:, 0:5], frames_box_feat_wo_cur])
        wh_roi = np.hstack([wh_roi[:, 0:5], box_feat])
        
        for b in range(batch_size):
            rois_backward = []
            rois_forward = []
            
            for t in range(0, len(img_list_f)-1):
                idx = np.where(abs(wh_frames_rois_wo_cur[:, 0] - (b*batch_size+t)) < 1e-6)[0]
                f_roi = wh_frames_rois_wo_cur[idx]
                f_roi[:, 0] = (f_roi[:, 0] - t) / batch_size
                rois_backward.append(f_roi)
                
            id_cur = np.where(abs(wh_roi[:, 0] - b) < 1e-6)[0]
            f_roi_cur = wh_roi[id_cur]
            rois_backward.append(f_roi_cur)
            rois_forward.append(f_roi_cur)
            
            for t in range(len(img_list_f)-1, len(img_list_f)+len(img_list_b)-2):
                idx = np.where(abs(wh_frames_rois_wo_cur[:, 0] - (b*batch_size+t)) < 1e-6)[0]
                f_roi = wh_frames_rois_wo_cur[idx]
                f_roi[:, 0] = (f_roi[:, 0] - t) / batch_size
                rois_forward.append(f_roi)
            

            _, tracklet_2 = self.obj_tracking(img_list_b, rois_forward, is_reverse=False)
            _, tracklet_1 = self.obj_tracking(img_list_f, rois_backward, is_reverse=True)
            tracklet = np.concatenate([tracklet_1[:, :-1, :], tracklet_2], axis=1)
            

            tracklet[:, :, 2] = tracklet[:, :, 2] + tracklet[:, :, 0] - 1
            tracklet[:, :, 3] = tracklet[:, :, 3] + tracklet[:, :, 1] - 1
            
            id = b * np.ones((len(tracklet), len(img_list_f) + len(img_list_b) - 1, 1))
            tracklet = np.concatenate([id, tracklet], 2)
            
            track_list.append(tracklet)
            
        track_list = np.concatenate(track_list, 0)
        
        return track_list
        
    def feat_divide(self, pooled_features, batch_bboxnum_list, batch_tree_bboxnum_sum_list, batch_size):
        obj_feature = []
        region_feature = []
        if batch_size == 1:
            return pooled_features[batch_tree_bboxnum_sum_list[0]:batch_tree_bboxnum_sum_list[0]+batch_bboxnum_list[0]], \
                pooled_features[batch_tree_bboxnum_sum_list[0]+batch_bboxnum_list[0]:batch_tree_bboxnum_sum_list[1]]
        for b in range(batch_size):
            obj_feature.append(pooled_features[batch_tree_bboxnum_sum_list[b]:batch_tree_bboxnum_sum_list[b]+batch_bboxnum_list[b]])
            region_feature.append(pooled_features[batch_tree_bboxnum_sum_list[b]+batch_bboxnum_list[b]:batch_tree_bboxnum_sum_list[b+1]])
        obj_feature = torch.cat(obj_feature, 0).to(device=pooled_features.device)
        region_feature = torch.cat(region_feature, 0).to(device=pooled_features.device)
        return obj_feature, region_feature
    
    
    def get_spatio_atten_feat(self, roi, featmap, obj_feature, pooled_region_features_map, batch_bboxnum_list, 
      batch_bboxnum_sum_list, batch_link, batch_size, batch_nonleaf_bboxnum_sum, X_list, Y_list):
        """
            
            return:
                # ans: torch.tensor list. tensor size:(N, low_feat_vec_dim * 3, H, W)
                ans: torch.tensor. tensor size:(sum{i^2}, low_feat_vec_dim * 3, H, W)
                #lca_list: long np.array list
                lca_list: long torch.tensor list
        """
        ans1 = []
        ans2 = []
        atten_map_lca_list = []
        lca_list = []
        device_id = pooled_region_features_map.get_device()
        
        atten_s_q = self.spatio_atten_s(obj_feature)
        atten_o_q = self.spatio_atten_o(obj_feature)
        atten_map = self.spatio_atten_map(featmap)
        atten_map_q = self.atten_q_map(featmap)
        
        for b in range(batch_size):
            if len(X_list.shape) > 1 and X_list.shape[1] >= 2:
                bid = np.where(X_list[:, 0]==b)[0]
                X = X_list[bid, -1].reshape(-1)
                Y = Y_list[bid, -1].reshape(-1)
            else:
                X = X_list.reshape(-1)
                Y = Y_list.reshape(-1)
            lca = self.get_lca(X, Y, batch_link[b])
            
            
            q_s = atten_s_q[X + batch_bboxnum_sum_list[b]]
            q_o = atten_o_q[Y + batch_bboxnum_sum_list[b]]
            
            lca = lca - batch_bboxnum_list[b] + batch_nonleaf_bboxnum_sum[b]

            rel_rois = self.get_rel_rois(roi[X + batch_bboxnum_sum_list[b]], 
                                        roi[Y + batch_bboxnum_sum_list[b]])
            rel_rois = dict(rois=rel_rois)
        
            
            atten_map_q_lca = self.roi_feature_transform(atten_map_q, 
                        rel_rois, resolution=self.pool_size, spatial_scale=self.rel_spatial_scale)
            atten_map_lca = self.roi_feature_transform(atten_map, 
                        rel_rois, resolution=self.pool_size, spatial_scale=self.rel_spatial_scale)
            
            s_map = self.get_attention_mask_map(q_s, atten_map_q_lca, atten_map_lca)
            o_map = self.get_attention_mask_map(q_o, atten_map_q_lca, atten_map_lca)
            

            lca_list.append(lca)
            if batch_size == 1:
                return s_map, o_map, atten_map_lca, lca_list
                
            ans1.append(s_map)
            ans2.append(o_map)
            atten_map_lca_list.append(atten_map_lca)
            
        ans1 = torch.cat(ans1, 0).to(device=device_id)
        ans2 = torch.cat(ans2, 0).to(device=device_id)
        atten_map_lca_list = torch.cat(atten_map_lca_list, 0).to(device=device_id)
        return ans1, ans2, atten_map_lca_list, lca_list
    
    def get_rel_rois(self, rois1, rois2):
        assert (rois1[:, 0] == rois2[:, 0]).all()
        xmin = np.minimum(rois1[:, 1], rois2[:, 1])
        ymin = np.minimum(rois1[:, 2], rois2[:, 2])
        xmax = np.maximum(rois1[:, 3], rois2[:, 3])
        ymax = np.maximum(rois1[:, 4], rois2[:, 4])
        return np.vstack((rois1[:, 0], xmin, ymin, xmax, ymax)).transpose()
        
    def get_attention_mask_map(self, q_vec, q_map, v_map):
        """
            q_vec: size:(N, K)
            q_map: size:(N, K, H, W)
            
            return:
                ans: size:(N, V, H, W)
        """
        q_vec = q_vec.view(q_vec.shape[0], q_vec.shape[1], 1, 1)
        
        score = torch.mul(q_vec, q_map).sum(1)
        score = score.view(score.shape[0], -1)
        score = F.softmax(score, 1)    # N, H*W
        score = score.view(q_map.shape[0], 1, q_map.shape[2], q_map.shape[3])
        
        ans = torch.mul(v_map, score)
        return ans
            
    def get_lca(self, X, Y, link_mat):
        pos = (link_mat[X] != link_mat[Y]) * link_mat[X]
        pos = pos.argmax(1) + 1
        x_link_mat = link_mat[X]
        lca = x_link_mat[range(len(pos)), pos]
        return lca
        
    def obj_map_attention(self, q_vec, q_map, v_vec):
        """
            q_vec: size:(N, K)
            v_vec: size:(N, V)
            q_map: size:(N, K, H, W)
            
            return:
                ans: size:(N, V, H, W)
        """
        tq_vec = q_vec.unsqueeze(2)
        tq_map = q_map.view(q_map.shape[0], q_map.shape[1], -1)
        
        score = torch.mul(tq_vec, tq_map).sum(1)
        score = F.softmax(score, 1)    # N, H*W
        score = score.view(q_map.shape[0], q_map.shape[2], q_map.shape[3])
        score = score.unsqueeze(1)    # N, 1, H, W
        
        tv_vec = v_vec.view(v_vec.shape[0], v_vec.shape[1], 1, 1)
        
        ans = torch.mul(score, tv_vec)    # N, V, H, W
        return ans
        
    def get_sem_feat(self, obj_feature, region_feature, lca_list, X_list, Y_list, softmax_obj_vec, batch_size, dataset_name):
        ans = []
        device_id = obj_feature.get_device()
        e_s = self.embed_s(softmax_obj_vec)
        e_o = self.embed_o(softmax_obj_vec)
        low_dim_s_feat = self.sem_s_e(e_s)
        low_dim_o_feat = self.sem_o_e(e_o)
        low_dim_r_feat = self.sem_r(region_feature)
        
        
        for b in range(batch_size):
            if len(X_list.shape) > 1 and X_list.shape[1] >= 2:
                bid = np.where(X_list[:, 0]==b)[0]
                X = X_list[bid, -1].reshape(-1)
                Y = Y_list[bid, -1].reshape(-1)
            else :
                X = X_list.reshape(-1)
                Y = Y_list.reshape(-1)
            lca = lca_list[b]
            sem_feat_vec = low_dim_s_feat[X] + low_dim_r_feat[lca] + low_dim_o_feat[Y]
            if batch_size == 1:
                return sem_feat_vec
                
            ans.append(sem_feat_vec)
        ans = torch.cat(ans, 0).to(device=obj_feature.device)
        return ans
    
    def frames_sample(self, tot_frames_len, bf_cur_len):
        ans = []
        bf_id = np.random.choice(bf_cur_len, 1)[0]
        if bf_cur_len <= 0:
            if tot_frames_len - bf_cur_len - 1 >= 1:
                bh_id = np.random.choice(tot_frames_len - bf_cur_len - 1, 1)[0] + bf_cur_len + 1
                ans = [bf_cur_len, bh_id]
                cur_id = 0
            else:
                ans = [bf_cur_len]
                cur_id = 0
        else:
            if tot_frames_len - bf_cur_len - 1 >= 1:
                bh_id = np.random.choice(tot_frames_len - bf_cur_len - 1, 1)[0] + bf_cur_len + 1
                ans = [bf_id, bf_cur_len, bh_id]
                cur_id = 1
            else:
                ans = [bf_id, bf_cur_len]
                cur_id = 1
        return cur_id, ans
    
    def frames_sample_for_saving_mem(self, tot_frames_len, bf_cur_len):
        ans = []
        candi_list = []
        vid_len = min(cfg.MAX_FRAMES_NUM_IN_MEMORY, tot_frames_len)
        for i in range(bf_cur_len+vid_len-tot_frames_len, bf_cur_len+1):
            if bf_cur_len - i < 0:
                continue
            if bf_cur_len + vid_len - i > tot_frames_len:
                continue
            st = bf_cur_len - i
            ed = bf_cur_len + vid_len - i
            cur_id = i
            candi_list.append((st, ed, cur_id, ed-st))
        candi_id = np.random.choice(len(candi_list), 1)[0]
        st, ed, cur_id, _ = candi_list[candi_id]
        ans = [i for i in range(st, ed)]
        return cur_id, ans
    
    def saved_tracklet_mapping(self, cur_rois, temporal_roi, bf_cur_len):
        tracklet = []
        flg = False
        if len(cur_rois) != len(temporal_roi):
            flg = True
            return tracklet, flg
            
        vis = np.zeros(len(cur_rois), dtype=np.int32)
        for i in range(len(cur_rois)):
            idx = np.where(np.sum(abs(cur_rois[i, :] - temporal_roi[:, bf_cur_len, :]), axis=1) < 4e-1)[0]
            if len(idx) <= 0:
                flg = True
                print('miss tracklet.')
                tmp = cur_rois[[i], :]
                tmp = np.pad(tmp, ((bf_cur_len,temporal_roi.shape[1]-bf_cur_len-1), (0,0)),
                            'constant',constant_values=0)
                tracklet.append(tmp)
                continue
                
            t_id = idx[0]
            for j in range(len(idx)):
                if vis[idx[j]] == 0:
                    t_id = idx[j]
            tracklet.append(temporal_roi[t_id, :, :])
            vis[t_id] += 1
        tracklet = np.stack(tracklet, 0)
        return tracklet, flg
        
    def _add_rel_multilevel_rois(self, blobs, key_id='rois'):
        """By default training RoIs are added for a single feature map level only.
        When using FPN, the RoIs must be distributed over different FPN levels
        according the level assignment heuristic (see: modeling.FPN.
        map_rois_to_fpn_levels).
        """
        lvl_min = cfg.FPN.ROI_MIN_LEVEL
        lvl_max = cfg.FPN.ROI_MAX_LEVEL

        def _distribute_rois_over_fpn_levels(rois_blob_names):
            """Distribute rois over the different FPN levels."""
            # Get target level for each roi
            # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
            # the box coordinates from columns 1:5
            lowest_target_lvls = None
            for rois_blob_name in rois_blob_names:
                target_lvls = fpn_utils.map_rois_to_fpn_levels(
                    blobs[rois_blob_name][:, 1:5], lvl_min, lvl_max)
                if lowest_target_lvls is None:
                    lowest_target_lvls = target_lvls
                else:
                    lowest_target_lvls = np.minimum(lowest_target_lvls, target_lvls)
            for rois_blob_name in rois_blob_names:
                # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
                fpn_utils.add_multilevel_roi_blobs(
                    blobs, rois_blob_name, blobs[rois_blob_name], lowest_target_lvls, lvl_min,
                    lvl_max)
        _distribute_rois_over_fpn_levels([key_id])
        
    def downsample_for_3d(self, rois, t_rois, t_feat_map, resized_scale, t_scale):
        resized_scale = 1.0 * t_scale * resized_scale
        resize_rois = rois.copy()
        resize_rois[:, 1:5] *= float(resized_scale)
        resize_rois = resize_rois.astype(np.float32)
        
        if t_rois is not None:
            resize_t_rois = t_rois.copy()
            resize_t_rois[:, :, 1:5] *= float(resized_scale)
            resize_t_rois = resize_t_rois.astype(np.float32)
        else:
            resize_t_rois = None
        
        resize_t_feat_map = t_feat_map
        return resize_rois, resize_t_rois, resize_t_feat_map, resized_scale