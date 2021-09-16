import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import random
import time
import copy
from torch.autograd import Variable

import combine_tree.utils_t as utils_t
from combine_tree.simple_motionsqueeze import simple_motionsqueeze as MSq
from combine_tree.cluster_tree import *

from core.config import cfg
import utils.fpn as fpn_utils
from model.roi_layers import ROIPool, ROIAlign

class cluster_tree3d(cluster_tree2D):
    def __init__(self, feat_map_shape, feat_combine, feat_vec_dim, feat_region_vec_dim, hidden_dim, low_feat_vec_dim, Roialign, spatial_scale, group_coeffi=1, max_tree_height=7, pool_size=5, BoxHead=None, cluster_num_partition=2, msp_mode='tree-gru'):
        if BoxHead is not None:
            feat_region_vec_dim = BoxHead.dim_out
        super(cluster_tree3d, self).__init__(feat_map_shape, 
                                                feat_combine, 
                                                feat_vec_dim, 
                                                feat_region_vec_dim, 
                                                hidden_dim, 
                                                Roialign, 
                                                spatial_scale, 
                                                group_coeffi=group_coeffi, 
                                                max_tree_height=max_tree_height, 
                                                pool_size=pool_size, 
                                                cluster_num_partition=cluster_num_partition, 
                                                msp_mode=msp_mode)
        C, H, W = feat_map_shape
        
        self.temporal_ROIalign_obj = self.roi_feature_transform
        
        self.sqrt_low_dim_embed_dim = math.sqrt(low_feat_vec_dim)

        
        self.temporal_ROIalign_region = self.roi_feature_transform
        
        
        temporal_obj_feat_val_dim = feat_vec_dim
        self.use_pretrain_BoxHead = False
        if BoxHead is not None:
            print('Loading BoxHead in combine_tree.')
            self.temporal_obj_fc = copy.deepcopy(BoxHead)
            #del self.temporal_obj_fc.roi_xform
            for p in self.temporal_obj_fc.parameters():
                p.requires_grad = True
                
            self.temporal_r_fc = copy.deepcopy(BoxHead)
            #del self.temporal_r_fc.roi_xform
            for p in self.temporal_r_fc.parameters():
                p.requires_grad = True
                
            self.use_pretrain_BoxHead = True
            
            temporal_obj_feat_val_dim = BoxHead.dim_out
        else:
            self.temporal_obj_fc = nn.Sequential(
                                    nn.Linear(C*pool_size*pool_size, feat_vec_dim, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(feat_vec_dim, feat_vec_dim, bias=True),)   
            self.temporal_r_fc = nn.Sequential(
                                    nn.Linear(C*pool_size*pool_size, feat_region_vec_dim, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(feat_region_vec_dim, feat_region_vec_dim, bias=True),)
        
        self.temporal_obj_feat_val = None
        self.temporal_r_feat_val = None
        
        self.temporal_obj_atten_q_s = nn.Sequential(
                                    nn.Linear(feat_vec_dim, low_feat_vec_dim, bias=False),)    
        self.temporal_obj_atten_q_o = nn.Sequential(
                                    nn.Linear(temporal_obj_feat_val_dim, low_feat_vec_dim, bias=False),)
        self.temporal_obj_feat_val = nn.Sequential(
                                    nn.Linear(temporal_obj_feat_val_dim, feat_vec_dim//4, bias=False),
                                    nn.LayerNorm(feat_vec_dim//4),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(feat_vec_dim//4, feat_vec_dim, bias=False),)
        

        self.temporal_r_atten_q_s = nn.Sequential(
                                nn.Linear(feat_region_vec_dim, low_feat_vec_dim, bias=False),)
        self.temporal_r_atten_q_o = nn.Sequential(
                                nn.Linear(feat_region_vec_dim, low_feat_vec_dim, bias=False),)
        self.temporal_r_feat_val = nn.Sequential(
                                nn.Linear(feat_region_vec_dim, feat_region_vec_dim//4, bias=False),
                                nn.LayerNorm(feat_region_vec_dim//4),
                                nn.ReLU(inplace=True),
                                nn.Linear(feat_region_vec_dim//4, feat_region_vec_dim, bias=False),)
        
        self.reset_temporal()
        
        
    def reset_temporal(self):
        self.temporal_obj_atten_q_s.apply(utils_t.weight_init_mynn_Xavier)
        self.temporal_obj_atten_q_o.apply(utils_t.weight_init_mynn_Xavier)
        self.temporal_obj_feat_val.apply(utils_t.weight_init_mynn_Xavier)
        
        self.temporal_r_atten_q_s.apply(utils_t.weight_init_mynn_Xavier)
        self.temporal_r_atten_q_o.apply(utils_t.weight_init_mynn_Xavier)
        self.temporal_r_feat_val.apply(utils_t.weight_init_mynn_Xavier)
        
        if not self.use_pretrain_BoxHead:
            self.temporal_r_fc.apply(utils_t.weight_init_mynn_Xavier)    
            self.temporal_obj_fc.apply(utils_t.weight_init_mynn_Xavier)
        
        
    def forward(self, roi, feat_map, obj_feat_vec, temporal_roi, temporal_feat_map, temporal_obj_feat_map, sbj_id=None, obj_id=None):
        """
            # roi: torch.tensor: (batch_id, xmin, ymin, xmax, ymax). size:(bbox_num, 5)    # current frame roi
            roi: numpy. size:(bbox_num, 5)    # current frame roi
            feat_map: torch.tensor: size:(batch_size, C, H, W)
            obj_feat_vec: torch.tensor: size:(bbox_num, K)
            
            # temporal_roi: torch.tensor: (batch_id, xmin, ymin, xmax, ymax, class, score). size:(bbox_num, T, 7)    # T must from past to future
            temporal_roi: numpy. size:(bbox_num, 5). size:(bbox_num, T, 5)
            temporal_feat_map: torch.tensor: size:(batch_size, T, C, H, W)
            
            # Now, temporal_feat_map == temporal_obj_feat_map
        """
        
        if isinstance(feat_map, list):
            batch_size = feat_map[0].shape[0]
        else:
            batch_size = feat_map.shape[0]
            
        if isinstance(temporal_feat_map, list):
            T = temporal_feat_map[0].shape[1]
        else:
            T = temporal_feat_map.shape[1]
        
        bbox_id, batch_bboxnum_list, batch_bboxnum_sum_list = self.get_bbox_id(roi, batch_size)
        
        if isinstance(temporal_feat_map, list):
            temporal_feat_map_flat = []
            C, H, W = temporal_feat_map[0].shape[2:5]
            for i in range(len(temporal_feat_map)):
                temporal_feat_map_flat.append(temporal_feat_map[i].view(batch_size * T, C, H, W))
        else:
            C, H, W = temporal_feat_map.shape[2:5]
            temporal_feat_map_flat = temporal_feat_map.view(batch_size * T, C, H, W)
        
        if isinstance(temporal_obj_feat_map, list):
            temporal_obj_feat_map_flat = []
            C, H, W = temporal_obj_feat_map[0].shape[2:5]
            for i in range(len(temporal_obj_feat_map)):
                temporal_obj_feat_map_flat.append(temporal_obj_feat_map[i].view(batch_size * T, C, H, W))
        else:
            C, H, W = temporal_obj_feat_map.shape[2:5]
            temporal_obj_feat_map_flat = temporal_obj_feat_map.view(batch_size * T, C, H, W)
        
        temporal_batch_leaf_bbox = temporal_roi[:, :, 1:5]
        temporal_batch_leaf_bbox = temporal_batch_leaf_bbox.reshape(-1, 4)
        temporal_batch_boxes_id = [i * np.ones((batch_bboxnum_list[i // T], 1), dtype=np.int) for i in range(batch_size * T)]
        temporal_batch_boxes_id = np.vstack(temporal_batch_boxes_id)
        temporal_batch_boxes_for_align = np.concatenate([temporal_batch_boxes_id, temporal_batch_leaf_bbox], 1).astype(np.float32)
        
        temporal_obj_roi_blob = {'rois': temporal_batch_boxes_for_align}
        if isinstance(temporal_obj_feat_map_flat, list):
            self._add_rel_multilevel_rois(temporal_obj_roi_blob)
        
        temporal_aggregated_obj_feat = []
        temporal_batch_pooled_obj_features_map = \
            self.temporal_ROIalign_obj(temporal_obj_feat_map_flat, 
                                        temporal_obj_roi_blob, 
                                        resolution=self.pool_size, 
                                        spatial_scale=self.spatial_scale)
        
        if cfg.NO_TEMPORAL_FUSION:
            temporal_aggregated_obj_feat = obj_feat_vec
        else:
            for b in range(batch_size):
                n = batch_bboxnum_sum_list[b+1] - batch_bboxnum_sum_list[b]
                temporal_pooled_obj_features_map = temporal_batch_pooled_obj_features_map[batch_bboxnum_sum_list[b]*T : batch_bboxnum_sum_list[b+1]*T]
                temporal_pooled_obj_features_map = temporal_pooled_obj_features_map.view(T, n, -1, self.pool_size, self.pool_size)
                temporal_pooled_obj_features_map = temporal_pooled_obj_features_map.permute(1, 0, 2, 3, 4).contiguous() # n, t, c, P, P
                
                temporal_pooled_obj_features = temporal_pooled_obj_features_map
                if self.use_pretrain_BoxHead:
                    temporal_pooled_obj_features = self.temporal_obj_fc(temporal_pooled_obj_features)    # n, T, K
                    temporal_pooled_obj_features = temporal_pooled_obj_features.view(n, T, -1)
                else:
                    
                    temporal_pooled_obj_features = self.temporal_obj_fc(temporal_pooled_obj_features)    # n, T, K
                
                atten_obj_feat = self.temporal_obj_attention(
                    obj_feat_vec[batch_bboxnum_sum_list[b]:batch_bboxnum_sum_list[b+1]], temporal_pooled_obj_features)    # n, K
                if self.temporal_obj_feat_val is not None:
                    atten_obj_feat = self.temporal_obj_feat_val(atten_obj_feat)
                temporal_aggregated_obj_feat.append(atten_obj_feat)
            if batch_size == 1:
                temporal_aggregated_obj_feat = atten_obj_feat
            else:
                temporal_aggregated_obj_feat = torch.cat(temporal_aggregated_obj_feat, 0).to(device=obj_feat_vec.device)
            temporal_aggregated_obj_feat = obj_feat_vec + temporal_aggregated_obj_feat
        
        
        batch_leaf_bbox = roi[:, 1:5]
        batch_nonleaf_node_layer_list, batch_tree_node_bbox = \
            self.node_cluster(batch_leaf_bbox, 
                                batch_size, 
                                batch_bboxnum_list, 
                                batch_bboxnum_sum_list, 
                                dist_type='euclid', 
                                feat_type='coordinate', 
                                sbj_id=sbj_id, 
                                obj_id=obj_id,
                                add_batch_bbox_feat=None)
        
        tree_bbox_id, batch_tree_bboxnum_list, batch_tree_bboxnum_sum_list = \
            self.get_bbox_id(batch_tree_node_bbox, batch_size)
        
        batch_A, batch_A_T, batch_link = \
            self.get_A_and_link(batch_size, batch_tree_bboxnum_list, batch_nonleaf_node_layer_list, obj_feat_vec.device)
        
        batch_non_leaf_bbox = []
        batch_nonleaf_bboxnum = []
        batch_nonleaf_bboxnum_sum = [0]
        s = 0
        for b in range(batch_size):
            batch_non_leaf_bbox.append(batch_tree_node_bbox[batch_tree_bboxnum_sum_list[b]+batch_bboxnum_list[b] : batch_tree_bboxnum_sum_list[b+1], :])
            s = s + batch_tree_bboxnum_list[b]-batch_bboxnum_list[b]
            batch_nonleaf_bboxnum.append(batch_tree_bboxnum_list[b]-batch_bboxnum_list[b])
            batch_nonleaf_bboxnum_sum.append(s)
        batch_non_leaf_bbox = np.concatenate(batch_non_leaf_bbox, 0).astype(np.float32)
        

        region_roi_blob = {'rois': batch_non_leaf_bbox}
        if isinstance(feat_map, list):
            self._add_rel_multilevel_rois(region_roi_blob)
        
        pooled_region_features_map = \
            self.ROIalign_region(feat_map, 
                                region_roi_blob, 
                                resolution=self.pool_size, 
                                spatial_scale=self.spatial_scale)
        
        pooled_region_features = \
            self.fc_r(pooled_region_features_map.view(len(pooled_region_features_map), -1))
        
        temporal_nonleaf_node_roi = \
            self.temporal_get_new_bbox_feat(batch_nonleaf_node_layer_list, \
                batch_tree_bboxnum_sum_list, batch_bboxnum_list, \
                batch_bboxnum_sum_list, temporal_roi[:, :, 1:5], \
                batch_size, feat_type='coordinate')
        
        
        temporal_nonleaf_node_roi_flat = temporal_nonleaf_node_roi.reshape(-1, 4)
        
        temporal_batch_nonleaf_node_id = \
            [i * np.ones((batch_nonleaf_bboxnum[i // T], 1), dtype=np.int) for i in range(batch_size * T)]
        temporal_batch_nonleaf_node_id = np.vstack(temporal_batch_nonleaf_node_id)
        temporal_batch_nonleaf_node_for_align = \
            np.concatenate([temporal_batch_nonleaf_node_id, temporal_nonleaf_node_roi_flat], 1).astype(np.float32)
        
        temporal_reg_roi_blob = {'rois': temporal_batch_nonleaf_node_for_align}
        if isinstance(temporal_obj_feat_map_flat, list):
            self._add_rel_multilevel_rois(temporal_reg_roi_blob)
        
        
        temporal_aggregated_region_feat = []
        temporal_batch_pooled_region_features_map = \
            self.temporal_ROIalign_region(temporal_obj_feat_map_flat, 
                                            temporal_reg_roi_blob, 
                                            resolution=self.pool_size, 
                                            spatial_scale=self.spatial_scale)
        if cfg.NO_TEMPORAL_FUSION:
            temporal_aggregated_region_feat = pooled_region_features
        else:
            for b in range(batch_size):
                n = batch_nonleaf_bboxnum[b]
                temporal_pooled_region_features_map = \
                    temporal_batch_pooled_region_features_map[batch_nonleaf_bboxnum_sum[b]*T : batch_nonleaf_bboxnum_sum[b+1]*T]
                temporal_pooled_region_features_map = \
                    temporal_pooled_region_features_map.view(T, n, -1, self.pool_size, self.pool_size)
                temporal_pooled_region_features_map = \
                    temporal_pooled_region_features_map.permute(1, 0, 2, 3, 4).contiguous() # n, t, c, P, P
                
                
                temporal_pooled_region_features = temporal_pooled_region_features_map
                if self.use_pretrain_BoxHead:
                    temporal_pooled_region_features = self.temporal_r_fc(temporal_pooled_region_features)    # n, T, K
                    temporal_pooled_region_features = temporal_pooled_region_features.view(n, T, -1)
                else:
                    temporal_pooled_region_features = self.temporal_r_fc(temporal_pooled_region_features)    # n, T, K
                
                atten_r_feat = self.temporal_r_attention(
                        pooled_region_features[batch_nonleaf_bboxnum_sum[b]:batch_nonleaf_bboxnum_sum[b+1]], 
                        temporal_pooled_region_features)    # n, K
                if self.temporal_r_feat_val is not None:
                    atten_r_feat = self.temporal_r_feat_val(atten_r_feat)
                
                temporal_aggregated_region_feat.append(atten_r_feat)
            if batch_size == 1:
                temporal_aggregated_region_feat = atten_r_feat
            else:
                temporal_aggregated_region_feat = \
                    torch.cat(temporal_aggregated_region_feat, 0).to(device=pooled_region_features.device)
            temporal_aggregated_region_feat = pooled_region_features + temporal_aggregated_region_feat
        
        temporal_aggregated_r2c_feat = self.r2c(temporal_aggregated_region_feat)
        temporal_aggregated_o2c_feat = self.o2c(temporal_aggregated_obj_feat)
        
        pooled_features = []
        for b in range(batch_size):
            pooled_features.append(
                temporal_aggregated_o2c_feat[batch_bboxnum_sum_list[b] : batch_bboxnum_sum_list[b+1]])
            pooled_features.append(
                temporal_aggregated_r2c_feat[batch_nonleaf_bboxnum_sum[b] : batch_nonleaf_bboxnum_sum[b+1]])
        pooled_features = torch.cat(pooled_features, 0).to(device=temporal_aggregated_o2c_feat.device)
        
        aggregated_feat = []
        for i in range(self.group_coeffi):
            input_group_feat = \
                pooled_features[:, (self.feat_combine//self.group_coeffi)*i : (self.feat_combine//self.group_coeffi)*(i+1)]
            group_aggerated_feat = \
                self.tree_message_passing_list[i](batch_tree_node_bbox, \
                    input_group_feat, batch_size, batch_nonleaf_node_layer_list, \
                    batch_tree_bboxnum_list, batch_tree_bboxnum_sum_list, \
                    batch_bboxnum_list, batch_A, batch_A_T, batch_link)
            aggregated_feat.append(group_aggerated_feat)
        if self.group_coeffi > 0:
            aggregated_feat = torch.cat(aggregated_feat, 1).to(device=pooled_features.device)
            pooled_features = self.FFN(aggregated_feat)
        
        return pooled_features, pooled_region_features_map, batch_nonleaf_node_layer_list, \
            batch_tree_bboxnum_list, batch_tree_bboxnum_sum_list, batch_bboxnum_list, \
            batch_bboxnum_sum_list, batch_nonleaf_bboxnum, batch_nonleaf_bboxnum_sum, \
            batch_A, batch_A_T, batch_link, batch_non_leaf_bbox
        
    def temporal_obj_attention(self, s, o):
        """
            s: size:(N, K)
            o: size:(N, T, K)
            
            return:
                ans: size:(N, K)
        """
        q_s = self.temporal_obj_atten_q_s(s).unsqueeze(1)    # N, 1, K
        q_o = self.temporal_obj_atten_q_o(o)
        
        score = torch.mul(q_s, q_o).sum(2) / self.sqrt_low_dim_embed_dim    # N, T
        score = F.softmax(score, 1).unsqueeze(2)    # N, T, 1
        ans = (score * o).sum(1)    # N, K
        return ans
        
    def temporal_r_attention(self, s, o):
        """
            s: size:(N, K)
            o: size:(N, T, K)
            
            return:
                ans: size:(N, K)
        """
        q_s = self.temporal_r_atten_q_s(s).unsqueeze(1)    # N, 1, K
        q_o = self.temporal_r_atten_q_o(o)
        score = torch.mul(q_s, q_o).sum(2) / self.sqrt_low_dim_embed_dim    # N, T
        score = F.softmax(score, 1).unsqueeze(2)    # N, T, 1
        ans = (score * o).sum(1)    # N, K
        return ans
    
    def temporal_get_new_bbox_feat(self, batch_nonleaf_node_layer_list, batch_tree_bboxnum_sum_list, batch_bboxnum_list, batch_bboxnum_sum_list, temporal_bbox_feat, batch_size, feat_type='coordinate'):
        """
            batch_nonleaf_node_layer_list: list. example: [[{1: [0]}], [{1: [0]}], [{3: [2], 4: [1, 0]}, {5: [3, 4]}]]
            temporal_bbox_feat: numpy.array: size: (bbox_num, T, 4)
            return:
                temporal_nonleaf_node_roi: temporal_nonleaf_node_roi: np.array: (xmin, ymin, xmax, ymax). size:(bbox_num, T, 4)
        """
        T = temporal_bbox_feat.shape[1]
        new_node_feat = np.zeros((batch_tree_bboxnum_sum_list[batch_size], T, 4), dtype=np.float32)
        temporal_nonleaf_node_roi = []
        if feat_type == 'coordinate':
            for b in range(batch_size):
                new_node_feat[batch_tree_bboxnum_sum_list[b]:batch_tree_bboxnum_sum_list[b]+batch_bboxnum_list[b],:,:] = \
                    temporal_bbox_feat[batch_bboxnum_sum_list[b]:batch_bboxnum_sum_list[b+1],:,:]
                layer_list = batch_nonleaf_node_layer_list[b]
                for layer in layer_list:
                    for fa, son in layer.items():
                        son_list = np.array(son) + batch_tree_bboxnum_sum_list[b]
                        son_feat = new_node_feat[son_list]
                        head_feat = \
                            np.array([son_feat[:, :, 0].min(0), son_feat[:, :, 1].min(0), son_feat[:, :, 2].max(0), son_feat[:, :, 3].max(0)]) # T,4
                        new_node_feat[fa + batch_tree_bboxnum_sum_list[b]] = np.transpose(head_feat)
                        temporal_nonleaf_node_roi.append(new_node_feat[fa + batch_tree_bboxnum_sum_list[b]])
            temporal_nonleaf_node_roi = np.stack(temporal_nonleaf_node_roi, axis=0)
        elif feat_type == 'appearance' or feat_type == 'semantic':
            for b in range(batch_size):
                new_node_feat[batch_tree_bboxnum_sum_list[b]:batch_tree_bboxnum_sum_list[b]+batch_bboxnum_list[b],:,:] = \
                    temporal_bbox_feat[batch_bboxnum_sum_list[b]:batch_bboxnum_sum_list[b+1],:,:]
                layer_list = batch_nonleaf_node_layer_list[b]
                for layer in layer_list:
                    for fa, son in layer.items():
                        son_list = np.array(son) + batch_tree_bboxnum_sum_list[b]
                        son_feat = new_node_feat[son_list]
                        head_feat = np.mean(son_feat, axis=0)    # T, K
                        new_node_feat[fa+batch_tree_bboxnum_sum_list[b]] = head_feat
                        temporal_nonleaf_node_roi.append(head_feat)
            temporal_nonleaf_node_roi = np.stack(temporal_nonleaf_node_roi, axis=0)
        else:
            raise Exception('No such feature type!')

        return temporal_nonleaf_node_roi

