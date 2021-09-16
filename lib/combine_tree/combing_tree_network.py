import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import random
import time
import os

from combine_tree.word_vectors import obj_edge_vectors
from combine_tree.cluster_tree3d import cluster_tree3d
from combine_tree.tracking import track_frames
from combine_tree.tracking import track_frames_from_featmap

class combing_tree_network(nn.Module):
    def __init__(self, feat_map_shape, feat_combine, feat_vec_dim, feat_region_vec_dim, hidden_dim, low_feat_vec_dim, n_relation, Roialign, spatial_scale, group_coeffi=1, max_tree_height=7, pool_size=5, segment_len=30, root='./', obj_class_file_path='data/objects.json', obj_embed_file_path='data/obj_embed', obj_embed_dim=300):
        super(combing_tree_network, self).__init__()
        C, H, W = feat_map_shape
        
        self.cluster_tree3d = cluster_tree3d(feat_map_shape, 
                                                feat_combine, 
                                                feat_vec_dim, 
                                                feat_region_vec_dim, 
                                                hidden_dim, 
                                                low_feat_vec_dim, 
                                                Roialign, 
                                                spatial_scale, 
                                                group_coeffi=group_coeffi, 
                                                max_tree_height=max_tree_height, 
                                                pool_size=pool_size)
        
        self.obj_sem_embed = nn.Linear(feat_vec_dim, feat_vec_dim, bias=True)
        self.obj_visual_embed = nn.Linear(C*pool_size*pool_size, feat_vec_dim, bias=True)
        

        self.obj_embed_dim = obj_embed_dim
        obj_cats = json.load(open(os.path.join(root, obj_class_file_path)))
        self.obj_class = tuple(['__background__'] + obj_cats)
        embed_vecs = obj_edge_vectors(self.obj_class, os.path.join(root, obj_embed_file_path), wv_dim=self.obj_embed_dim)
        n_obj_class = len(self.obj_class)
        self.embed_obj_det = nn.Linear(n_obj_class, self.obj_embed_dim, bias=False)
        self.embed_obj_det.weight.data = embed_vecs.clone().t()
        self.embed_s = nn.Linear(n_obj_class, self.obj_embed_dim, bias=False)
        self.embed_s.weight.data = embed_vecs.clone().t()
        self.embed_o = nn.Linear(n_obj_class, self.obj_embed_dim, bias=False)
        self.embed_o.weight.data = embed_vecs.clone().t()
        
        self.after_obj_det_embed = nn.Sequential(
                                    nn.ReLU(),
                                    nn.Linear(self.obj_embed_dim, feat_vec_dim))
        
        self.obj_det_model = None ### to-do
        
        self.obj_tracking = track_frames_from_featmap
        self.segment_len = segment_len
        
        self.restore_obj_feat = nn.Sequential(
                                    nn.ReLU(),
                                    nn.Linear(feat_combine, feat_vec_dim, bias=True))
        
        self.spatio_atten_s = nn.Sequential(
                                nn.ReLU(),
                                nn.Linear(feat_vec_dim, low_feat_vec_dim, bias=True))
        self.spatio_atten_o = nn.Sequential(
                                nn.ReLU(),
                                nn.Linear(feat_vec_dim, low_feat_vec_dim, bias=True))
        self.spatio_atten_map = nn.Conv2d(C, low_feat_vec_dim, 1, stride=1, padding=0, bias=True)
        
        self.atten_q_s = nn.Linear(low_feat_vec_dim, low_feat_vec_dim, bias=True)
        self.atten_q_o = nn.Linear(low_feat_vec_dim, low_feat_vec_dim, bias=True)
        self.atten_q_map = nn.Conv2d(low_feat_vec_dim, low_feat_vec_dim, 1, stride=1, padding=0, bias=True)
        
        self.sqrt_low_dim_embed_dim = math.sqrt(low_feat_vec_dim)
        
        self.spatio_infer = nn.Sequential(
                                nn.Conv2d(low_feat_vec_dim*3, low_feat_vec_dim, 1, stride=1, padding=0, bias=True),
                                nn.Flatten(),
                                nn.ReLU(inplace=True),
                                nn.Linear(low_feat_vec_dim*pool_size*pool_size, low_feat_vec_dim, bias=True),)
        
        self.sem_s = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(feat_vec_dim + self.obj_embed_dim, low_feat_vec_dim, bias=True))
        self.sem_o = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(feat_vec_dim + self.obj_embed_dim, low_feat_vec_dim, bias=True))
        self.sem_r = nn.Sequential(
                        nn.Linear(feat_combine, feat_vec_dim, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Linear(feat_vec_dim, low_feat_vec_dim, bias=True))
        self.sem_infer = nn.Sequential(
                            nn.Linear(low_feat_vec_dim*3, low_feat_vec_dim, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Linear(low_feat_vec_dim, low_feat_vec_dim, bias=True),)
        
        self.relation_infer = nn.Sequential(
                                nn.ReLU(),
                                nn.Linear(low_feat_vec_dim, n_relation, bias=True),
                                nn.Sigmoid(inplace=True),)
        
        self.obj_infer = nn.Sequential(
                            nn.ReLU(),
                            nn.Linear(feat_vec_dim, n_obj_class, bias=True),)
        
        self.relu = nn.ReLU(inplace=True)
        self.reset()
    
    def reset(self):
        utils_t.weight_init_kaiming_uniform(self.obj_sem_embed)
        utils_t.weight_init_kaiming_uniform(self.obj_visual_embed)
        
        self.after_obj_det_embed.apply(utils_t.weight_init_kaiming_uniform)
        
        self.restore_obj_feat.apply(utils_t.weight_init_kaiming_uniform)
        
        self.spatio_atten_s.apply(utils_t.weight_init_kaiming_uniform)
        self.spatio_atten_o.apply(utils_t.weight_init_kaiming_uniform)
        
        utils_t.weight_init_kaiming_uniform(self.spatio_atten_map)
        
        utils_t.weight_init_kaiming_uniform(self.atten_q_s)
        utils_t.weight_init_kaiming_uniform(self.atten_q_o)
        utils_t.weight_init_kaiming_uniform(self.atten_q_map)
        
        
        self.spatio_infer.apply(utils_t.weight_init_kaiming_uniform)
        self.sem_s.apply(utils_t.weight_init_kaiming_uniform)
        self.sem_o.apply(utils_t.weight_init_kaiming_uniform)
        self.sem_r.apply(utils_t.weight_init_kaiming_uniform)
        self.sem_infer.apply(utils_t.weight_init_kaiming_uniform)
        self.relation_infer.apply(utils_t.weight_init_kaiming_uniform)
        self.obj_infer.apply(utils_t.weight_init_kaiming_uniform)
        
    def forward(self, img, frames_path, img_list_f, img_list_b, temporal_imgs, sampled_id, bf_cur_len):
        """
            img：torch.tensor: size:(B, C, H, W)
            frames_path: str list
            img_list_f, img_list_b: torch.tensor list: [img_0, img_1, ...]. img_i: cv2.imread. (no sampled, for tracking)
            temporal_imgs: torch.tensor: size:(B, T, C, H, W). (sampled, for generating featmap)
            sampled_id: int list. (for selecting tracklets in temporal dimantion)
            bf_cur_len: int. (the number of sampled imgs before current img)
            
            return:
                
        """
        feat_map, roi, obj_det_vec, softmax_obj_vec, visual_vec, temporal_feat_map = self.object_detection_one_pic(img, temporal_imgs, bf_cur_len)
        
        obj_det_embed = self.embed_obj_det(softmax_obj_vec)
        obj_feat_vec = self.obj_sem_embed(obj_det_vec) + self.obj_visual_embed(visual_vec) + self.after_obj_det_embed(obj_det_embed)
        

        batch_size = feat_map.shape[0]
        temporal_roi = self.get_tracklet(img_list_f, img_list_b, roi, batch_size)
        
        pooled_features, pooled_region_features_map, batch_nonleaf_node_layer_list, batch_tree_bboxnum_list, batch_tree_bboxnum_sum_list, batch_bboxnum_list, batch_bboxnum_sum_list, batch_nonleaf_bboxnum, batch_nonleaf_bboxnum_sum, batch_A, batch_A_T, batch_link = self.cluster_tree3d(roi, feat_map, obj_feat_vec, temporal_roi, temporal_feat_map)
        
        obj_feature, region_feature = self.feat_divide(pooled_features, batch_bboxnum_list, batch_tree_bboxnum_sum_list, batch_size)
        
        obj_feature = obj_feat_vec + self.restore_obj_feat(obj_feature)
        
        spatio_feat, X_list, Y_list, lca_list = self.get_spatio_atten_feat(obj_feature, pooled_region_features_map, batch_bboxnum_list, batch_bboxnum_sum_list, batch_link, batch_size, batch_nonleaf_bboxnum_sum)
        sem_feat = self.get_sem_feat(obj_feature, region_feature, lca_list, X_list, Y_list, softmax_obj_vec, batch_size)
        
        spatio_feat = self.spatio_infer(spatio_feat)
        sem_feat = self.sem_infer(sem_feat)
        
        relation_binary_score = self.relation_infer(spatio_feat + sem_feat)
        obj_score = self.obj_infer(obj_feature)
        
        return obj_score, relation_binary_score
        
    def object_detection_one_pic(self, img, temporal_imgs, bf_cur_len):
        """
            img：torch.tensor: size:(B, C, H, W)
            temporal_imgs: torch.tensor: size:(B, T, C, H, W). (sampled, for generating featmap)
            bf_cur_len: int. (the number of sampled imgs before current img)
            
            return:
                feat_map
                roi: torch.tensor: (batch_id, xmin, ymin, xmax, ymax, class, score). size:(bbox_num, 6)     # rois should be sorted by batch_id
                obj_det_vec: size:(bbox_num, K)
                softmax_obj_vec: size:(bbox_num, obj_num)
                visual_vec: size:(bbox_num, C*pool_size*pool_size)
                temporal_feat_map
        """
        feat_map, roi, obj_det_vec, softmax_obj_vec, visual_vec = self.obj_det_model(img, is_generating_rois=True)
        
        imgs_wo_cur = torch.cat([temporal_imgs[:, 0:bf_cur_len], temporal_imgs[:, bf_cur_len+1:]], 1)
        imgs_wo_cur = imgs_wo_cur.view(-1, img.shape[1], img.shape[2], img.shape[3])
        
        temporal_feat_map = self.obj_det_model(imgs_wo_cur, is_generating_rois=False)
        temporal_feat_map = temporal_feat_map.view(feat_map.shape[0], -1, feat_map.shape[1], feat_map.shape[2], feat_map.shape[3])
        temporal_feat_map = torch.cat([temporal_feat_map[:, 0:bf_cur_len], feat_map.unsqueeze(1), temporal_feat_map[:, bf_cur_len:]], 1)
        
        return feat_map, roi, obj_det_vec, softmax_obj_vec, visual_vec, temporal_feat_map
        
    def get_tracklet(self, img_list_f, img_list_b, roi, batch_size):
        """
            roi: torch.tensor: (batch_id, xmin, ymin, xmax, ymax, class, score). size:(bbox_num, 6)
            
            return:
                track_list: temporal_roi: torch.tensor: (batch_id, xmin, ymin, xmax, ymax). size:(bbox_num, T, 5)
        """
        if len(img_list_f) <= 1 and len(img_list_b) <= 1:
            print('No need for tracking')
            return roi.unsqueeze(1)
        track_list = []
        for b in range(batch_size):
            cur_batch_roi_id = torch.where(roi[:, 0] == b)[0]
            cur_batch_roi = roi[cur_batch_roi_id, 1:5]
            cur_batch_roi[:, 2] = cur_batch_roi[:, 2] - cur_batch_roi[:, 0] + 1
            cur_batch_roi[:, 3] = cur_batch_roi[:, 3] - cur_batch_roi[:, 1] + 1
            
            tracklet = self.obj_tracking(img_list_f, cur_batch_roi, tracker_type='dsst')
            tracklet_forward = torch.tensor(tracklet, dtype=torch.float)    # bbox_num, segment_len, 4
            
            tracklet = self.obj_tracking(img_list_b, cur_batch_roi, tracker_type='dsst', is_reverse=True)
            tracklet_back = torch.tensor(tracklet, dtype=torch.float)
            tracklet_back = tracklet_back[:, range(tracklet_back.shape[1]-1, -1, -1), :]
            
            segment_len = len(img_list_f) + len(img_list_b) - 1
            id = b * torch.ones(len(tracklet), segment_len, 1, dtype=torch.float)
            tracklet = torch.cat([tracklet_back[:, :-1, :], tracklet_forward], 1)
            tracklet = torch.cat([id, tracklet], 2)
            
            track_list.append(tracklet)
        track_list = torch.cat(track_list, 0)
        return track_list
    
    def feat_divide(self, pooled_features, batch_bboxnum_list, batch_tree_bboxnum_sum_list, batch_size):
        obj_feature = []
        region_feature = []
        for b in range(batch_size):
            obj_feature.append(pooled_features[batch_tree_bboxnum_sum_list[b]:batch_tree_bboxnum_sum_list[b]+batch_bboxnum_list[b]])
            region_feature.append(pooled_features[batch_tree_bboxnum_sum_list[b]+batch_bboxnum_list[b]:batch_tree_bboxnum_sum_list[b+1]])
        obj_feature = torch.cat(obj_feature, 0)    
        region_feature = torch.cat(region_feature, 0)    
        return obj_feature, region_feature
    
    
    def get_spatio_atten_feat(self, obj_feature, pooled_region_features_map, batch_bboxnum_list, batch_bboxnum_sum_list, batch_link, batch_size, batch_nonleaf_bboxnum_sum, X_list, Y_list):
        """
            
            return:
                # ans: torch.tensor list. tensor size:(N, low_feat_vec_dim * 3, H, W)
                ans: torch.tensor. tensor size:(sum{i^2}, low_feat_vec_dim * 3, H, W)
                lca_list: long np.array list
        """
        ans = []
        lca_list = []
        
        atten_s = self.spatio_atten_s(obj_feature)
        atten_o = self.spatio_atten_o(obj_feature)
        atten_map = self.spatio_atten_map(pooled_region_features_map)
        
        atten_s_q = self.atten_q_s(atten_s)
        atten_o_q = self.atten_q_o(atten_o)
        atten_map_q = self.atten_q_map(atten_map)
        for b in range(batch_size):
            if len(X_list.shape) > 1 and X_list.shape[1] == 2:
                bid = np.where(X_list[:, 0]==b)[0]
                X = X_list[bid, 1:].reshape(-1)
                Y = Y_list[bid, 1:].reshape(-1)
            else :
                X = X_list.reshape(-1)
                Y = Y_list.reshape(-1)

            q_s = atten_s_q[X + batch_bboxnum_sum_list[b]]
            q_o = atten_o_q[Y + batch_bboxnum_sum_list[b]]
            v_s = atten_s[X + batch_bboxnum_sum_list[b]]
            v_o = atten_o[Y + batch_bboxnum_sum_list[b]]
            
            lca = self.get_lca(X, Y, batch_link[b])
            lca = lca - batch_bboxnum_list[b] + batch_nonleaf_bboxnum_sum[b]
            
            
            s_map = self.obj_map_attention(q_s, atten_map_q[lca], v_s)
            o_map = self.obj_map_attention(q_o, atten_map_q[lca], v_o)
            
            spatio_map = torch.cat([s_map, o_map, atten_map[lca]], 1)
            
            ans.append(spatio_map)
            lca_list.append(lca)
        
        ans = torch.cat(ans, 0)
        return ans, lca_list
            
            
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
        
    def get_sem_feat(self, obj_feature, region_feature, lca_list, X_list, Y_list, softmax_obj_vec, batch_size):
        ans = []
        e_s = self.embed_s(softmax_obj_vec)
        e_o = self.embed_o(softmax_obj_vec)
        low_dim_s_feat = self.sem_s(torch.cat([obj_feature, e_s], 1))
        low_dim_o_feat = self.sem_o(torch.cat([obj_feature, e_o], 1))
        low_dim_r_feat = self.sem_r(region_feature)
        for b in range(batch_size):
            if len(X_list.shape) > 1 and X_list.shape[1] == 2:
                bid = np.where(X_list[:, 0]==b)[0]
                X = X_list[bid, 1:].reshape(-1)
                Y = Y_list[bid, 1:].reshape(-1)
            else :
                X = X_list.reshape(-1)
                Y = Y_list.reshape(-1)
            #X = X_list[b]
            #Y = Y_list[b]
            lca = lca_list[b]
            sem_feat_vec = torch.cat([low_dim_s_feat[X], low_dim_r_feat[lca], low_dim_o_feat[Y]], 1)
            ans.append(sem_feat_vec)
            
        ans = torch.cat(ans, 0)
        return ans