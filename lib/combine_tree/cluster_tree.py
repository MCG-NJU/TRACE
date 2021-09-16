import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import random
import time
from torch.autograd import Variable

import combine_tree.utils_t as utils_t

from core.config import cfg
import utils.fpn as fpn_utils
from model.roi_layers import ROIPool, ROIAlign

def roi_area(id, roi):
    x = ((roi[id, 3]-roi[id, 1]+1)*(roi[id, 4]-roi[id, 2]+1)).item()
    #print('x:'+str(x))
    return x

def debug_calu_degree(batch_size, batch_nonleaf_node_layer_list, new_roi, batch_bboxnum_sum_list):
    ans = 0
    minans = 999
    non1avgn = 0
    non1avgans = 0
    n = 0
    avgans = 0
    h = 0
    non1avgans2 = 0
    
    flatten_rate = 0
    
    print(batch_size)
    print(batch_bboxnum_sum_list)
    print(new_roi.shape)
    
    for b in range(batch_size):
        a = batch_nonleaf_node_layer_list[b]
        h = max(h, len(a))
        for l in a:
            for k,v in l.items():
                if len(v) > 1:
                    non1avgans = non1avgans*non1avgn + len(v)
                    non1avgans2 = non1avgans2*non1avgn + len(v)**2
                    non1avgn = non1avgn + 1
                    non1avgans = non1avgans / non1avgn
                    non1avgans2 = non1avgans2 / non1avgn
                avgans = avgans*n + len(v)
                
                S = 0
                for j in v:
                    S = max(S, roi_area(batch_bboxnum_sum_list[b]+j,new_roi))
                flatten_rate = flatten_rate*n + roi_area(batch_bboxnum_sum_list[b]+k,new_roi) / (S+1e-5)
                n = n + 1
                avgans = avgans / n
                flatten_rate = flatten_rate / n
                ans = max(ans, len(v))
                minans = min(minans, len(v))
                
    print(ans)
    print(non1avgans)
    print(math.sqrt(non1avgans2 - (non1avgans**2)))
    print(avgans)
    print(minans)
    print(h)
    print(flatten_rate)
    

class tree_message_passing2D(nn.Module):
    def __init__(self, feat_vec_dim, hidden_dim, mode='tree-gru', low_dim_embed_dim=None):
        super(tree_message_passing2D, self).__init__()
        self.hidden_dim = hidden_dim
        self.feat_vec_dim = feat_vec_dim
        
        self.low_dim_embed_dim = None
        self.sqrt_low_dim_embed_dim = None
        self.low_dim_embed_head = None
        self.low_dim_embed_son = None
        self.val_embed = None
        self.reflect_back = None
        self.low_dim_embed_center = None
        self.low_dim_embed_neighbor = None
        self.val_embed_nodir = None
        self.reflect_back_nodir = None
        
        self.Wh = None
        self.Wz = None
        self.Wr = None
        self.Uh = None
        self.Uz = None
        self.Ur = None
        self.Whd = None
        self.Wzd = None
        self.Wrd = None
        self.Uhd = None
        self.Uzd = None
        self.Urd = None
        self.FFN = None
        
        
        self.mode = mode
        
        self.Wh = nn.Sequential(
                    nn.Linear(feat_vec_dim, hidden_dim, bias=False),)
        self.Wz = nn.Sequential(
                    nn.Linear(feat_vec_dim, hidden_dim, bias=False),)
        self.Wr = nn.Sequential(
                    nn.Linear(feat_vec_dim, hidden_dim, bias=False),)            
        self.Uh = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias=False),)
        self.Uz = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias=True),)
        self.Ur = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias=True),)
        
        self.Whd = nn.Sequential(
                    nn.Linear(feat_vec_dim, hidden_dim, bias=False),)
        self.Wzd = nn.Sequential(
                    nn.Linear(feat_vec_dim, hidden_dim, bias=False),)
        self.Wrd = nn.Sequential(
                    nn.Linear(feat_vec_dim, hidden_dim, bias=False),)  
        self.Uhd = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias=False),)
        self.Uzd = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias=True),)
        self.Urd = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias=True),)
        
        self.FFN1 = nn.Sequential(
                        nn.Linear(hidden_dim, feat_vec_dim, bias=True),)
        self.FFN2 = nn.Sequential(
                        nn.Linear(hidden_dim, feat_vec_dim, bias=True),)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.reset()
        
    def reset(self):
        self.Wh.apply(utils_t.weights_orthogonal_init)
        self.Wz.apply(utils_t.weights_orthogonal_init)
        self.Wr.apply(utils_t.weights_orthogonal_init)
        self.Uh.apply(utils_t.weights_orthogonal_init)
        self.Uz.apply(utils_t.weights_orthogonal_init)
        self.Ur.apply(utils_t.weights_orthogonal_init)
        
        self.Whd.apply(utils_t.weights_orthogonal_init)
        self.Wzd.apply(utils_t.weights_orthogonal_init)
        self.Wrd.apply(utils_t.weights_orthogonal_init)
        self.Uhd.apply(utils_t.weights_orthogonal_init)
        self.Uzd.apply(utils_t.weights_orthogonal_init)
        self.Urd.apply(utils_t.weights_orthogonal_init)
        
        self.FFN1.apply(utils_t.weight_init_mynn_Xavier)
        self.FFN2.apply(utils_t.weight_init_mynn_Xavier)
        
    def forward(self, roi, feat_vec, batch_size, batch_nonleaf_node_layer_list, batch_tree_bboxnum_list, batch_tree_bboxnum_sum_list, batch_leaf_bboxnum_list, batch_A, batch_A_T, batch_link):
        out = self.forward_v1(roi, feat_vec, batch_size, batch_nonleaf_node_layer_list, 
                    batch_tree_bboxnum_list, batch_tree_bboxnum_sum_list, batch_leaf_bboxnum_list, 
                    batch_A, batch_A_T, batch_link)
        return out
    
    def forward_v1(self, roi, feat_vec, batch_size, batch_nonleaf_node_layer_list, batch_tree_bboxnum_list, batch_tree_bboxnum_sum_list, batch_leaf_bboxnum_list, batch_A, batch_A_T, batch_link):

        h = torch.FloatTensor(batch_tree_bboxnum_sum_list[batch_size], \
            self.hidden_dim).type_as(feat_vec).to(device=feat_vec.device)
        h.fill_(0.)
        
        cross_batch_nonleaf_node_layer_list, cross_batch_fa_set_list, \
        cross_batch_son_set_list, max_layer_num = \
            self.get_cross_batch_nonleaf_node_layer_list(batch_nonleaf_node_layer_list, \
                batch_size, batch_tree_bboxnum_sum_list)
        
        cross_batch_A_T = \
            self.get_cross_batch_adjmatrix(batch_size, batch_tree_bboxnum_sum_list, batch_A_T, h)
        
        
        h[cross_batch_son_set_list[0]] = \
            self.tree_gru_bottom_up(feat_vec[cross_batch_son_set_list[0]], is_leaf=True)
        
        
        for layer_id in range(max_layer_num):
            h[cross_batch_fa_set_list[layer_id]] = self.tree_gru_bottom_up(
                feat_vec[cross_batch_fa_set_list[layer_id]], 
                h[cross_batch_son_set_list[layer_id]], 
                is_leaf=False, 
                curr_layer_A_T=cross_batch_A_T[cross_batch_fa_set_list[layer_id]][:, cross_batch_son_set_list[layer_id]])

        hd = torch.FloatTensor(batch_tree_bboxnum_sum_list[batch_size], \
            self.hidden_dim).type_as(feat_vec).to(device=feat_vec.device)
        hd.fill_(0.)
        hd[cross_batch_fa_set_list[max_layer_num-1]] = h[cross_batch_fa_set_list[max_layer_num-1]]
        
        cross_batch_A = self.get_cross_batch_adjmatrix(batch_size, batch_tree_bboxnum_sum_list, batch_A, hd)
        for layer_id in range(max_layer_num-1, -1, -1):
            hd[cross_batch_son_set_list[layer_id]] = \
                    self.tree_gru_top_down(feat_vec[cross_batch_son_set_list[layer_id]], 
                                    hd[cross_batch_fa_set_list[layer_id]], 
                                    cross_batch_A[cross_batch_son_set_list[layer_id]][:, cross_batch_fa_set_list[layer_id]], 
                                    msp_mode=self.mode)
        

        out = self.FFN1(h) + self.FFN2(hd)
        return out
    
    def get_cross_batch_adjmatrix(self, batch_size, batch_tree_bboxnum_sum_list, batch_adj, h):
        cross_batch_adjmatrix = \
            np.zeros((batch_tree_bboxnum_sum_list[batch_size], \
                batch_tree_bboxnum_sum_list[batch_size]), dtype=np.float32)

        for b in range(batch_size):
            A = batch_adj[b]
            cross_batch_adjmatrix[batch_tree_bboxnum_sum_list[b]:batch_tree_bboxnum_sum_list[b+1], batch_tree_bboxnum_sum_list[b]:batch_tree_bboxnum_sum_list[b+1]] = A
        ans = torch.from_numpy(cross_batch_adjmatrix).cuda(h.device)
        return ans
    
    def get_cross_batch_nonleaf_node_layer_list(self, batch_nonleaf_node_layer_list, batch_size, batch_tree_bboxnum_sum_list):
        """
            batch_nonleaf_node_layer_list: list. example: [[{1: [0]}], [{1: [0]}], [{3: [2], 4: [1, 0]}, {5: [3, 4]}]]
            return:
                cross_batch_nonleaf_node_layer_list: [{1:np.array([0]), 3:np.array([2]), 8:np.array([5,4]), 7:np.array([6])}, {9:np.array([7,8])}]. len(cross_batch_nonleaf_node_layer_list) = max_layer_num. 
                cross_batch_fa_set_list: [[1,3,8,7], [9]]
                cross_batch_son_set_list: [np.array(0,2,4,5,6), np.array(7,8)]
        """
        cross_batch_nonleaf_node_layer_list = []
        cross_batch_fa_set_list = []
        cross_batch_son_set_list = []
        for b in range(batch_size):
            layer_list = batch_nonleaf_node_layer_list[b]
            layer_id = 0
            for layer in layer_list:
                if len(cross_batch_nonleaf_node_layer_list) < layer_id+1:
                    cross_batch_nonleaf_node_layer_list.append({})
                    cross_batch_fa_set_list.append([])
                    cross_batch_son_set_list.append([])    
                for fa, son in layer.items():
                    cross_batch_fa = batch_tree_bboxnum_sum_list[b] + fa
                    cross_batch_son = batch_tree_bboxnum_sum_list[b] + np.array(son)
                    cross_batch_nonleaf_node_layer_list[layer_id][cross_batch_fa] = cross_batch_son
                    cross_batch_fa_set_list[layer_id].append(batch_tree_bboxnum_sum_list[b] + fa)
                    cross_batch_son_set_list[layer_id].append(cross_batch_son)
                layer_id = layer_id + 1

        max_layer_num = len(cross_batch_nonleaf_node_layer_list)
        for l_id in range(max_layer_num):
            cross_batch_son_set_list[l_id] = np.hstack(cross_batch_son_set_list[l_id])
            
        return cross_batch_nonleaf_node_layer_list, cross_batch_fa_set_list, cross_batch_son_set_list, max_layer_num
    
    
    def atten_message_passing(self, head_vec, son_vec):
        """
            head_vec: torch.tensor: size:(1, K)
            son_vec: torch.tensor: size:(SON_NUM, K)
            
            return:
                feat_vec: torch.tensor: size:(1, K)
        """
        v_son = self.val_embed(son_vec)
        q_head = self.low_dim_embed_head(head_vec)
        q_son = self.low_dim_embed_son(son_vec)
        atten_score = torch.matmul(q_son, q_head.t()) / self.sqrt_low_dim_embed_dim
        atten_score = F.softmax(atten_score, 0)
        feat_vec = head_vec + self.reflect_back((atten_score * v_son).sum(0, keepdim=True))
        return feat_vec
    
    def atten_message_passing_gcn(self, f_vec, A):
        q_c = self.low_dim_embed_center(f_vec)
        q_n = self.low_dim_embed_neighbor(f_vec)
        v = self.val_embed_nodir(f_vec)
        ans = []
        for i in range(len(A)):
            neigh = torch.where(A[i] > 0)[0]
            atten_score = torch.matmul(q_c[neigh], q_n[[i]].t()) / self.sqrt_low_dim_embed_dim
            atten_score = F.softmax(atten_score, 0)
            tmp_i_v = self.reflect_back_nodir((atten_score * v[neigh]).sum(0, keepdim=True))
            ans.append(tmp_i_v)
        ans = f_vec + torch.cat(ans, 0)
        return ans
    
    def tree_gru_bottom_up(self, head_input, h_son_vec=None, is_leaf=False, curr_layer_A_T=None):
        """
            head_input: torch.tensor: size:(fa_n, K)
            h_son_vec: torch.tensor: size:(all_node_NUM, K)
            curr_layer_A_T: torch.tensor: size:(fa_NUM, K)
            return:
                h: torch.tensor: size:(fa_n, K)
        """
        if is_leaf is True:
            z = self.sigmoid(self.Wz(head_input))
            r = self.sigmoid(self.Wr(head_input))
            h = torch.mul((1-z), self.tanh(self.Wh(head_input)))
        else:
            h_son_sum = torch.matmul(curr_layer_A_T, h_son_vec)
            
            z = self.sigmoid(self.Wz(head_input) + self.Uz(h_son_sum))    # (fa_n, K)
            
            
            part1 = self.Wr(head_input)
            part1 = part1.unsqueeze(1)
            
            part2 = self.Ur(h_son_vec)
            part2 = part2.unsqueeze(0)
            
            part_sum = self.sigmoid(part1 + part2)
            t_clAT = curr_layer_A_T.unsqueeze(2)
            r = torch.mul(t_clAT, part_sum)    # N_fa, N_son, K
            
            t_h_son_vec = h_son_vec.unsqueeze(0)
            t_fa_g = torch.mul(r, t_h_son_vec)
            fa_g = t_fa_g.sum(1)

            fa_g = self.Uh(fa_g)
            
            h = torch.mul(z, h_son_sum) + torch.mul((1-z), self.tanh(self.Wh(head_input) + fa_g))
        return h
        
    def tree_gru_top_down(self, son_input, h_fa_vec, curr_layer_A, msp_mode='tree'):
        """
            son_input: torch.tensor: size:(SON_NUM, K)
            h_fa_vec: torch.tensor: size:(all_node_NUM, K)
            
            return:
                h: torch.tensor: size:(SON_NUM, K)
        """
        if msp_mode.find('tree') >= 0:
            son, fa = torch.where(curr_layer_A > 0)
            z = self.sigmoid(self.Wzd(son_input) + self.Uzd(h_fa_vec[fa]))
            r = self.sigmoid(self.Wrd(son_input) + self.Urd(h_fa_vec[fa]))
            h = torch.mul(z, h_fa_vec[fa]) + torch.mul((1-z), self.tanh(self.Whd(son_input) + self.Uhd(r * h_fa_vec[fa])))
        else:
            h_fa_sum = torch.matmul(curr_layer_A, h_fa_vec)
            
            z = self.sigmoid(self.Wzd(son_input) + self.Uzd(h_fa_sum))
            part1 = self.Wrd(son_input)
            part1 = part1.unsqueeze(1)
            
            part2 = self.Urd(h_fa_vec)
            part2 = part2.unsqueeze(0)
            
            part_sum = self.sigmoid(part1 + part2)
            t_clAT = curr_layer_A.unsqueeze(2)
            r = torch.mul(t_clAT, part_sum)    # N_fa, N_son, K
            
            t_h_fa_vec = h_fa_vec.unsqueeze(0)
            t_fa_g = torch.mul(r, t_h_fa_vec)
            fa_g = t_fa_g.sum(1)

            fa_g = self.Uhd(fa_g)
            
            h = torch.mul(z, h_fa_sum) + torch.mul((1-z), self.tanh(self.Whd(son_input) + fa_g))
        return h    
        
        
class cluster_tree2D(nn.Module):
    def __init__(self, feat_map_shape, feat_combine, feat_vec_dim, feat_region_vec_dim, hidden_dim, Roialign, spatial_scale, group_coeffi=1, max_tree_height=7, pool_size=5, cluster_num_partition=2, msp_mode='tree-gru'):
        super(cluster_tree2D, self).__init__()
        self.cluster_num_partition = cluster_num_partition
        self.max_tree_height = max_tree_height
        
        self.pool_size = pool_size
        self.spatial_scale = spatial_scale
        self.ROIalign_region = self.roi_feature_transform
        
        C, H, W = feat_map_shape
        self.feat_vec_dim = feat_vec_dim
        self.feat_combine = feat_combine

        self.fc_r = nn.Sequential(
                    nn.Linear(C*pool_size*pool_size, feat_region_vec_dim, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(feat_region_vec_dim, feat_region_vec_dim, bias=True),)
        
        self.r2c = nn.Sequential(
                        nn.ReLU(inplace=True), 
                        nn.Linear(feat_region_vec_dim, feat_combine, bias=False),)
        self.o2c = nn.Sequential(
                        nn.ReLU(inplace=True), 
                        nn.Linear(feat_vec_dim, feat_combine, bias=False),)     
        
        self.group_coeffi = group_coeffi
        self.tree_message_passing_list = torch.nn.ModuleList()
        for i in range(group_coeffi):
            self.tree_message_passing_list.append(
                tree_message_passing2D(feat_combine // group_coeffi, \
                    hidden_dim // group_coeffi, mode=msp_mode))
        self.FFN = nn.Sequential(
                    nn.ReLU(inplace=True), 
                    nn.Linear(feat_combine, feat_combine, bias=True),)
        self.msp_mode = msp_mode
        self.reset()
        
    def reset(self):
        self.fc_r.apply(utils_t.weight_init_mynn_Xavier)
        self.FFN.apply(utils_t.weight_init_mynn_Xavier)
        
        self.r2c.apply(utils_t.weight_init_mynn_Xavier)
        self.o2c.apply(utils_t.weight_init_mynn_Xavier)
    
    def get_A_and_link(self, batch_size, batch_tree_bboxnum_list, batch_nonleaf_node_layer_list, device):
        """
            return:
                batch_A: [adj_1, adj_2,...], adj_i: 2D torch.tensor
                batch_A_T: [adj^T_1, adj^T_2,...], adj^T_i: 2D torch.tensor
                batch_link: [link_1, ...], link_i: long np.array
        """
        batch_A = []
        batch_A_T = []
        batch_link = []
        
        for b in range(batch_size):
            A_T = np.zeros((batch_tree_bboxnum_list[b], batch_tree_bboxnum_list[b]), dtype=np.float32)
            A = np.zeros((batch_tree_bboxnum_list[b], batch_tree_bboxnum_list[b]), dtype=np.float32)

            layer_list = batch_nonleaf_node_layer_list[b][::-1]
            link = (-1) * np.ones((batch_tree_bboxnum_list[b], len(layer_list)+1), dtype=np.int)
            link[batch_tree_bboxnum_list[b]-1, len(layer_list)] = batch_tree_bboxnum_list[b]-1
            j = len(layer_list)-1
            for layer in layer_list:
                for fa, son in layer.items():
                    A_T[fa, son] = 1.0
                    A[son, fa] = 1.0
                    link[son, j+1:] = link[fa, j+1:]
                    link[son, j] = son    
                j = j - 1    
            batch_A.append(A)
            batch_A_T.append(A_T)
            batch_link.append(link)
        
        return batch_A, batch_A_T, batch_link
        
    def forward(self, roi, feat_map, obj_feat_vec):
        """
            roi: torch.tensor: (batch_id, xmin, ymin, xmax, ymax, score). size:(bbox_num, 6)
            feat_map: torch.tensor: size:(batch_size, C, H, W)
        """
        batch_size = feat_map.shape[0]
        C = feat_map.shape[1]
        H = feat_map.shape[2]
        W = feat_map.shape[3]
        
        bbox_id, batch_bboxnum_list, batch_bboxnum_sum_list = self.get_bbox_id(roi, batch_size)
        
        batch_leaf_bbox = roi[:, 1:5].data.cpu().numpy()
        
        
        batch_nonleaf_node_layer_list, batch_tree_node_bbox = \
            self.node_cluster(batch_leaf_bbox, 
                                batch_size, 
                                batch_bboxnum_list, 
                                batch_bboxnum_sum_list, 
                                dist_type='euclid', 
                                feat_type='coordinate',
                                add_batch_bbox_feat=obj_feat_vec.data.cpu().numpy())
        batch_tree_node_bbox = \
            torch.from_numpy(batch_tree_node_bbox).type(torch.FloatTensor).to(device=roi.device)
        batch_tree_node_bbox.requires_grad = False
        
        tree_bbox_id, batch_tree_bboxnum_list, batch_tree_bboxnum_sum_list = \
            self.get_bbox_id(batch_tree_node_bbox, batch_size)
        
        batch_A, batch_A_T, batch_link = \
            self.get_A_and_link(batch_size, batch_tree_bboxnum_list, \
                batch_nonleaf_node_layer_list, roi.device)

        batch_non_leaf_bbox = []
        for b in range(batch_size):
            batch_non_leaf_bbox.append(
                batch_tree_node_bbox[batch_tree_bboxnum_sum_list[b]+batch_bboxnum_list[b] : batch_tree_bboxnum_sum_list[b+1], :])
        batch_non_leaf_bbox = torch.cat(batch_non_leaf_bbox, 0)
        pooled_region_features_map = self.ROIalign_region(feat_map, batch_non_leaf_bbox)
        pooled_region_features = self.fc_r(pooled_region_features_map.view(len(pooled_region_features_map), -1))
        
        r2c_feat = self.r2c(pooled_region_features)
        o2c_feat = self.o2c(obj_feat_vec)
        
        pooled_features = []
        s_non_leaf = 0
        for b in range(batch_size):
            pooled_features.append(
                o2c_feat[batch_bboxnum_sum_list[b] : batch_bboxnum_sum_list[b] + batch_bboxnum_list[b]])
            pooled_features.append(
                r2c_feat[s_non_leaf : s_non_leaf + batch_tree_bboxnum_list[b] - batch_bboxnum_list[b]])
            s_non_leaf = s_non_leaf + batch_tree_bboxnum_list[b] - batch_bboxnum_list[b]
        pooled_features = torch.cat(pooled_features, 0)
        
        aggerated_feat = []
        for i in range(self.group_coeffi):
            input_group_feat = \
                pooled_features[:, (self.feat_combine//self.group_coeffi)*i : (self.feat_combine//self.group_coeffi)*(i+1)]
            group_aggerated_feat = \
                self.tree_message_passing_list[i](batch_tree_node_bbox, \
                    input_group_feat, batch_size, batch_nonleaf_node_layer_list, \
                    batch_tree_bboxnum_list, batch_tree_bboxnum_sum_list, \
                    batch_bboxnum_list, batch_A, batch_A_T, batch_link)
            aggerated_feat.append(group_aggerated_feat)
        aggerated_feat = torch.cat(aggerated_feat, 1)
        pooled_features = pooled_features + self.FFN(aggerated_feat)
        
        return pooled_features, batch_nonleaf_node_layer_list, batch_tree_node_bbox, \
            batch_tree_bboxnum_list, batch_tree_bboxnum_sum_list, batch_bboxnum_list, \
            batch_bboxnum_sum_list, batch_A, batch_A_T, batch_link
    
    def node_cluster(self, batch_bbox_feat, batch_size, batch_leaf_node_num, batch_bboxnum_sum_list, dist_type='euclid', feat_type='coordinate', sbj_id=None, obj_id=None, add_batch_bbox_feat=None):
        """
            batch_bbox_feat: numpy.array: (batch_id, xmin, ymin, xmax, ymax)    # batch_bbox_feat[:, 1:5].data.cpu().numpy()
            batch_leaf_node_num: batch_bboxnum_list: list
            batch_bboxnum_sum_list: list: size: batch_size+1
            
            return:
                batch_nonleaf_node_layer_list: list. example: [[{1: [0]}], [{1: [0]}], [{3: [2], 4: [1, 0]}, {5: [3, 4]}]]
                batch_tree_feat: batch_tree_node_bbox: numpy.array: (batch_id, xmin, ymin, xmax, ymax). size:(bbox_num, 5)
        """
        max_tree_height = self.max_tree_height
        batch_nonleaf_node_layer_list = []
        batch_tree_feat = []
        for b in range(batch_size):
            rel_cnt = batch_leaf_node_num[b]
            nonleaf_node_layer_list = []
            cluster_num = int(math.ceil(batch_leaf_node_num[b]))
            if cluster_num <= 0:
                continue
            bbox_feat = batch_bbox_feat[batch_bboxnum_sum_list[b]:batch_bboxnum_sum_list[b+1], :].copy()
            feat2 = None
            if add_batch_bbox_feat is not None:
                feat2 = add_batch_bbox_feat[batch_bboxnum_sum_list[b]:batch_bboxnum_sum_list[b+1], :].copy()
            l = 0
            r = cluster_num-1
            for layer_id in range(max_tree_height):
                cluster_num = int(math.ceil(float(cluster_num) / self.cluster_num_partition))
                new_node_dict = \
                    self.cluster_algorithm(bbox_feat, cluster_num, l, r, dist_type=dist_type, node_feat2=feat2)
                nonleaf_node_layer_list.append(new_node_dict)
                bbox_feat, feat2 = \
                    self.get_new_bbox_feat(bbox_feat, new_node_dict, feat_type=feat_type, feat2=feat2)
                l = r + 1
                r = r + cluster_num
                if cluster_num <= 0:
                    raise Exception('cluster_num begins to smaller than 1!')
                if cluster_num <= 1:
                    break
            
            if cluster_num > 1:
                new_node_dict = \
                    self.cluster_algorithm(bbox_feat, 1, l, r, dist_type=dist_type, node_feat2=feat2)
                nonleaf_node_layer_list.append(new_node_dict)
                bbox_feat, feat2 = \
                    self.get_new_bbox_feat(bbox_feat, new_node_dict, feat_type=feat_type, feat2=feat2)
            
                
            batch_nonleaf_node_layer_list.append(nonleaf_node_layer_list)
            bbox_feat = np.hstack((b * np.ones(len(bbox_feat), dtype=np.float32).reshape(-1, 1), bbox_feat))
            batch_tree_feat.append(bbox_feat)
        
        batch_tree_feat = np.vstack(batch_tree_feat)
        return batch_nonleaf_node_layer_list, batch_tree_feat
        
    def cluster_algorithm(self, bbox_feat, cluster_num, l, r, dist_type='euclid', node_feat2=None):
        """
            bbox_feat: numpy.array: (xmin, ymin, xmax, ymax)
            
            return:
                new_node: dict. example: {9: [5, 4], 10: [6], 11: [7, 8], 12: [2, 3, 1]}
        """
        if r < l:
            raise Exception('r < l!')
        if r-l+1 < cluster_num:
            raise Exception('cluster_num too large!')
        id = np.arange(l, r+1)
        
        id = self.get_cluster(id, bbox_feat, cluster_num, partitial=2, \
                cluster_score_dist_type='euclid', node_feat2=node_feat2)
        
        cluster_id = id[:cluster_num]
        combine_id = id[cluster_num:]
        combine_flat, cluster_flat = np.meshgrid(combine_id, cluster_id)
        combine_flat = combine_flat.reshape(-1)
        cluster_flat = cluster_flat.reshape(-1)
        distance = self.get_distance(bbox_feat[cluster_flat], bbox_feat[combine_flat], dist_type=dist_type)
        if node_feat2 is not None:
            distance = distance + \
                self.get_distance(node_feat2[cluster_flat], node_feat2[combine_flat], dist_type='cosine')
        distance = distance.reshape(cluster_num, -1)
        depend_id = np.argmin(distance, axis=0)
        
        new_node = {}
        for i in range(cluster_num):
            new_node[r+1+i] = [cluster_id[i]]
        combine_len = r-l+1 - cluster_num
        for i in range(combine_len):
            new_node[r+1+depend_id[i]].append(combine_id[i])
        
        return new_node
    
    def get_cluster(self, id, bbox_feat, cluster_num, partitial=2, cluster_score_dist_type='euclid', node_feat2=None):
        X, Y = np.meshgrid(id, id, indexing='ij')
        distance = self.get_distance(bbox_feat[X], bbox_feat[Y], dist_type=cluster_score_dist_type)
        if node_feat2 is not None:
            distance = distance + self.get_distance(node_feat2[X], node_feat2[Y], dist_type='cosine') 
        distance = distance.reshape(len(id), -1)
        score = -np.sum(np.exp(-distance), axis=1)  
        id_id = np.argsort(score)
        id = id[id_id]
        c_id = [i for i in range(0, len(id), 2)]
        not_c_id = [i for i in range(1, len(id), 2)]
        id = np.hstack((id[c_id], id[not_c_id]))
        return id
    
    def get_distance(self, x, y, dist_type='euclid'):
        if dist_type == 'euclid':
            return ((x - y)**2).sum(-1)
        elif dist_type == 'cosine':
            return 0.5 * (x*y).mean(-1)
        else:
            raise Exception('No such distance!')
    
    def get_new_bbox_feat(self, ori_bbox_feat, new_node_dict, feat_type='coordinate', feat2=None):
        """
            ori_bbox_feat: numpy.array: (xmin, ymin, xmax, ymax)
        """
        if feat_type == 'coordinate':
            new_node_feat = []
            for head, son_list in new_node_dict.items():
                son_feat = ori_bbox_feat[son_list]
                head_feat = np.array([son_feat[:, 0].min(), son_feat[:, 1].min(), son_feat[:, 2].max(), son_feat[:, 3].max()])
                new_node_feat.append(head_feat)
            new_node_feat = np.stack(new_node_feat, axis=0)
            new_bbox_feat = np.vstack((ori_bbox_feat, new_node_feat))
        elif feat_type == 'appearance' or feat_type == 'semantic':
            new_node_feat = []
            for head, son_list in new_node_dict.items():
                son_feat = ori_bbox_feat[son_list]
                head_feat = np.mean(son_feat, axis=0)
                new_node_feat.append(head_feat)
            new_node_feat = np.stack(new_node_feat, axis=0)
            new_bbox_feat = np.vstack((ori_bbox_feat, new_node_feat))
        else:
            raise Exception('No such feature type!')
        
        if feat2 is not None:
            new_node_feat2 = []
            for head, son_list in new_node_dict.items():
                son_feat = feat2[son_list]
                head_feat = np.mean(son_feat, axis=0)
                new_node_feat2.append(head_feat)
            new_node_feat2 = np.stack(new_node_feat2, axis=0)
            feat2 = np.vstack((feat2, new_node_feat2))
    
        return new_bbox_feat, feat2
        
    def get_bbox_id(self, roi, batch_size):
        """
            roi: torch.tensor
            return:
                bbox_id: torch.tensor: size: (len(roi), 1)
                batch_bboxnum_list: list: bbox number in each batch.
                batch_bboxnum_sum_list: list: accumulated bbox number in batches.
        """
        if isinstance(roi, torch.Tensor):
            bbox_id = []
            accu_bbox_num = 0
            batch_bboxnum_list = []
            batch_bboxnum_sum_list = [0]
            for i in range(batch_size):
                batch_bbox_num = int((roi[:, 0] <= i).sum().item() - accu_bbox_num)
                bbox_id.append(torch.arange(0, batch_bbox_num))
                accu_bbox_num += batch_bbox_num
                batch_bboxnum_list.append(batch_bbox_num)
                batch_bboxnum_sum_list.append(accu_bbox_num)
                
            bbox_id = torch.cat(bbox_id, 0).view(-1, 1).to(device=roi.device)
            bbox_id.requires_grad = False
        else:
            bbox_id = []
            accu_bbox_num = 0
            batch_bboxnum_list = []
            batch_bboxnum_sum_list = [0]
            for i in range(batch_size):
                batch_bbox_num = int((roi[:, 0] <= i).sum() - accu_bbox_num)
                bbox_id.append(np.arange(0, batch_bbox_num))
                accu_bbox_num += batch_bbox_num
                batch_bboxnum_list.append(batch_bbox_num)
                batch_bboxnum_sum_list.append(accu_bbox_num)
                
            bbox_id = np.concatenate(bbox_id, 0).reshape(-1, 1)
            
        return bbox_id, batch_bboxnum_list, batch_bboxnum_sum_list
    
    def roi_feature_transform(self, im, rpn_ret, blob_rois='rois',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """

        if isinstance(im, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = im[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(im) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = im[k_max - lvl]  # im is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois].astype(np.float32))).cuda(device_id)
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
            device_id = im.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            xform_out = ROIAlign(
                (resolution, resolution), spatial_scale, sampling_ratio)(im, rois)

        return xform_out
    
    def _add_rel_multilevel_rois(self, blobs):
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
        _distribute_rois_over_fpn_levels(['rois'])
        
