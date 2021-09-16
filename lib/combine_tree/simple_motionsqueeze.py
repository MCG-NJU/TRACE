import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import combine_tree.utils_t as utils_t

class simple_motionsqueeze(nn.Module):
    def __init__(self, C):
        super(simple_motionsqueeze, self).__init__()
        self.correspond_conv = nn.Sequential(
                    nn.ReLU(), 
                    nn.Conv3d(C, C//2, kernel_size=(1,1,1), stride=1, padding=(0,0,0)),)
        
        self.motion_conv = nn.Sequential(
                    nn.ReLU(), 
                    nn.Conv3d(3, 64, kernel_size=(1,5,5), stride=1, padding=(0,1,1)),
                    nn.ReLU(inplace=True), 
                    nn.Conv3d(64, 128, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
                    nn.ReLU(inplace=True), 
                    nn.Conv3d(128, C//2, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
                    nn.ReLU(inplace=True), 
                    nn.Conv3d(C//2, C, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),)
        
        self.W = nn.Sequential(
                    nn.ReLU(), 
                    nn.Linear(C, C),)
        
        self.avg_pool_motion = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avg_pool_appearance = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        
    def reset(self):
        self.correspond_conv.apply(utils_t.weight_init_mynn_conv_MSRAFill)
        self.motion_conv.apply(utils_t.weight_init_mynn_conv_MSRAFill)
        self.W.apply(utils_t.weight_init_mynn_Xavier)
    
    def forward(self, input_tem_map):
        '''
            input_tem_map: N, C, T, H, W
            
        '''
        y = self.avg_pool_appearance(input_tem_map)
        y = y.view(y.shape[0], -1)
        
        if input_tem_map.shape[2] > 1:
            sim_motion = self.sim_flow(input_tem_map)
            
            sim_motion = sim_motion.permute(0, 2, 1, 3, 4) # n, T, C, P, P ->  n, C, T, P, P
            sim_motion = self.motion_conv(sim_motion)
            motion_vec = self.avg_pool_motion(sim_motion)
            motion_vec = motion_vec.view(input_tem_map.shape[0], -1)
            
            z = self.W(y + motion_vec)
        else:
            z = self.W(y)
        
        return z
    
    def sim_flow(self, input_tem_map, eta=0.01, sig=5.):
        '''
            input_tem_map: N, C, T, H, W
            
        '''
        device_id = input_tem_map.get_device()
        
        tem_map = self.correspond_conv(input_tem_map)
        tem_map = tem_map.permute(0, 2, 1, 3, 4) # n, C, T, P, P -> n, T, C, P, P
        
        bh_map = tem_map[:, 1:]
        bf_map = tem_map[:, :-1]
        N, Tp, C, H, W = bh_map.shape
        bh_map = bh_map.view(N, Tp, C, H, W, 1, 1)
        bf_map = bf_map.view(N, Tp, C, 1, 1, H, W)
        score = torch.mul(bh_map, bf_map).sum(2) # N, Tp, H, W, H, W
        
        p_vec_x = torch.arange(0, H).float().cuda(device_id)
        p_vec_y = torch.arange(0, W).float().cuda(device_id)
        inflat_p_vec_x = p_vec_x.view(1, 1, H, 1, 1)
        inflat_p_vec_y = p_vec_y.view(1, 1, W, 1, 1)
        
        flat_score = score.view(N, Tp, H*W, H, W)
        max_score, score_p_inds = torch.max(flat_score, dim=2, keepdims=True)
        score_p_inds_x = score_p_inds // H
        score_p_inds_x = score_p_inds_x.float() # N, Tp, 1, H, W
        score_p_inds_y = score_p_inds % H
        score_p_inds_y = score_p_inds_y.float() # N, Tp, 1, H, W
        
        # N, Tp, H, H, W
        g_x = torch.exp((inflat_p_vec_x - score_p_inds_x) / (sig**2)) / (math.sqrt(2 * math.pi) * sig)
        # N, Tp, W, H, W
        g_y = torch.exp((inflat_p_vec_y - score_p_inds_y) / (sig**2)) / (math.sqrt(2 * math.pi) * sig)
        
        score_x = torch.mul(g_x.unsqueeze(3), score)
        score_x = score_x.view(N, Tp, H*W, H, W)
        coef_x = F.softmax(score_x / eta, dim=2)
        
        score_y = torch.mul(g_y.unsqueeze(2), score)
        score_y = score_y.view(N, Tp, H*W, H, W)
        coef_y = F.softmax(score_y / eta, dim=2)
        
        expd_px, expd_py = torch.meshgrid(p_vec_x, p_vec_y)
        expd_px = expd_px.reshape(-1)
        expd_py = expd_py.reshape(-1)
        
        expd_px = expd_px.view(1, 1, H*W, 1, 1)
        expd_py = expd_py.view(1, 1, H*W, 1, 1)
        
        d_x = torch.mul(expd_px, coef_x).sum(2, keepdims=True)
        d_y = torch.mul(expd_py, coef_y).sum(2, keepdims=True)
        
        ans = torch.cat([d_x, d_y, max_score], 2)
        return ans