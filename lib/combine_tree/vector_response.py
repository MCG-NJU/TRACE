import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import combine_tree.utils_t as utils_t

class vector_response(nn.Module):
    def __init__(self, x_dim_in, v_dim_in, dim_inflat):
        super(vector_response, self).__init__()
        self.inflat_fc = nn.Sequential(
                            nn.Linear(1, dim_inflat), 
                            nn.ReLU(inplace=True), 
                            nn.Linear(dim_inflat, dim_inflat),)
        self.q_trans = nn.Sequential(
                            nn.ReLU(), 
                            nn.Linear(x_dim_in, dim_inflat),)                  
        self.score_trans = nn.Sequential(
                            nn.ReLU(), 
                            nn.Linear(v_dim_in, v_dim_in),)
        
        self.sigmoid = nn.Sigmoid()                    
    def reset(self):
        self.inflat_fc.apply(utils_t.weight_init_mynn_Xavier)
        self.q_trans.apply(utils_t.weight_init_mynn_Xavier)
        self.score_trans.apply(utils_t.weight_init_mynn_Xavier)
        
    def forward(self, x_vec, v_vec, k_id=None):
        '''
            x_vec: N x K
            
            q_vec: N x H 
            k_vec: N x C
            v_vec: N x C
        '''
        q_vec = self.q_trans(x_vec)
        q_vec = q_vec.unsqueeze(1)
        
        k_vec = v_vec.unsqueeze(2)
        
        inflat_k_vec = self.inflat_fc(k_vec)
        if k_id is not None:
            inflat_k_vec = inflat_k_vec[k_id]
        score = torch.mul(inflat_k_vec, q_vec).sum(2)
        
        score = self.score_trans(score)
        
        if k_id is not None:
            ans = score + v_vec[k_id]
        else:
            ans = score + v_vec
        return ans