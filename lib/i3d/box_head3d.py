import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils
from .resnet3d import BasicBlock3d
from .resnet3d import Bottleneck3d

class Tail_Layer_3D(nn.Module):
    def __init__(self, net, net_layer_names, roi_xform_func, spatial_scale, num_stages=3):
        super().__init__()
        self.dim_in = net.final_feat_dim
        self.layer_names = []
        for i, layer_name in enumerate(net_layer_names[num_stages:]):
            res_layer = getattr(net, layer_name)
            self.add_module(layer_name, res_layer)
            self.layer_names.append(layer_name)
        
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        
        sr = 2 # cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        if cfg.I3D_DC5:
            roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
            self.fc1 = nn.AdaptiveAvgPool2d((1, 1))
            self.fc2 = nn.Linear(self.dim_in, hidden_dim)
        else:
            roi_size = \
                (cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // sr + (1 if cfg.FAST_RCNN.ROI_XFORM_RESOLUTION % sr != 0 else 0))
            #self.fc1 = nn.Linear(self.dim_in * roi_size**2, hidden_dim)
            self.fc1 = nn.AdaptiveAvgPool2d((1, 1))
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            
        self._init_weights()    
        
    def _init_weights(self):
        if not cfg.I3D_DC5:
            mynn.init.XavierFill(self.fc1.weight)
            init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)
        
    def last_layer_forward(self, x):
        for i, layer_name in enumerate(self.layer_names):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        return x
        
    def map2vec(self, x, use_relu=True):
        if len(x.shape) <= 4:
            batch_size = x.size(0)
        else:
            batch_size = x.size(0) * x.size(1)
        
        if cfg.I3D_DC5:
            x = self.fc1(x.view(batch_size, x.shape[-3], x.shape[-2], x.shape[-1]))
            x = x.view(batch_size, -1)
            #x = F.relu(x, inplace=True)
        else:
            #x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
            x = self.fc1(x.view(batch_size, x.shape[-3], x.shape[-2], x.shape[-1]))
            x = x.view(batch_size, -1)
        if use_relu:
            x = F.relu(self.fc2(x), inplace=True)
        else:
            x = self.fc2(x)
        return x
    
    def forward(self, x, use_relu=True):
        if len(self.layer_names) > 0:
            x = x.permute(0, 2, 1, 3, 4).contiguous() # n, T, C, P, P -> n, C, T, p, p
            x = self.last_layer_forward(x) # n, C, T, P, P; print(y.shape) #torch.Size([25, 2048, 2, 4, 4])
            x = x.permute(0, 2, 1, 3, 4).contiguous() # n, C, T, P, P -> n, T, c, p, p ;print(y.shape) #torch.Size([25, 2, 2048, 4, 4])
        y = self.map2vec(x, use_relu=use_relu)
        return y
                 
class Box_Head_3D(nn.Module):
    def __init__(self, 
                 in_channels, 
                 roi_xform_func, 
                 spatial_scale):
        super().__init__()
            
        self.dim_in = in_channels
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        
        self.fc1 = nn.Linear(in_channels * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()    
        
    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def forward(self, x, rpn_ret, rois_name='rois', use_relu=True):
        y = self.roi2map(x, rpn_ret, rois_name=rois_name)
        y = self.map2vec(y, use_relu=use_relu)
        return y
        
    def roi2map(self, x, rpn_ret, rois_name='rois'):
        y = self.roi_xform(
            x, rpn_ret,
            blob_rois=rois_name,
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        return y
        
    def map2vec(self, x, use_relu=True):
        if len(x.shape) <= 4:
            batch_size = x.size(0)
        else:
            batch_size = x.size(0) * x.size(1)
            
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        if use_relu:
            x = F.relu(self.fc2(x), inplace=True)
        else:
            x = self.fc2(x)
        return x
