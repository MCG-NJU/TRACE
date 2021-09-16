import pickle
import json
import argparse
import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.lines as lines
from tqdm import tqdm

import _init_paths
from core.config import cfg

from datasets_rel.pytorch_misc import intersect_2d, argsort_desc

from functools import reduce
#from utils.boxes import bbox_overlaps
#from utils_rel.boxes_rel import boxes_union


from graphviz import Digraph, Graph
import seaborn as sns
#sns.set_theme()

parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
parser.add_argument(
    '--output_dir',
    help='output directory to save the testing results. If not provided, '
         'defaults to [args.load_ckpt|args.load_detectron]/../test.')
parser.add_argument(
    '--filename',
    help='Visualization file',
    default='rel_detection_range_12_15', type=str)    
args = parser.parse_args()
strfname = args.filename.split('_')
st,ed = strfname[-2], strfname[-1]
st,ed = int(st), int(ed)
def visualize_feature_map(feat_map, save_path, fname='feat_map'):
    fig = plt.figure()
    plt.imshow(feat_map)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, fname+'.png'))
    plt.close(fig)

def vis_box(img, box, save_path, fname='obj_box'):
    fig = plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.imshow(img)
    plt.axis('off')
    for i in range(len(box)):
        x, y, x1, y1 = box[i, 1:].astype(np.int)
        srect = plt.Rectangle((x,y),x1-x,y1-y, fill=False, edgecolor='b', linewidth=1)
        ax.add_patch(srect)
        #ax.text(x, y,
        #    color='white',
        #    bbox=dict(facecolor='orange', alpha=0.5, pad=0, edgecolor='none'))
    plt.savefig(os.path.join(save_path, fname+'.png'))
    plt.close(fig)
        
    
with open(os.path.join(args.output_dir, args.filename+'.pkl'), 'rb') as f:
    ret = pickle.load(f)
    f.close()

for ii, return_dict2 in enumerate(ret):
    if return_dict2 is None: continue
    blob_conv = return_dict2['blob_conv']
    feat_map = return_dict2['feat_map']  
    temporal_blob_conv_prd = return_dict2['temporal_blob_conv_prd']
    batch_A = return_dict2['batch_A']
    batch_non_leaf_bbox = return_dict2['batch_non_leaf_bbox']
    spatio_feat1 = return_dict2['spatio_feat1']
    spatio_feat2 = return_dict2['spatio_feat2']
    
    if batch_non_leaf_bbox is not None:
        batch_non_leaf_bbox = batch_non_leaf_bbox.data.cpu().numpy()
    if return_dict2['roi'] is not None:
        roi = return_dict2['roi'].data.cpu().numpy()
    else:
        roi = None
    im = return_dict2['im']
    
    save_dir = os.path.join(args.output_dir, str(ii+st))
    if not os.path.exists(os.path.join(args.output_dir, str(ii+st))):
        os.mkdir(save_dir)
    else:
        continue
        
    all_frames = return_dict2['all_frames']
    
    if isinstance(blob_conv, list):
        for i,v in enumerate(blob_conv):
            visualize_feature_map(v, save_dir, fname='obj_feat_map'+str(i))
    else:
        visualize_feature_map(blob_conv, save_dir, fname='obj_feat_map')
    im = im[:,:,::-1]
    visualize_feature_map(im, save_dir, fname='origin')
    
    
    if all_frames is not None:
        all_frames = all_frames.squeeze(0)
        channel_swap = (0, 2, 3, 1)
        all_frames = all_frames.transpose(channel_swap)
        frame_sampled = all_frames[0] + cfg.PIXEL_MEANS
        frame_sampled = frame_sampled.astype(np.int)
        frame_sampled = frame_sampled[:,:,::-1]
        visualize_feature_map(frame_sampled, save_dir, fname='frame_sampled')
    
    visualize_feature_map(feat_map, save_dir)
    
    if temporal_blob_conv_prd is not None:
        for i in range(len(temporal_blob_conv_prd)):
            visualize_feature_map(temporal_blob_conv_prd[i], 
                            save_dir, fname='temporal_feat_map'+str(i))
    if roi is not None:
        vis_box(im, roi, save_dir)
    
    if batch_A is not None:
        rid_f2s = set()
        dot = Graph(filename=('tree'))
        dot.body.append('size="16,16"')
        #dot.body.append('rankdir="LR"')
        son, fa = np.where(batch_A > 0)
        for i in range(len(batch_A)):
            dot.node(str(i), str(i), color='black')
        for i in range(len(fa)):
            if (son[i], fa[i]) in rid_f2s or (son[i], fa[i]) in rid_f2s: continue
            dot.edge(str(son[i]), str(fa[i]), color='black')
            rid_f2s.add((son[i], fa[i]))
            
        dot.render(os.path.join(save_dir, 'tree'), cleanup=True)
        
        for i in range(len(batch_A)):
            if i < len(roi):
                x, y, x1, y1 = roi[i, 1:].astype(np.int)
            else:
                x, y, x1, y1 = batch_non_leaf_bbox[i-len(roi), 1:].astype(np.int)
            subim = im[y:y1, x:x1, :]
            if not os.path.exists(os.path.join(save_dir, 'subim')):
                os.mkdir(os.path.join(save_dir, 'subim'))
            visualize_feature_map(subim, 
                    os.path.join(save_dir, 'subim'), str(i))
    
    if not os.path.exists(os.path.join(save_dir, 'subject_vis_branch')):
        os.mkdir(os.path.join(save_dir, 'subject_vis_branch'))
    for i in range(len(spatio_feat1)):
        print(spatio_feat1[i])
        print(spatio_feat2[i])
        print()
        #visualize_feature_map(spatio_feat1[i], os.path.join(save_dir, 'subject_vis_branch'), str(i))
    
    #if not os.path.exists(os.path.join(save_dir, 'object_vis_branch')):
    #    os.mkdir(os.path.join(save_dir, 'object_vis_branch'))
    #for i in range(len(spatio_feat2)):
    #    visualize_feature_map(spatio_feat2[i], os.path.join(save_dir, 'object_vis_branch'), str(i))