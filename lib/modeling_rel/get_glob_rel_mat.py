import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import json

from core.config import cfg
from datasets_rel.dataset_catalog_rel import ANN_FN2
from datasets_rel.dataset_catalog_rel import DATASETS



def get_glob_rel_mat(num_prd_classes, is_multi_relation=True, f_path=None):
    rel_mat = None
    if not is_multi_relation:
        return None, None, None, None, None, None
    
    if f_path is None:
        if len(cfg.TRAIN.DATASETS) > 0:
            with open(DATASETS[cfg.TRAIN.DATASETS[0]][ANN_FN2]) as f:
                train_data = json.load(f)
        else:
            ds_keywords = cfg.TEST.DATASETS[0].split('_')
            if ds_keywords[-1] == 'val' or ds_keywords[-1] == 'test' or ds_keywords[-1] == 'all' or ds_keywords[-1] == 'train':
                ds_keywords = ds_keywords[:-1]
            elif ds_keywords[-2] == 'of':  # those "x_of_3" test json files
                ds_keywords = ds_keywords[:-3]
            ds_name = '_'.join(ds_keywords + ['train'])
            with open(DATASETS[ds_name][ANN_FN2]) as f:
                train_data = json.load(f)
    else:
        with open(f_path) as f:
            train_data = json.load(f)
            f.close()
    
    rel_mat = np.zeros((
            num_prd_classes, 
            num_prd_classes), dtype=np.float32)
    rel_mat_single = np.zeros(num_prd_classes, dtype=np.float32)
    for _, im_rels in train_data.items():
        all_trip = dict()
        for trip in im_rels:
            s = [trip['object']['category'], ]
            s = s + list(trip['object']['bbox'])
            o = [trip['subject']['category'], ]
            o = o + list(trip['subject']['bbox'])
            t = tuple(s + o)
            if t in all_trip:
                all_trip[t].append(trip['predicate'])
            else:
                all_trip[t] = []
                all_trip[t].append(trip['predicate'])
                
            rel_mat_single[trip['predicate']] += 1
        for i in range(num_prd_classes):
            for j in range(i+1, num_prd_classes):
                for t, trip in all_trip.items():
                    if i in trip and j in trip:
                        rel_mat[i, j] += 1
    
    rel_mat_cou = np.zeros((
            num_prd_classes, 
            num_prd_classes), dtype=np.float32)
    rel_mat_single = rel_mat_single.reshape(1, -1).repeat(num_prd_classes, axis=0)
    rel_mat_single_T = rel_mat_single.T
    
    idx, idy = np.where(rel_mat_single > rel_mat_single_T)
    rel_mat_cou[idx, idy] = rel_mat_single_T[idx, idy]
    idx, idy = np.where(rel_mat_single <= rel_mat_single_T)
    rel_mat_cou[idx, idy] = rel_mat_single[idx, idy]
    
    rel_mat = rel_mat / (rel_mat_cou + 1e-4)
    for i in range(0, num_prd_classes):
        for j in range(0, i+1):
            rel_mat[i, j] = -1.
    
    n_x, n_y = np.where((rel_mat < 0.25) & (rel_mat >= -1e-4) & (rel_mat_cou > 0))
    p_x, p_y = np.where((rel_mat > 0.75) & (rel_mat_cou > 0))
    return n_x, n_y, p_x, p_y, rel_mat, rel_mat_cou
    
    
if __name__ == '__main__':
    n_x, n_y, p_x, p_y, a, rel_mat_cou = get_glob_rel_mat(26, f_path = './data/ag/annotations/new_annotations_train.json')
    b = (a * 100).astype(np.int)
    print(b)