# Adapted by Ji Zhang in 2019
# From Detectron.pytorch/lib/core/test.py
# Original license text below
# --------------------------------------------------------
# Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
from six.moves import cPickle as pickle
import cv2
import logging
import numpy as np

from torch.autograd import Variable
import torch

from core.config import cfg
from utils.timer import Timer
import utils.blob as blob_utils
import utils.fpn as fpn_utils
import utils.image as image_utils

logger = logging.getLogger(__name__)


def im_detect_rels(model, im, all_frames, dataset_name, box_proposals, do_vis=False, timers=None, roidb=None, use_gt_labels=False, frames_rpn_ret=None, temporal_roi=None, bf_cur_len=None, f_scale=None):
    rel_results = im_get_det_rels(model, im, all_frames, dataset_name, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, box_proposals, do_vis, roidb, use_gt_labels, frames_rpn_ret, temporal_roi, bf_cur_len, f_scale)
    return rel_results


def im_get_det_rels(model, im, all_frames, dataset_name, target_scale, target_max_size, boxes=None, do_vis=False, roidb=None, use_gt_labels=False, frames_rpn_ret=None, temporal_roi=None, bf_cur_len=None, f_scale=None):
    """Prepare the bbox for testing"""

    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
        inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']), volatile=True)]
    else:
        inputs['data'] = [torch.from_numpy(inputs['data']).contiguous()]
        inputs['im_info'] = [torch.from_numpy(inputs['im_info']).contiguous()]
    if dataset_name is not None:
        inputs['dataset_name'] = [blob_utils.serialize(dataset_name)]
    
    inputs['do_vis'] = [do_vis]
    if roidb is not None:
        inputs['roidb'] = [roidb]
    if use_gt_labels:
        inputs['use_gt_labels'] = [use_gt_labels]
    
    if temporal_roi is None:
        inputs['pre_processed_temporal_roi'] = [None]
    else:
        inputs['pre_processed_temporal_roi'] = [temporal_roi]
    if frames_rpn_ret is None:
        inputs['pre_processed_frames_rpn_ret'] = [None]
    else:
        inputs['pre_processed_frames_rpn_ret'] = [frames_rpn_ret]
    
    #inputs['pre_processed_temporal_roi'] = [None]
    #inputs['pre_processed_frames_rpn_ret'] = [None]
    if bf_cur_len is None:
        inputs['bf_cur_len'] = [None]
    else:
        inputs['bf_cur_len'] = [bf_cur_len]
    if f_scale is None:
        inputs['f_scale'] = [None]
    else:
        inputs['f_scale'] = [f_scale]
    
    inputs['all_frames'] = [torch.from_numpy(all_frames).contiguous()]
    
    return_dict = model(**inputs)
    
    return_dict2 = {}
    
    if return_dict['sbj_rois'] is not None:
        sbj_boxes = return_dict['sbj_rois'].data.cpu().numpy()[:, 1:5] / im_scale
        sbj_labels = return_dict['sbj_labels'].data.cpu().numpy() - 1
        sbj_scores = return_dict['sbj_scores'].data.cpu().numpy()
        obj_boxes = return_dict['obj_rois'].data.cpu().numpy()[:, 1:5] / im_scale
        obj_labels = return_dict['obj_labels'].data.cpu().numpy() - 1
        obj_scores = return_dict['obj_scores'].data.cpu().numpy()
        prd_scores = return_dict['prd_scores'].data.cpu().numpy()
        if cfg.MODEL.USE_FREQ_BIAS:
            prd_scores_bias = return_dict['prd_scores_bias'].data.cpu().numpy()
        if cfg.MODEL.USE_SPATIAL_FEAT:
            prd_scores_spt = return_dict['prd_scores_spt'].data.cpu().numpy()
        if cfg.MODEL.ADD_SCORES_ALL:
            prd_scores_ttl = return_dict['prd_ttl_scores'].data.cpu().numpy()

        return_dict2 = dict(sbj_boxes=sbj_boxes,
                            sbj_labels=sbj_labels.astype(np.int32, copy=False),
                            sbj_scores=sbj_scores,
                            obj_boxes=obj_boxes,
                            obj_labels=obj_labels.astype(np.int32, copy=False),
                            obj_scores=obj_scores,
                            prd_scores=prd_scores)
        if cfg.MODEL.ADD_SCORES_ALL:
            return_dict2['prd_scores_ttl'] = prd_scores_ttl

        if cfg.MODEL.USE_FREQ_BIAS:
            return_dict2['prd_scores_bias'] = prd_scores_bias
        if cfg.MODEL.USE_SPATIAL_FEAT:
            return_dict2['prd_scores_spt'] = prd_scores_spt
        if do_vis:
            if isinstance(return_dict['blob_conv'], list):
                blob_conv = [b.data.cpu().numpy().squeeze() for b in return_dict['blob_conv']]
                blob_conv = [b.mean(axis=0) for b in blob_conv]
            else:
                blob_conv = return_dict['blob_conv'].data.cpu().numpy().squeeze()
                blob_conv = blob_conv.mean(axis=0)
            if isinstance(return_dict['feat_map'], list):
                feat_map = [b.data.cpu().numpy().squeeze() for b in return_dict['feat_map']]
                feat_map = [b.mean(axis=0) for b in feat_map]
            else:
                feat_map = return_dict['feat_map'].data.cpu().numpy().squeeze()
                feat_map = feat_map.mean(axis=0)
                
            if isinstance(return_dict['temporal_blob_conv_prd'], list):
                temporal_blob_conv_prd = [b.data.cpu().numpy().squeeze() for b in return_dict['temporal_blob_conv_prd']]
                temporal_blob_conv_prd = [b.mean(axis=1) for b in temporal_blob_conv_prd]
            elif return_dict['temporal_blob_conv_prd'] is not None:
                temporal_blob_conv_prd = return_dict['temporal_blob_conv_prd'].data.cpu().numpy().squeeze(0)
                temporal_blob_conv_prd = temporal_blob_conv_prd.mean(axis=1)
            else:
                temporal_blob_conv_prd = None
                
            if isinstance(return_dict['batch_A'], list):
                batch_A = return_dict['batch_A'][0]
            else:
                batch_A = return_dict['batch_A']
            if isinstance(return_dict['batch_non_leaf_bbox'], list):
                batch_non_leaf_bbox = return_dict['batch_non_leaf_bbox'][0]
            else:
                batch_non_leaf_bbox = return_dict['batch_non_leaf_bbox']    
            if isinstance(return_dict['roi'], list):
                roi = return_dict['roi'][0]
            else:
                roi = return_dict['roi']
            if 'spatio_feat1' in return_dict:
                spatio_feat1 = return_dict['spatio_feat1']
                spatio_feat1 = spatio_feat1.data.cpu().numpy()
                spatio_feat1 = spatio_feat1.mean(axis=1)
            if 'spatio_feat2' in return_dict:
                spatio_feat2 = return_dict['spatio_feat2']
                spatio_feat2 = spatio_feat2.data.cpu().numpy()
                spatio_feat2 = spatio_feat2.mean(axis=1)    
                
            return_dict2['blob_conv'] = blob_conv 
            return_dict2['feat_map'] = feat_map  
            return_dict2['temporal_blob_conv_prd'] = temporal_blob_conv_prd
            return_dict2['batch_A'] = batch_A
            return_dict2['batch_non_leaf_bbox'] = batch_non_leaf_bbox
            return_dict2['roi'] = roi
            return_dict2['im'] = im
            return_dict2['all_frames'] = all_frames
            return_dict2['spatio_feat2'] = spatio_feat2
            return_dict2['spatio_feat1'] = spatio_feat1
            
    else:
        return_dict2 = dict(sbj_boxes=None,
                            sbj_labels=None,
                            sbj_scores=None,
                            obj_boxes=None,
                            obj_labels=None,
                            obj_scores=None,
                            prd_scores=None)
    
    if return_dict['pre_processed_frames_rpn_ret'] is not None:
        return_dict2['pre_processed_frames_rpn_ret'] = return_dict['pre_processed_frames_rpn_ret']
        return_dict2['pre_processed_temporal_roi'] = return_dict['pre_processed_temporal_roi']
    else:
        return_dict2['pre_processed_frames_rpn_ret'] = None
        return_dict2['pre_processed_temporal_roi'] = None
        
    return return_dict2


def _get_rois_blob(im_rois, im_scale):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels


def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn_utils.map_rois_to_fpn_levels(blobs[name][:, 1:5], lvl_min, lvl_max)
    fpn_utils.add_multilevel_roi_blobs(
        blobs, name, blobs[name], lvls, lvl_min, lvl_max
    )


def _get_blobs(im, rois, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    
    if cfg.FPN.FPN_ON:
        blobs['data'], im_scale, blobs['im_info'] = \
            blob_utils.get_image_blob(im, target_scale, target_max_size)
    else:
        processed_im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_scale], target_max_size
        )
        if isinstance(processed_im, list):
            processed_im = np.stack(processed_im)
        else:
            processed_im = np.expand_dims(processed_im, axis=0)
        channel_swap = (0, 3, 1, 2)
        blobs['data'] = processed_im.transpose(channel_swap)    # B, H, W, C --> B, C, H, W
        im_shapes = processed_im.shape[2:]
        max_shape = np.array(im_shapes, dtype=np.int32)
        if cfg.FPN.FPN_ON:
            stride = float(cfg.FPN.COARSEST_STRIDE)
            max_shape[0] = int(np.ceil(max_shape[0] / stride) * stride)
            max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)
        padh, padw = max_shape[0] - im_shapes[0], max_shape[1] - im_shapes[1]
        if padh != 0 or padw != 0:
            blobs['data'] = np.pad(blobs['data'], ((0,0), (0,0), (0,padh), (0,padw)),'constant',constant_values=0)
        height, width = blobs['data'].shape[2], blobs['data'].shape[3]
        blobs['im_info'] = np.hstack((height, width, im_scale))[np.newaxis, :].astype(np.float32)
    
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale)
    return blobs, im_scale

    