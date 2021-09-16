# Adapted by Ji Zhang, 2019
# from Detectron.pytorch/lib/core/test_engine.py
# Original license text below
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

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
from numpy import linalg as la
import os
import yaml
import json
from six.moves import cPickle as pickle

import torch
import torch.nn.functional as F
import nn as mynn
from torch.autograd import Variable

from core.config import cfg
from core.test_rel import im_detect_rels
from datasets_rel import task_evaluation_sg as task_evaluation_sg
from datasets_rel import task_evaluation_vg_and_vrd as task_evaluation_vg_and_vrd
from datasets_rel.json_dataset_rel import JsonDatasetRel
from modeling_rel import model_builder_rel
from utils.detectron_weight_helper import load_detectron_weight
import utils.env as envu
import utils_rel.net_rel as net_utils_rel
import utils_rel.subprocess_rel as subprocess_utils
import utils.vis as vis_utils
from utils.io import save_object
from utils.timer import Timer
import utils.blob as blob_utils

from combine_tree.line_order_frame_pipe import line_order_frame_pipe as LineOrderFramePipe

logger = logging.getLogger(__name__)

#debug_flag = 0

def get_eval_functions():
    # Determine which parent or child function should handle inference
    # Generic case that handles all network types other than RPN-only nets
    # and RetinaNet
    child_func = test_net
    parent_func = test_net_on_dataset

    return parent_func, child_func


def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = cfg.TEST.DATASETS[index]
    proposal_file = None

    return dataset_name, proposal_file


def run_inference(
        args, ind_range=None,
        multi_gpu_testing=False, gpu_id=0,
        check_expected_results=False):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None

    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            all_results = []
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                output_dir = args.output_dir
                results = parent_func(
                    args,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    multi_gpu=multi_gpu_testing
                )
                all_results.append(results)

            return all_results
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = args.output_dir
            return child_func(
                args,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results = result_getter()

    return all_results


def test_net_on_dataset(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        multi_gpu=False,
        gpu_id=0):
    """Run inference on a dataset."""
    dataset = JsonDatasetRel(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb(gt=args.do_val))
        all_results = multi_gpu_test_net_on_dataset(
            args, dataset_name, proposal_file, num_images, output_dir
        )
    else:
        all_results = test_net(
            args, dataset_name, proposal_file, output_dir, gpu_id=gpu_id
        )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    
    logger.info('Starting evaluation now...')
    if dataset_name.find('vg') >= 0 or dataset_name.find('vrd') >= 0 or dataset_name.find('ag') >= 0:
        task_evaluation_vg_and_vrd.eval_rel_results(all_results, output_dir, args.topk, args.do_val)
    else:
        task_evaluation_sg.eval_rel_results(all_results, output_dir, args.topk, args.do_val, args.do_vis, args.do_special)
    
    return all_results


def multi_gpu_test_net_on_dataset(
        args, dataset_name, proposal_file, num_images, output_dir):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, args.test_net_file + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Pass the target dataset and proposal file (if any) via the command line
    opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]
    if proposal_file:
        opts += ['TEST.PROPOSAL_FILES', '("{}",)'.format(proposal_file)]
        
    if args.do_val:
        opts += ['--do_val']
    if args.do_vis:
        opts += ['--do_vis']
    if args.do_special:
        opts += ['--do_special']
    if args.use_gt_boxes:
        opts += ['--use_gt_boxes']
    if args.use_gt_labels:
        opts += ['--use_gt_labels']
    if args.get_frame_rois:
        opts += ['--get_frame_rois']
    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel(
        'rel_detection', num_images, binary, output_dir,
        args.load_ckpt, args.load_detectron, opts
    )

    # Collate the results from each subprocess
    all_results = []
    for det_data in outputs:
        all_results += det_data
    
    if args.use_gt_boxes:
        if args.use_gt_labels:
            det_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_prdcls.pkl')
        else:
            det_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_sgcls.pkl')
    else:
        det_file = os.path.join(args.output_dir, 'rel_detections.pkl')
    save_object(all_results, det_file)
    logger.info('Wrote rel_detections to: {}'.format(os.path.abspath(det_file)))

    return all_results


def test_net(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'

    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        dataset_name, proposal_file, ind_range, args.do_val
    )
    model = initialize_model_from_cfg(args, gpu_id=gpu_id)
    
    line_order_frame_pipe = None
    #if cfg.FPN.FPN_ON:
    #    line_order_frame_pipe = LineOrderFramePipe(tot_length=21, max_time=18)
    
    N_list = [3, 7, 11, 14, 15, 19, 23, 27]
    N_list = [i for i in range(0, 30)]
    SEG_STEP = 15
    
    num_images = len(roidb)
    all_results = [None for _ in range(num_images)]
    timers = defaultdict(Timer)
    for i, entry in enumerate(roidb):
        img_name = entry['image']
        if dataset_name.find('vidvrd') >= 0:
            cur_frame_id = int(img_name.split('/')[-1].split('.')[0])
            
            flg = False
            for N in N_list:
                if (cur_frame_id - N) % SEG_STEP == 0 and cur_frame_id >= N:
                    flg = True
            if not flg: continue
        
        if cfg.ENABLE_FRAME_PRE_PROCESSING:
            frames_rpn_ret = entry['pre_processed_frames_rpn_ret']
            temporal_roi = entry['pre_processed_temporal_roi']
            
        else:
            frames_rpn_ret = None
            temporal_roi = None
        
        
        im = cv2.imread(entry['image'])
    
        box_proposals = None
        
        
        
        timers['im_detect_rels'].tic()
        
        
        cur_image_abs_path = entry['image']
        
        if (len(cfg.TRAIN.DATASETS) > 0 and \
            cfg.TRAIN.DATASETS[0].find('vidvrd') >= 0) or \
            (len(cfg.TEST.DATASETS) > 0 and \
            cfg.TEST.DATASETS[0].find('vidvrd') >= 0):
            im_id_st = 0
        elif (len(cfg.TRAIN.DATASETS) > 0 and \
            cfg.TRAIN.DATASETS[0].find('ag') >= 0) or \
            (len(cfg.TEST.DATASETS) > 0 and \
            cfg.TEST.DATASETS[0].find('ag') >= 0):
            im_id_st = 1
        else:
            im_id_st = 1
        
        
        
        all_frames_blob, bf_cur_len, f_scale = get_frames_blob(cur_image_abs_path, im, 
                                        im_id_st=im_id_st, 
                                        line_order_frame_pipe=line_order_frame_pipe)
        
        if args.use_gt_boxes:
            im_results = im_detect_rels(model, im, all_frames_blob, 
                                        dataset_name, 
                                        box_proposals, 
                                        args.do_vis, 
                                        timers, 
                                        entry, 
                                        args.use_gt_labels, 
                                        frames_rpn_ret, 
                                        temporal_roi,
                                        bf_cur_len,
                                        f_scale)
        else:
            im_results = im_detect_rels(model, im, all_frames_blob, dataset_name, box_proposals, args.do_vis, timers, frames_rpn_ret=frames_rpn_ret, temporal_roi=temporal_roi, bf_cur_len=bf_cur_len,f_scale=f_scale)
        
        im_results.update(dict(image=entry['image']))
        # add gt
        if args.do_val:
            im_results.update(
                dict(gt_sbj_boxes=entry['sbj_gt_boxes'],
                     gt_sbj_labels=entry['sbj_gt_classes'],
                     gt_obj_boxes=entry['obj_gt_boxes'],
                     gt_obj_labels=entry['obj_gt_classes'],
                     gt_prd_labels=entry['prd_gt_classes']))
        
        all_results[i] = im_results
        
        #if cfg.TEST.GET_FRAME_ROIS:
        #    cur_image_abs_path = cur_image_abs_path.split('/')
        #    cur_image_abs_path = cur_image_abs_path[-2] + '/' + cur_image_abs_path[-1]
        #    contain_frames_pkl[cur_image_abs_path] = (im_results['pre_processed_frames_rpn_ret'], im_results['pre_processed_temporal_roi'])
        
        timers['im_detect_rels'].toc()
        

        if i % cfg.TEST.DISPLAY_FREQ == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (timers['im_detect_rels'].average_time)
            logger.info((
                'im_detect: range [{:d}, {:d}] of {:d}: '
                '{:d}/{:d} {:.3f}s (eta: {})').format(
                start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                start_ind + num_images, det_time, eta))

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        det_name = 'rel_detection_range_%s_%s.pkl' % tuple(ind_range)
    else:
        if args.use_gt_boxes:
            if args.use_gt_labels:
                det_name = 'rel_detections_gt_boxes_prdcls.pkl'
            else:
                det_name = 'rel_detections_gt_boxes_sgcls.pkl'
        else:
            det_name = 'rel_detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    save_object(all_results, det_file)
    #if cfg.TEST.GET_FRAME_ROIS:
    #    frame_rois_det_file = os.path.join(output_dir, 'frame_rois.pkl')
    #    save_object(contain_frames_pkl, frame_rois_det_file)
    logger.info('Wrote rel_detections to: {}'.format(os.path.abspath(det_file)))
    return all_results


def initialize_model_from_cfg(args, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = model_builder_rel.Generalized_RCNN()
    model.eval()

    if args.cuda:
        model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)

        net_utils_rel.load_ckpt_rel(model, checkpoint['model'])
        
        
    if args.load_detectron:
        logger.info("loading detectron weights %s", args.load_detectron)
        load_detectron_weight(model, args.load_detectron)

    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    return model

def optimizer_debug(m):
    global debug_flag
    for k, v in m.items():
        if debug_flag > 0:
            break
        if isinstance(v, dict):
            optimizer_debug(v)
        elif isinstance(v, torch.Tensor):
            if len(v.shape) >= 5:
                debug_flag = 1
                break
            if len(v.squeeze().shape) >= 1:
                print(v)

def get_roidb_and_dataset(dataset_name, proposal_file, ind_range, do_val=True):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDatasetRel(dataset_name)
    roidb = dataset.get_roidb(gt=do_val)

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images

    
    
def get_frames_blob(cur_image_abs_path, im0, im_id_st=1, line_order_frame_pipe=None):
    frame_full_name = cur_image_abs_path.split('/')[-1]
    frame_id = int(frame_full_name.split('.')[0])
    tot_video_path_list = cur_image_abs_path.split('/')
    video_path_list = tot_video_path_list[:-3]
    video_path = '/'
    for j in video_path_list:
        video_path = os.path.join(video_path, j)
    #video_path = os.path.join(video_path, 'all_frames')
    video_path = os.path.join(video_path, 'sampled_frames') 
    video_path = os.path.join(video_path, tot_video_path_list[-2])
    
    processed_frames = []
    saved_frames = []
    not_saved_frames = []
    saved_processed_frames = []
    start_f_id = frame_id - (cfg.HALF_NUMBER_OF_FRAMES + 1) * cfg.FRAMES_INTERVAL
    end_f_id = frame_id + (cfg.HALF_NUMBER_OF_FRAMES + 1) * cfg.FRAMES_INTERVAL
    
    process_frames_id = []
    for j in range(frame_id, start_f_id, -cfg.FRAMES_INTERVAL):
        if j < im_id_st:
            break
        process_frames_id.append(j)
    process_frames_id = process_frames_id[::-1]
    process_frames_id = process_frames_id[:-1]
    for j in range(frame_id, end_f_id, cfg.FRAMES_INTERVAL):
        process_frames_id.append(j)
    
    off_set_f = 0
    off_set_b = cfg.HALF_NUMBER_OF_FRAMES
    k = 0
    img_cnt = 0
    
    
    for ii, j in enumerate(process_frames_id):
        if j < im_id_st:
            continue
        
        frame_path = os.path.join(video_path, '{:06d}'.format(j)+'.jpg')
        if j == frame_id:
            off_set_f = k
            k = 0
            frame_path = cur_image_abs_path
            
        if os.path.exists(frame_path):
            im = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            #target_size = cfg.TEST.SCALE
            target_size = cfg.TEMPORAL_SCALES
            
            if line_order_frame_pipe is not None and line_order_frame_pipe.frame_in(frame_path):
                im = line_order_frame_pipe.get_frame(frame_path)
                saved_frames.append(img_cnt)
                saved_processed_frames.append(im)
            else:
                im, f_scale = blob_utils.prep_im_for_blob(
                    im, cfg.PIXEL_MEANS, [target_size], 1000)
                not_saved_frames.append(img_cnt)
                processed_frames.append(im[0])
                
            k = k + 1
            img_cnt = img_cnt + 1
        else:
            off_set_b = k - 1
            break
    
    
    st = cfg.HALF_NUMBER_OF_FRAMES - off_set_f
    ed = cfg.HALF_NUMBER_OF_FRAMES + off_set_b
    
    
    if cfg.FPN.REL_FPN_ON:
        frames_blob = blob_utils.im_list_to_blob(processed_frames)
    else:
        #frames_blob = np.stack(processed_frames)
        frames_blob = np.array(processed_frames, dtype=np.float32)
        channel_swap = (0, 3, 1, 2)
        frames_blob = frames_blob.transpose(channel_swap)
        
    
    if line_order_frame_pipe is not None:
        h, w = frames_blob.shape[2:]
        pframes_blob = np.zeros(
            (len(saved_frames)+len(not_saved_frames), 3, h, w), dtype=np.float32)
        pframes_blob[not_saved_frames] = frames_blob
        
        if len(saved_frames) > 0:
            for i, f_id in enumerate(saved_frames):
                pframes_blob[f_id] = saved_processed_frames[i]
        
        for i, not_saved_frame_path in enumerate(not_saved_frames):
            line_order_frame_pipe.saving_frame(not_saved_frame_path, frames_blob[i])
        frames_blob = pframes_blob
        
    
    pad_st = st
    pad_ed = max(0, 2*cfg.HALF_NUMBER_OF_FRAMES - ed)
    bf_cur_len = off_set_f
    got_frames = frames_blob
    all_frames_blob = np.expand_dims(got_frames, axis=0)
    return all_frames_blob, bf_cur_len, f_scale[0]