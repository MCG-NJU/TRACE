"""
Written by Ji Zhang, 2019
Some functions are adapted from Rowan Zellers
Original source:
https://github.com/rowanz/neural-motifs/blob/master/lib/evaluation/sg_eval.py
"""

import os
import numpy as np
import logging
from six.moves import cPickle as pickle
import json
import csv
from tqdm import tqdm
import argparse

import cv2

import _init_paths

from core.config import cfg
from utils.boxes import bbox_overlaps
from utils_rel.boxes_rel import boxes_union

from datasets_rel.pytorch_misc import intersect_2d, argsort_desc
from vidvrd_utils.association import greedy_relational_association
from vidvrd_utils.common import voc_ap, viou

from collections import defaultdict


def get_belong_interval(f_id, seg_step=15, seg_len=30):
    y2 = f_id // seg_step
    y1 = max(0, y2 - 1)
    st1 = y1 * seg_step
    ed1 = y1 * seg_step + seg_len - 1
    st2 = y2 * seg_step
    ed2 = y2 * seg_step + seg_len - 1
    return st1, ed1, st2, ed2

def get_cur_full_frame_path(f_name):
    temp = f_name.split('/')[:-1]
    #ans = temp[0]
    #for i in temp[1:]:
    #    ans = ans + '/' + i
    ans = '.'
    flg = True
    for i in temp:
        if i != 'data' and flg: 
            continue
        else: 
            ans = ans + '/' + i
            flg = False
        
    return ans
    
def get_name_format_in_key_frame_set(f_name, st, ed):
    ans = get_cur_full_frame_path(f_name)
    ans = ans + '/' + '{:06d}'.format(st) + '_' + '{:06d}'.format(ed)
    return ans

def get_all_segment_interval_name(cur_full_frame_path, seg_step=15, seg_len=30):
    f_name_list = os.listdir(cur_full_frame_path)
    ans = []
    for i in f_name_list:
        t_f_id = int(i.split('.')[0])
        if t_f_id % seg_step == 0:
            st = t_f_id
            ed = st + seg_len - 1
            sufixx = '.' + i.split('.')[-1]
            ed_f_name = cur_full_frame_path + '/' + '{:06d}'.format(ed) + sufixx
            if os.path.exists(ed_f_name):
                sharp_dict_name = cur_full_frame_path + '/' + '{:06d}'.format(st) + '_' + '{:06d}'.format(ed)
                ans.append((sharp_dict_name, st, cur_full_frame_path))
    return ans

def judge_sharp_frame(key_frame_set, f_name, seg_step=15, seg_len=30):
    f_id = int(f_name.split('/')[-1].split('.')[0])
    st1, ed1, st2, ed2 = get_belong_interval(f_id, seg_step=seg_step, seg_len=seg_len)
    d1, d2 = f_id - st1, f_id - st2
    interval1_name = get_name_format_in_key_frame_set(f_name, st1, ed1)
    interval2_name = get_name_format_in_key_frame_set(f_name, st2, ed2)
    ans_N_list = []
    ans_w = []
    
    if interval1_name in key_frame_set and f_id in key_frame_set[interval1_name]:
        count_w1 = dict()
        for i in key_frame_set[interval1_name]:
            if i not in count_w1: count_w1[i] = 0.
            count_w1[i] += 1.
        
        ans_N_list.append(d1)
        ans_w.append(count_w1[f_id])
    if st1 != st2:
        if interval2_name in key_frame_set and f_id in key_frame_set[interval2_name]:
            count_w2 = dict()
            for i in key_frame_set[interval2_name]:
                if i not in count_w2: count_w2[i] = 0.
                count_w2[i] += 1.
        
            ans_N_list.append(d2)
            ans_w.append(count_w2[f_id])
    #print(interval1_name, key_frame_set[interval1_name], interval2_name, key_frame_set[interval2_name], ans_N_list)
    return ans_N_list, ans_w

def get_key_frame_set(full_video_path, prim_N_list, seg_step=15, seg_len=30, f_interval=3):
    key_frame_set = dict()
    interval_name_list = []
    
    video_name_list = os.listdir(full_video_path)
    for i in tqdm(video_name_list):
        video_path = os.path.join(full_video_path, i)
        cur_interval_name_list = \
            get_all_segment_interval_name(video_path, seg_step=seg_step, seg_len=seg_len)
        interval_name_list += cur_interval_name_list
    
    for j, st, cur_full_frame_path in tqdm(interval_name_list):
        ans_N_list = list()
        for i in prim_N_list:
            middle_id = st + i
            sharp_id = sharpest_frame_calibration(middle_id, \
                        cur_full_frame_path, f_interval=f_interval)
            ans_N_list.append(sharp_id)
        key_frame_set[j] = ans_N_list
    
    return key_frame_set



def sharpest_frame_calibration(middle_id, full_frame_path, f_interval=3):
    sufixx = '.png'
    if full_frame_path.find('sampled_frames') >= 0 or full_frame_path.find('all_frames') >= 0:
        sufixx = '.jpg'
    id_list = []
    val_list = []
    for i in range(middle_id - f_interval, middle_id + f_interval + 1):
        if i < 0: continue
        cur_f_path = os.path.join(full_frame_path, '{:06d}'.format(i)+sufixx)
        if os.path.exists(cur_f_path):
            img = cv2.imread(cur_f_path, 0)
            gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize = 3)
            abs_gray_lap = np.abs(gray_lap)
            mu = abs_gray_lap.mean()
            lap_var = ((abs_gray_lap - mu)**2).mean()
            val_list.append(lap_var)
            id_list.append(i)
    val_list = np.array(val_list, dtype=np.float)
    id_list = np.array(id_list, dtype=np.int)
    up_val_list = np.argsort(val_list)
    sharp_id = id_list[up_val_list[-1]]
    return sharp_id


def eval_detection_scores(gt_relations, pred_relations, viou_threshold):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    hit_scores = np.ones((len(pred_relations))) * -np.inf
    for pred_idx, pred_relation in enumerate(pred_relations):
        ov_max = -float('Inf')
        k_max = -1
        for gt_idx, gt_relation in enumerate(gt_relations):
            if not gt_detected[gt_idx]\
                    and tuple(pred_relation['triplet']) == tuple(gt_relation['triplet']):
                s_iou = viou(pred_relation['sub_traj'], pred_relation['duration'],
                        gt_relation['sub_traj'], gt_relation['duration'])
                o_iou = viou(pred_relation['obj_traj'], pred_relation['duration'],
                        gt_relation['obj_traj'], gt_relation['duration'])
                ov = min(s_iou, o_iou)
                if ov >= viou_threshold and ov > ov_max:
                    ov_max = ov
                    k_max = gt_idx
                    #print(pred_idx, gt_idx, ov)
        if k_max >= 0:
            hit_scores[pred_idx] = pred_relation['score']
            gt_detected[k_max] = True
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def eval_tagging_scores(gt_relations, pred_relations):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    # ignore trajectories
    gt_triplets = set(tuple(r['triplet']) for r in gt_relations)
    pred_triplets = []
    hit_scores = []
    for r in pred_relations:
        triplet = tuple(r['triplet'])
        if not triplet in pred_triplets:
            pred_triplets.append(triplet)
            hit_scores.append(r['score'])
    hit_scores = np.asarray(hit_scores)
    for i, t in enumerate(pred_triplets):
        if not t in gt_triplets:
            hit_scores[i] = -np.inf
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_triplets), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def evaluate(groundtruth, prediction, viou_threshold=0.5,
        det_nreturns=[50, 100], tag_nreturns=[1, 5, 10]):
    """ evaluate visual relation detection and visual 
    relation tagging.
    """
    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_relations = 0
    print('Computing average precision AP over {} videos...'.format(len(groundtruth)))
    for vid, gt_relations in groundtruth.items():
        if len(gt_relations)==0:
            continue
        tot_gt_relations += len(gt_relations)
        predict_relations = prediction.get(vid, [])
        # compute average precision and recalls in detection setting
        det_prec, det_rec, det_scores = eval_detection_scores(
                gt_relations, predict_relations, viou_threshold)
        video_ap[vid] = voc_ap(det_rec, det_prec)
        tp = np.isfinite(det_scores)
        for nre in det_nreturns:
            cut_off = min(nre, det_scores.size)
            tot_scores[nre].append(det_scores[:cut_off])
            tot_tp[nre].append(tp[:cut_off])
        # compute precisions in tagging setting
        tag_prec, _, _ = eval_tagging_scores(gt_relations, predict_relations)
        for nre in tag_nreturns:
            cut_off = min(nre, tag_prec.size)
            if cut_off > 0:
                prec_at_n[nre].append(tag_prec[cut_off - 1])
            else:
                prec_at_n[nre].append(0.)
    # calculate mean ap for detection
    mean_ap = np.mean(list(video_ap.values()))
    # calculate recall for detection
    rec_at_n = dict()
    for nre in det_nreturns:
        scores = np.concatenate(tot_scores[nre])
        tps = np.concatenate(tot_tp[nre])
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        rec_at_n[nre] = rec[-1]
    # calculate mean precision for tagging
    mprec_at_n = dict()
    for nre in tag_nreturns:
        mprec_at_n[nre] = np.mean(prec_at_n[nre])
    # print scores
    print('detection mean AP (used in challenge): {}'.format(mean_ap))
    print('detection recall@50: {}'.format(rec_at_n[50]))
    print('detection recall@100: {}'.format(rec_at_n[100]))
    print('tagging precision@1: {}'.format(mprec_at_n[1]))
    print('tagging precision@5: {}'.format(mprec_at_n[5]))
    print('tagging precision@10: {}'.format(mprec_at_n[10]))
    return mean_ap, rec_at_n, mprec_at_n

def get_relation_insts(vid, tet):
    with open(os.path.join('data/vidvrd/annotations', tet, vid)) as f:
        anno = json.load(f)
        f.close()
    sub_objs = dict()
    for so in anno['subject/objects']:
        sub_objs[so['tid']] = so['category']
    trajs = []
    for frame in anno['trajectories']:
        bboxes = dict()
        for bbox in frame:
            bboxes[bbox['tid']] = (bbox['bbox']['xmin'],
                                bbox['bbox']['ymin'],
                                bbox['bbox']['xmax'],
                                bbox['bbox']['ymax'])
        trajs.append(bboxes)
    relation_insts = []
    seg_relation_insts = dict()
    for anno_inst in anno['relation_instances']:
        inst = dict()
        inst['triplet'] = (sub_objs[anno_inst['subject_tid']],
                        anno_inst['predicate'],
                        sub_objs[anno_inst['object_tid']])
        inst['subject_tid'] = anno_inst['subject_tid']
        inst['object_tid'] = anno_inst['object_tid']
        inst['duration'] = (anno_inst['begin_fid'], anno_inst['end_fid'])
        inst['sub_traj'] = [bboxes[anno_inst['subject_tid']] for bboxes in
                trajs[inst['duration'][0]: inst['duration'][1]]]
        inst['obj_traj'] = [bboxes[anno_inst['object_tid']] for bboxes in
                trajs[inst['duration'][0]: inst['duration'][1]]]
        relation_insts.append(inst)
        
        for dur in range(0, anno_inst['end_fid']-anno_inst['begin_fid'], 15):
            if anno_inst['begin_fid']+dur+30 > anno_inst['end_fid']: continue
            tem_seg_relation_insts = dict()
            tem_seg_relation_insts['triplet'] = inst['triplet']
            tem_seg_relation_insts['duration'] = (anno_inst['begin_fid']+dur, anno_inst['begin_fid']+dur+30)
            tem_seg_relation_insts['sub_traj'] = inst['sub_traj'][dur:dur+30]
            tem_seg_relation_insts['subject_tid'] = anno_inst['subject_tid']
            tem_seg_relation_insts['obj_traj'] = inst['obj_traj'][dur:dur+30]
            tem_seg_relation_insts['object_tid'] = anno_inst['object_tid']
            if (anno_inst['begin_fid']+dur, anno_inst['begin_fid']+dur+30) not in seg_relation_insts:
                seg_relation_insts[(anno_inst['begin_fid']+dur, anno_inst['begin_fid']+dur+30)] = []
            seg_relation_insts[(anno_inst['begin_fid']+dur, anno_inst['begin_fid']+dur+30)].append(tem_seg_relation_insts)
        
    return relation_insts, seg_relation_insts


def eval_rel_results(args, all_results, output_dir, topk=100, is_gt_traj=False):
    print('Loading test_videos_list.json')
    suffix_cator = 'traj_cls_gt' if is_gt_traj else 'traj_cls'

    val_map_list_path = os.path.join(cfg.ROOT_DIR, 'data', 'vidvrd', 'annotations/val_fname_list.json')
    with open(val_map_list_path, 'r') as f:
        val_map_list = json.load(f)
        f.close()
    val_map_list_ = set()
    for i, v in enumerate(val_map_list):
        ll = v.split('/')
        if len(ll) >= 2:
            val_map_list_.add(ll[-2].split('.')[-2])
    val_map_list = list(val_map_list_)

    print('test_videos_list.json loaded.')
    
    
    #prd_k_set = (132, )
    prd_k_set = (20, )
    
    
    SEG_STEP = 15
    
    
    N_list = [3, 7, 11, 14, 15, 19, 23, 27]
    w = [1.0 ,1.0 ,1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    if args.sharp_frame:
        full_video_path = './data/vidvrd/frames'
        if os.path.exists(os.path.join('./data/vidvrd', 'sharp_key_frame_set.pkl')):
            with open(os.path.join('./data/vidvrd', 'sharp_key_frame_set.pkl'), 'rb') as f:
                key_frame_set = pickle.load(f)
                f.close()
        else:
            key_frame_set = get_key_frame_set(full_video_path, N_list, \
                seg_step=SEG_STEP, seg_len=30, f_interval=3)
            with open(os.path.join('./data/vidvrd', 'sharp_key_frame_set.pkl'), 'wb') as f:
                pickle.dump(key_frame_set, f, pickle.HIGHEST_PROTOCOL)
                f.close()
    
    for prd_k in prd_k_set:
        print('prd_k = {}:'.format(prd_k))
        mean_topk_dets = dict()
        for i in range(0, 30):
            mean_topk_dets[i] = dict()
        
        for im_i, res in enumerate(tqdm(all_results)):
            if res is None: continue
            mm = res['image'].split('/')
            file_real_name = mm[-2]
            im_id_id = int(mm[-1].split('.')[0])
            #print(res['image'])
            
            ans_N_list = N_list
            ans_w = w
            if args.sharp_frame:
                ans_N_list, ans_w = judge_sharp_frame(key_frame_set, res['image'], seg_step=SEG_STEP, seg_len=30)
            #print(res['image'], ans_N_list)
            if len(ans_N_list) == 0:
                continue
            #print(ans_N_list)
            for iN, N in enumerate(ans_N_list):
                #if im_id_id % N != 0 or im_id_id == 0: continue
                if (im_id_id - N) % SEG_STEP != 0 or im_id_id < N or im_id_id == 0: continue
                
                pstart, pend = im_id_id - N, im_id_id + 30 - N
                if not os.path.exists(os.path.join('data/vidvrd/features', suffix_cator, \
                    file_real_name[:-4], file_real_name[:-4]+'-'+'{:04d}'.format(pstart)+\
                    '-'+'{:04d}'.format(pend)+'-'+suffix_cator+'.json')): continue
                
                
                if res['prd_scores'] is None:
                    det_boxes_s_top = np.zeros((0, 4), dtype=np.float32)
                    det_boxes_o_top = np.zeros((0, 4), dtype=np.float32)
                    det_labels_s_top = np.zeros(0, dtype=np.int32)
                    det_labels_p_top = np.zeros(0, dtype=np.int32)
                    det_labels_o_top = np.zeros(0, dtype=np.int32)
                    det_scores_top = np.zeros(0, dtype=np.float32)
                else:
                    det_boxes_sbj = res['sbj_boxes']
                    det_boxes_obj = res['obj_boxes']
                    det_labels_sbj = res['sbj_labels']
                    det_labels_obj = res['obj_labels']
                    det_scores_sbj = res['sbj_scores']
                    det_scores_obj = res['obj_scores']
                    
                    if 'prd_scores_ttl' in res: det_scores_prd = res['prd_scores_ttl']
                    else: det_scores_prd = res['prd_scores']

                    det_labels_prd = np.argsort(-det_scores_prd, axis=1)
                    det_scores_prd = -np.sort(-det_scores_prd, axis=1)

                    det_scores_so = det_scores_sbj * det_scores_obj
                    det_scores_spo = det_scores_so[:, None] * det_scores_prd[:, :prd_k]
                    
                    det_scores_inds = argsort_desc(det_scores_spo)[:topk]
                    
                    
                    det_scores_top = det_scores_spo[det_scores_inds[:, 0], det_scores_inds[:, 1]]
                    det_boxes_so_top = np.hstack(
                        (det_boxes_sbj[det_scores_inds[:, 0]], det_boxes_obj[det_scores_inds[:, 0]]))
                    det_labels_p_top = det_labels_prd[det_scores_inds[:, 0], det_scores_inds[:, 1]]
                    det_labels_spo_top = np.vstack(
                        (det_labels_sbj[det_scores_inds[:, 0]], 
                            det_labels_p_top, 
                            det_labels_obj[det_scores_inds[:, 0]])
                            ).transpose()

                    det_boxes_s_top = det_boxes_so_top[:, :4]
                    det_boxes_o_top = det_boxes_so_top[:, 4:]
                    det_labels_s_top = det_labels_spo_top[:, 0]
                    det_labels_p_top = det_labels_spo_top[:, 1]
                    det_labels_o_top = det_labels_spo_top[:, 2]
                    
                    
                    all_traj = []
                    xxyy_box = []
                    
                    
                    with open(os.path.join('data/vidvrd/features', suffix_cator, \
                        file_real_name[:-4], file_real_name[:-4]+'-'+'{:04d}'.format(pstart)+\
                        '-'+'{:04d}'.format(pend)+'-'+suffix_cator+'.json'), 'r') as f:
                        anno_list = json.load(f)
                        f.close()
                    for i in anno_list:
                        xxyy_box.append(i['rois'][N])
                        all_traj.append(i['rois'])
                    xxyy_box = np.stack(xxyy_box)
                    
                    sbj_to_pre_rois_overlaps = bbox_overlaps(
                        det_boxes_s_top.astype(dtype=np.float32, copy=False),
                        xxyy_box.astype(dtype=np.float32, copy=False)
                    )
                    obj_to_pre_rois_overlaps = bbox_overlaps(
                        det_boxes_o_top.astype(dtype=np.float32, copy=False),
                        xxyy_box.astype(dtype=np.float32, copy=False)
                    )
                    #if is_gt_traj:
                    #    sbj_maxes = sbj_to_pre_rois_overlaps.max(axis=1)
                    #    obj_maxes = obj_to_pre_rois_overlaps.max(axis=1)
                    #
                    #    filtered_inds = np.where((sbj_maxes > 1e-2) & (obj_maxes > 1e-2))[0]
                    #    sbj_to_pre_rois_overlaps = sbj_to_pre_rois_overlaps[filtered_inds]
                    #    obj_to_pre_rois_overlaps = obj_to_pre_rois_overlaps[filtered_inds]
                    #    det_boxes_s_top = det_boxes_s_top[filtered_inds]
                    #    det_boxes_o_top = det_boxes_o_top[filtered_inds]
                    #    det_scores_top = det_scores_top[filtered_inds]
                    #    det_labels_s_top = det_labels_s_top[filtered_inds]
                    #    det_labels_p_top = det_labels_p_top[filtered_inds]
                    #    det_labels_o_top = det_labels_o_top[filtered_inds]
                        
                    
                    sbj_argmaxes = sbj_to_pre_rois_overlaps.argmax(axis=1)
                    sbj_maxes = sbj_to_pre_rois_overlaps.max(axis=1)
                    s_traj_inds = sbj_argmaxes.copy()
                    
                    obj_argmaxes = obj_to_pre_rois_overlaps.argmax(axis=1)
                    obj_maxes = obj_to_pre_rois_overlaps.max(axis=1)
                    o_traj_inds = obj_argmaxes.copy()
                    
                    bad_inds = np.where(sbj_argmaxes == obj_argmaxes)[0]

                    if sbj_to_pre_rois_overlaps.shape[1] >= 3:
                        s_top2_inds = np.argpartition(sbj_to_pre_rois_overlaps[bad_inds], kth=2, axis=1)
                        o_top2_inds = np.argpartition(obj_to_pre_rois_overlaps[bad_inds], kth=2, axis=1)
                        for i, bad_inds_id in enumerate(bad_inds):
                            if sbj_maxes[bad_inds_id] > obj_maxes[bad_inds_id]:
                                s_traj_inds[bad_inds_id] = s_top2_inds[i][-2]
                            else:
                                o_traj_inds[bad_inds_id] = o_top2_inds[i][-2]
                                
                    elif sbj_to_pre_rois_overlaps.shape[1] == 2:
                        for i, bad_inds_id in enumerate(bad_inds):
                            if sbj_maxes[bad_inds_id] > obj_maxes[bad_inds_id]:
                                s_traj_inds[bad_inds_id] = 1 - s_traj_inds[bad_inds_id]
                            else:
                                o_traj_inds[bad_inds_id] = 1 - o_traj_inds[bad_inds_id]
                ans = []
                for i in range(len(det_boxes_s_top)):
                    st_weight = 1.0
                    ans.append([[det_scores_top[i] * ans_w[iN], ], 
                            (det_labels_s_top[i], det_labels_p_top[i], det_labels_o_top[i]), 
                            (s_traj_inds[i], o_traj_inds[i])
                            ])
                
                
                index = (file_real_name[:-4], pstart, pend)
                #if file_real_name[:-4] not in topk_dets: topk_dets[file_real_name[:-4]] = []
                #topk_dets[file_real_name[:-4]].append((index, ans))
                
                if index not in mean_topk_dets[N]: mean_topk_dets[N][index] = 0
                #print(N, index, mean_topk_dets[N][index])
                if mean_topk_dets[N][index] != 0: #assert False, 'Temporal Not Merged!'
                    for idxi, i in enumerate(mean_topk_dets[N][index]):
                        mean_topk_dets[N][index][idxi][0].append(i[0][0])
                        #print(mean_topk_dets[N][index][idxi][0])
                else:
                    mean_topk_dets[N][index] = ans #!
        
        
        mean_list_topk_dets = dict()
        for N, v in mean_topk_dets.items(): # N=15, v={index1: ans1, index2: ans2}
            for index, single_video_res_list in tqdm(v.items()): # single_video_res_list=[(x1, x2, ...]
                if index not in mean_list_topk_dets:
                    mean_list_topk_dets[index] = single_video_res_list.copy()
                else:
                    for quintuple in single_video_res_list:
                        is_hit = False
                        for i, new_quintuple in enumerate(mean_list_topk_dets[index]):
                            if quintuple[1] == new_quintuple[1] and quintuple[2] == new_quintuple[2]:
                                mean_list_topk_dets[index][i][0] += quintuple[0]
                                is_hit = True
                        if not is_hit:
                            mean_list_topk_dets[index].append(quintuple)
        
        topk_dets = dict()
        for index, single_video_res_list in mean_list_topk_dets.items():
            filename, pst, ped = index
            if filename not in topk_dets: topk_dets[filename] = []
            topk_dets[filename].append([index, single_video_res_list])
        
        for k, v_list in topk_dets.items():
            for j, v in enumerate(v_list):
                index, ans = v
                for i, rel_info in enumerate(ans):
                    #topk_dets[k][j][1][i][0] = np.mean(np.array(rel_info[0], dtype=np.float32))
                    topk_dets[k][j][1][i][0] = np.sum(np.array(rel_info[0], dtype=np.float32))
        
        print('Saving transform_vidvrd_results dets...')
        topk_dets_f = os.path.join(output_dir, 'transform_vidvrd_results.pkl')
        with open(topk_dets_f, 'wb') as f:
            pickle.dump(topk_dets, f, pickle.HIGHEST_PROTOCOL)
        print('Done.')
    return topk_dets

def form_trans(obj_name_mapping, pred_name_mapping, topk_dets, is_gt_traj=True):
    ans = dict()
    for k, v_list in tqdm(topk_dets.items()):
        for j, v in enumerate(v_list):
            index, single_video_res_list = v
            file_real_name, st, ed = index
            ans[file_real_name+'-'+str(st)+'-'+str(ed)] = []
            for i, rel_info in enumerate(single_video_res_list):
                score = rel_info[0]
                sl, pl, ol = rel_info[1]
                stj_inds, otj_inds = rel_info[2]
                
                suffix_cator = 'traj_cls_gt' if is_gt_traj else 'traj_cls'
                if not os.path.exists(os.path.join('data/vidvrd/features', suffix_cator, \
                    file_real_name, file_real_name+'-'+'{:04d}'.format(st)+\
                    '-'+'{:04d}'.format(ed)+'-'+suffix_cator+'.json')):
                        assert False
                    
                with open(os.path.join('data/vidvrd/features', suffix_cator, \
                    file_real_name, file_real_name+'-'+'{:04d}'.format(st)+\
                    '-'+'{:04d}'.format(ed)+'-'+suffix_cator+'.json'), 'r') as f:
                    anno_list = json.load(f)
                    f.close()
                
                
                d = dict()
                d['sub_traj'] = anno_list[stj_inds]['rois']
                d['obj_traj'] = anno_list[otj_inds]['rois']
                d['score'] = score
                d['triplet'] = [
                    obj_name_mapping[sl],
                    pred_name_mapping[pl],
                    obj_name_mapping[ol]
                ]
                d['duration'] = (st, ed)
                ans[file_real_name+'-'+str(st)+'-'+str(ed)].append(d)
    return ans            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="transform_vidvrd_results")
    parser.add_argument("--input_dir", default="Outputs/ag_val_50_vidvrd",
                        help="input_dir.")
    parser.add_argument("--output_dir", default="Outputs/ag_val_50_vidvrd",
                        help="output_dir.")
    parser.add_argument("--is_gt_traj", action="store_true",
                        help="is_gt_traj.")
    parser.add_argument("--topk", default=200, type=int,
                        help="output_dir.")
    parser.add_argument("--adapt_longtail_st", default=False,
                        help="output_dir.")
    parser.add_argument('--segment_groundtruth', 
            default="data/vidvrd/annotations/test_seg_gt.json", 
            help='A ground truth JSON file generated by yourself')
    parser.add_argument('--is_train', 
            action='store_true', 
            help='is_test')
    parser.add_argument('--is_seg_test', 
            action='store_true', 
            help='is_test')
    parser.add_argument('--sharp_frame', 
            action='store_true', 
            help='sharp_frame')
    args = parser.parse_args()
    
    path = './data/vidvrd/annotations'
    obj_json_path = os.path.join(path, 'objects.json')
    pred_json_path = os.path.join(path, 'predicates.json')
    with open(obj_json_path, 'r') as f:
        obj_class_list = json.load(f)
        f.close()
    with open(pred_json_path, 'r') as f:
        pred_class_list = json.load(f)
        f.close()
    
    if os.path.exists(os.path.join(cfg.ROOT_DIR, args.input_dir, 'rel_detections.pkl')):
        print('Loading rel_detections.pkl')
        with open(os.path.join(cfg.ROOT_DIR, args.input_dir, 'rel_detections.pkl'), 'rb') as f:
            all_results = pickle.load(f)
            f.close()
    elif os.path.exists(os.path.join(cfg.ROOT_DIR, args.input_dir, 'rel_detections_gt_boxes_sgcls.pkl')):
        print('Loading rel_detections_gt_boxes_sgcls.pkl')
        with open(os.path.join(cfg.ROOT_DIR, args.input_dir, 'rel_detections_gt_boxes_sgcls.pkl'), 'rb') as f:
            all_results = pickle.load(f)
            f.close()
    else:
        raise Exception
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    if not os.path.exists(os.path.join(args.output_dir, 'transform_vidvrd_results.pkl')):
        topk_dets = eval_rel_results(args, all_results, args.output_dir, 
                            #topk=args.topk, is_gt_traj=args.is_gt_traj)
                            topk=2*args.topk, is_gt_traj=args.is_gt_traj)
    else:
        with open(os.path.join(args.output_dir, 'transform_vidvrd_results.pkl'), 'rb') as f:
            topk_dets = pickle.load(f)
            f.close()
    
    if args.is_seg_test:
        if os.path.exists(args.segment_groundtruth):
            print('Loading segment ground truth from {}'.format(args.segment_groundtruth))
            with open(args.segment_groundtruth, 'r') as fp:
                gt_seg = json.load(fp)
                fp.close()
            print('Number of videos in ground truth: {}'.format(len(gt_seg)))
        else:
            gt_seg = dict()
            tet = 'train' if args.is_train else 'test'
            init_path = os.path.join('data/vidvrd/annotations', tet)
            video_anno_list = os.listdir(init_path)
            for i in tqdm(video_anno_list):
                vid = i.split('.')[0]
                _, x = get_relation_insts(i, tet)
                for k, v in x.items():
                    gt_seg[vid+'-'+str(k[0])+'-'+str(k[1])] = v
            with open(args.segment_groundtruth, 'w') as fp:
                json.dump(gt_seg, fp)
                fp.close()
        serid_topk_dets = form_trans(obj_class_list, pred_class_list, topk_dets, is_gt_traj=args.is_gt_traj)
        print('Seg metric:')
        mean_ap, rec_at_n, mprec_at_n = evaluate(gt_seg, serid_topk_dets)
    else:
    
        if not os.path.exists(os.path.join(args.output_dir, 'baseline_relation_prediction.json')):
            video_relations = dict()
            for vid in tqdm(topk_dets.keys()):
                video_relations[vid] = \
                    greedy_relational_association(obj_class_list, pred_class_list, 
                                        topk_dets[vid], max_traj_num_in_clip=args.topk, is_gt=args.is_gt_traj)
            with open(os.path.join(args.output_dir, 'baseline_relation_prediction.json'), 'w') as f:
                output = {
                    'version': 'VERSION 1.0',
                    'results': video_relations
                }
                json.dump(output, f)
                f.close()
        else:
            with open(os.path.join(args.output_dir, 'baseline_relation_prediction.json'), 'r') as f:
                output = json.load(f)
                f.close()
            video_relations = output['results']
            #for vid in tqdm(topk_dets.keys()):
            #    print(vid)
            #    print(video_relations[vid])
            #    assert False
    