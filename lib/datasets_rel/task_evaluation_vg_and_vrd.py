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

from core.config import cfg
from functools import reduce
from utils.boxes import bbox_overlaps
from utils_rel.boxes_rel import boxes_union

from .pytorch_misc import intersect_2d, argsort_desc

np.set_printoptions(precision=3)

logger = logging.getLogger(__name__)

#def get_prior_adj(path, num_rel, num_obj):
#    with open(path, 'r') as f:
#        anno = json.load(f)
#        f.close()
#    A = np.zeros((num_obj, num_obj, num_rel), dtype=np.int32)
#    A_adj = np.zeros((num_obj, num_obj, num_rel), dtype=np.int32)
#    for img_id, v_list in anno.items():
#        for v in v_list:
#            p = v['predicate']
#            sc = v['subject']['category']
#            oc = v['object']['category']
#            A[sc, oc, p] += 1
#            A_adj[sc, oc, p] = 1
#    return A, A_adj
def debug_match(anslist):
    ans = []
    pred_inds = []
    for id, i in enumerate(anslist):
        if len(i) > 0:
            ans.append(i[0])
            pred_inds.append(id)
    ans = np.array(ans, dtype=np.int)
    pred_inds = np.array(pred_inds, dtype=np.int)
    return ans, pred_inds

def eval_rel_results(all_results, output_dir, topk=100, do_val=True):
    print('Loading test_videos_list.json')
    if cfg.TEST.DATASETS[0].find('ag') >= 0:
        val_map_list_path = os.path.join(cfg.ROOT_DIR, 'data', 'ag', 'annotations/test_videos_list.json')
        with open(val_map_list_path, 'r') as f:
            val_map_list = json.load(f)
            f.close()    
    elif cfg.TEST.DATASETS[0].find('vidvrd_train') >= 0:
        val_map_list_path = os.path.join(cfg.ROOT_DIR, 'data', 'vidvrd', 'annotations/train_fname_list.json')
        with open(val_map_list_path, 'r') as f:
            val_map_list = json.load(f)
            f.close()
        val_map_list_ = set()
        for i, v in enumerate(val_map_list):
            ll = v.split('/')
            if len(ll) >= 2:
                val_map_list_.add(ll[-2].split('.')[-2])
        val_map_list = list(val_map_list_)
    elif cfg.TEST.DATASETS[0].find('vidvrd') >= 0:
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
    else:
        raise Exception
    print('test_videos_list.json loaded.')
    
    if cfg.TEST.DATASETS[0].find('ag') >= 0:
        #prd_k_set = (2, 3, 4, 10, 26, )
        ###prd_k_set = (7, 6)
        prd_k_set = (6, 7)
    elif cfg.TEST.DATASETS[0].find('vidvrd') >= 0:
        #prd_k_set = []
        #for i in range(132, 9, -10):
        #    prd_k_set.append(i)
        #prd_k_set = tuple(prd_k_set)
        #prd_k_set = (132, )
        prd_k_set = (20, )
    elif cfg.TEST.DATASETS[0].find('vg') >= 0:
        prd_k_set = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20)
    elif cfg.TEST.DATASETS[0].find('vrd') >= 0:
        prd_k_set = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 70)
    else:
        prd_k_set = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    
    if cfg.TEST.DATASETS[0].find('vrd') < 0:
        eval_sets = (False,)
    else:
        eval_sets = (False, True)
    if cfg.TEST.DATASETS[0].find('vidvrd') >= 0:
        eval_sets = (False,)
    
    N_list = [15, 19, 11, 23, 3]
    N_list = [3, 7, 11, 14, 15, 19, 23, 27]
    SEG_STEP = 15
    cnt = 0.
    alcnt = 0.
    #prior_A, prior_A_adj = get_prior_adj(
    #            os.path.join(cfg.ROOT_DIR, 'data', \
    #                cfg.TEST.DATASETS[0].split('_')[0], \
    #                'annotations/new_annotations_train.json'), 
    #            cfg.MODEL.NUM_PRD_CLASSES, cfg.MODEL.NUM_CLASSES)
    
    for phrdet in eval_sets:
        eval_metric = 'phrdet' if phrdet else 'reldet(SGGen)' ###!!!
        print('================== {} =================='.format(eval_metric))

        for prd_k in prd_k_set:
            print('prd_k = {}:'.format(prd_k))

            recalls = {10:0, 20: 0, 50: 0, 100: 0}
            tot_recalls = {10:0, 20: 0, 50: 0, 100: 0}
            if do_val:
                all_gt_cnt = 0
                video_gt_cnt = {}
                video_recalls = {}
                video_len = len(val_map_list)
                for j, i in enumerate(val_map_list):
                    file_real_name = i + '.mp4'
                    video_gt_cnt[file_real_name] = 0
                    video_recalls[file_real_name] = {10:0, 20: 0, 50: 0, 100: 0}

            topk_dets = []
            for im_i, res in enumerate(tqdm(all_results)):
                if cfg.TEST.DATASETS[0].find('vidvrd') >= 0:
                    if res is None:
                        continue
                
                mm = res['image'].split('/')
                file_real_name = mm[-2]
                cur_frame_id = int(mm[-1].split('.')[0])
                
                if cfg.TEST.DATASETS[0].find('vidvrd') >= 0:
                    flg = False
                    for N in N_list:
                        if (cur_frame_id - N) % SEG_STEP == 0 and cur_frame_id >= N and cur_frame_id != 0:
                            flg = True
                    if not flg: continue
                
                
                
                # in oi_all_rel some images have no dets
                if res['prd_scores'] is None:
                    det_boxes_s_top = np.zeros((0, 4), dtype=np.float32)
                    det_boxes_o_top = np.zeros((0, 4), dtype=np.float32)
                    det_labels_s_top = np.zeros(0, dtype=np.int32)
                    det_labels_p_top = np.zeros(0, dtype=np.int32)
                    det_labels_o_top = np.zeros(0, dtype=np.int32)
                    det_scores_top = np.zeros(0, dtype=np.float32)
                else:
                    det_boxes_sbj = res['sbj_boxes']  # (#num_rel, 4)
                    det_boxes_obj = res['obj_boxes']  # (#num_rel, 4)
                    det_labels_sbj = res['sbj_labels']  # (#num_rel,)
                    det_labels_obj = res['obj_labels']  # (#num_rel,)
                    
                    if len(res['sbj_scores'].shape) < 2:
                        det_scores_sbj = res['sbj_scores']  # (#num_rel,)
                    else:
                        det_scores_sbj = np.amax(res['sbj_scores'][:, 1:], axis=1)
                    if len(res['obj_scores'].shape) < 2:
                        det_scores_obj = res['obj_scores']  # (#num_rel,)
                    else:
                        det_scores_obj = np.amax(res['obj_scores'][:, 1:], axis=1)
                    if cfg.MODEL.MULTI_RELATION:
                        if 'prd_scores_ttl' in res:
                            det_scores_prd = res['prd_scores_ttl']
                        else:
                            det_scores_prd = res['prd_scores']
                    else:
                        if 'prd_scores_ttl' in res:
                            det_scores_prd = res['prd_scores_ttl'][:, 1:]
                        else:
                            det_scores_prd = res['prd_scores'][:, 1:]
                    
                    #if cfg.TEST.DATASETS[0].find('ag') >= 0:
                    #    person_id = np.where(det_labels_sbj == 0)[0]
                    #    det_boxes_sbj = det_boxes_sbj[person_id]
                    #    det_boxes_obj = det_boxes_obj[person_id]
                    #    det_labels_sbj = det_labels_sbj[person_id]
                    #    det_labels_obj = det_labels_obj[person_id]
                    #    det_scores_sbj = det_scores_sbj[person_id]
                    #    det_scores_obj = det_scores_obj[person_id]
                    #    det_scores_prd = det_scores_prd[person_id]
                    
                    #det_scores_prd = det_scores_prd * prior_A_adj[det_labels_sbj, det_labels_obj]
                    
                    #det_scores_prd = np.ones_like(det_scores_prd, dtype=np.float32) #!!!!
                    #det_scores_prd = np.random.rand(det_scores_prd.shape[0], det_scores_prd.shape[1]) #!!!!
                    
                    det_labels_prd = np.argsort(-det_scores_prd, axis=1)
                    det_scores_prd = -np.sort(-det_scores_prd, axis=1)
                    
                    
                    
                    ##det_scores_sbj = np.asarray(det_scores_sbj) / np.linalg.norm(det_scores_sbj, ord=1)
                    ##det_scores_obj = np.asarray(det_scores_obj) / np.linalg.norm(det_scores_obj, ord=1)
                    det_scores_so = det_scores_sbj * det_scores_obj
                    det_scores_spo = det_scores_so[:, None] * det_scores_prd[:, :prd_k]
                    ##det_scores_spo = det_scores_prd[:, :prd_k]
                    
                    #det_scores_so = det_scores_sbj + det_scores_obj
                    #det_scores_spo = det_scores_so[:, None] + det_scores_prd[:, :prd_k]
                    
                    
                    
                    
                    cnt += 1. *  (det_scores_spo.size <= topk)
                    alcnt += 1.
                    
                    det_scores_inds = argsort_desc(det_scores_spo)[:topk]
                    det_scores_top = det_scores_spo[det_scores_inds[:, 0], det_scores_inds[:, 1]]
                    
                    
                    #valid_inds = np.where(det_scores_top >= 0.5)[0] ###!!! 
                    #det_scores_inds = det_scores_inds[valid_inds, :]
                    #det_scores_top = det_scores_top[valid_inds]
                    
                    
                    det_boxes_so_top = np.hstack(
                        (det_boxes_sbj[det_scores_inds[:, 0]], det_boxes_obj[det_scores_inds[:, 0]]))
                    det_labels_p_top = det_labels_prd[det_scores_inds[:, 0], det_scores_inds[:, 1]]
                    det_labels_spo_top = np.vstack(
                        (det_labels_sbj[det_scores_inds[:, 0]], det_labels_p_top, det_labels_obj[det_scores_inds[:, 0]])).transpose()
                    
                    #print(det_scores_sbj[:10], det_scores_obj[:10], det_scores_prd[:10, :20])
                    #print()

                    det_boxes_s_top = det_boxes_so_top[:, :4]
                    det_boxes_o_top = det_boxes_so_top[:, 4:]
                    det_labels_s_top = det_labels_spo_top[:, 0]
                    det_labels_p_top = det_labels_spo_top[:, 1]
                    det_labels_o_top = det_labels_spo_top[:, 2]
                

                
                
                topk_dets.append(dict(image=res['image'],
                                      det_boxes_s_top=det_boxes_s_top,
                                      det_boxes_o_top=det_boxes_o_top,
                                      det_labels_s_top=det_labels_s_top,
                                      det_labels_p_top=det_labels_p_top,
                                      det_labels_o_top=det_labels_o_top,
                                      det_scores_top=det_scores_top))

                if do_val:
                    gt_boxes_sbj = res['gt_sbj_boxes']  # (#num_gt, 4)
                    gt_boxes_obj = res['gt_obj_boxes']  # (#num_gt, 4)
                    gt_labels_sbj = res['gt_sbj_labels']  # (#num_gt,)
                    gt_labels_obj = res['gt_obj_labels']  # (#num_gt,)
                    gt_labels_prd = res['gt_prd_labels']  # (#num_gt,)
                    gt_boxes_so = np.hstack((gt_boxes_sbj, gt_boxes_obj))
                    gt_labels_spo = np.vstack((gt_labels_sbj, gt_labels_prd, gt_labels_obj)).transpose()
                    # Compute recall. It's most efficient to match once and then do recall after
                    # det_boxes_so_top is (#num_rel, 8)
                    # det_labels_spo_top is (#num_rel, 3)
                    
                    #print(mm)
                    #print(gt_labels_spo)
                    #print(gt_boxes_so)
                    #print()
                    
                    if phrdet:
                        det_boxes_r_top = boxes_union(det_boxes_s_top, det_boxes_o_top)
                        gt_boxes_r = boxes_union(gt_boxes_sbj, gt_boxes_obj)
                        pred_to_gt = _compute_pred_matches(
                            gt_labels_spo, det_labels_spo_top,
                            gt_boxes_r, det_boxes_r_top,
                            iou_thresh=0.5,
                            phrdet=phrdet)
                    else:
                        pred_to_gt = _compute_pred_matches(
                            gt_labels_spo, det_labels_spo_top,
                            gt_boxes_so, det_boxes_so_top,
                            iou_thresh=0.5,
                            phrdet=phrdet)
                    
                    all_gt_cnt += gt_labels_spo.shape[0]
                    #all_gt_cnt += 1
                    
                    video_gt_cnt[file_real_name] += gt_labels_spo.shape[0]
                    #video_gt_cnt[file_real_name] += 1
                    
                    for k in recalls:
                        if len(pred_to_gt):
                            match = reduce(np.union1d, pred_to_gt[:k])
                            
                            #if k == 50:
                            #    print()
                            #    print(match)
                            #    print(gt_labels_spo.shape)
                            #    print()
                            #    gtid, prdid = debug_match(pred_to_gt[:k])
                            #    print(gt_labels_spo[gtid])
                            #    print(det_labels_spo_top[prdid])
                            #    print(gt_boxes_so[gtid])
                            #    print(det_boxes_so_top[prdid])
                        else:
                            match = []
                            
                        recalls[k] += len(match)
                        #recalls[k] += float(len(match)) / (float(gt_labels_spo.shape[0]) + 1e-12)

                        video_recalls[file_real_name][k] += len(match)
                        #video_recalls[file_real_name][k] += float(len(match)) / (float(gt_labels_spo.shape[0]) + 1e-12)

                    topk_dets[-1].update(dict(gt_boxes_sbj=gt_boxes_sbj,
                                              gt_boxes_obj=gt_boxes_obj,
                                              gt_labels_sbj=gt_labels_sbj,
                                              gt_labels_obj=gt_labels_obj,
                                              gt_labels_prd=gt_labels_prd))

            if do_val:
                for k in recalls:
                    recalls[k] = float(recalls[k]) / (float(all_gt_cnt) + 1e-12)
                
                for file_real_name, v in video_recalls.items():
                    for k, vals in v.items():
                        video_recalls[file_real_name][k] = float(vals) / (float(video_gt_cnt[file_real_name]) + 1e-12)
                        tot_recalls[k] += video_recalls[file_real_name][k]
                
                for k in tot_recalls:
                    tot_recalls[k] = float(tot_recalls[k]) / (float(video_len) + 1e-12)
                    
                print('=========== ' + 'Image_ver' + ' ===========')
                print_stats(recalls)
                print('=========== ' + 'Video_ver' + ' ===========')
                print_stats(tot_recalls)
                
        print(cnt / alcnt)
        
        
        print('Saving topk dets...')
        topk_dets_f = os.path.join(output_dir, 'rel_detections_topk.pkl')
        with open(topk_dets_f, 'wb') as f:
            pickle.dump(topk_dets, f, pickle.HIGHEST_PROTOCOL)
        logger.info('topk_dets size: {}'.format(len(topk_dets)))
        print('Done.')


def print_stats(recalls):
    # print('====================== ' + 'sgdet' + ' ============================')
    for k, v in recalls.items():
        print('R@%i: %.2f' % (k, 100 * v))


# This function is adapted from Rowan Zellers' code:
# https://github.com/rowanz/neural-motifs/blob/master/lib/evaluation/sg_eval.py
# Modified for this project to work with PyTorch v0.4
def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh=0.5, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets: 
    :param pred_triplets: 
    :param gt_boxes: 
    :param pred_boxes: 
    :param iou_thresh: 
    :return: 
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            gt_box = gt_box.astype(dtype=np.float32, copy=False)
            boxes = boxes.astype(dtype=np.float32, copy=False)
            rel_iou = bbox_overlaps(gt_box[None, :], boxes)[0]

            inds = rel_iou >= iou_thresh
        else:
            gt_box = gt_box.astype(dtype=np.float32, copy=False)
            boxes = boxes.astype(dtype=np.float32, copy=False)
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt
