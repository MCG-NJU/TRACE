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

from datasets_rel.pytorch_misc import intersect_2d, argsort_desc

from functools import reduce
from utils.boxes import bbox_overlaps
from utils_rel.boxes_rel import boxes_union




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
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')
    parser.add_argument(
        '--num',
        help='Visualization Number',
        default=10, type=int)
    parser.add_argument(
        '--no_do_vis',
        help='do not visualize',
        action='store_true')
    parser.add_argument(
        '--rel_class_recall', help='rel class recall.',
        action='store_true')
    parser.add_argument(
        '--phrdet', help='use phrdet.',
        action='store_true')
    parser.add_argument(
        '--dataset',
        help='Visualization Number',
        default='ag', type=str)    
    parser.add_argument(
        '--filename',
        help='Visualization file',
        default='rel_detections_topk', type=str) 
    parser.add_argument(
        '--cnt_lim',
        help='Visualization Number',
        default=10, type=int)
    parser.add_argument(
        '--lim',
        help='Visualization Number',
        default=0, type=int)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.output_dir, 'vis_gt')):
        os.mkdir(os.path.join(args.output_dir, 'vis_gt'))
    saved_path = os.path.join(args.output_dir, 'vis_gt')
    
    topk_dets_f = os.path.join(args.output_dir, args.filename+'.pkl')
    with open(topk_dets_f, 'rb') as f:
        res = pickle.load(f)
        f.close()
    
    with open(os.path.join('data', args.dataset,'annotations/objects.json'), 'r') as f:
        obj_list = json.load(f)
        f.close()
    with open(os.path.join('data', args.dataset, 'annotations/predicates.json'), 'r') as f:
        rel_list = json.load(f)
        f.close()
    
    edge_width = 3
    font_size = 18
    
    rel_class_recalls = [{20: 0, 50: 0} for i in range(len(rel_list))]
    rel_class_gt_num = [0 for i in range(len(rel_list))]
    recalls = [20, 50]
    print('total {} images. '.format(len(res)))
    args.num = min(args.num, len(res))
    print('Number is {}. '.format(args.num))
    cnt = 0
    for res_i in res[:args.num]:
        r_ans = {20: 0, 50: 0}
        r_score = {20: 0, 50: 0}
        f_name = res_i['image']
        det_boxes_s_top = res_i['det_boxes_s_top']
        det_boxes_o_top = res_i['det_boxes_o_top']
        det_labels_s_top = res_i['det_labels_s_top']
        det_labels_p_top = res_i['det_labels_p_top']
        det_labels_o_top = res_i['det_labels_o_top']
        det_scores_top = res_i['det_scores_top']
        gt_boxes_sbj = res_i['gt_boxes_sbj']
        gt_boxes_obj = res_i['gt_boxes_obj']
        gt_labels_sbj = res_i['gt_labels_sbj']
        gt_labels_obj = res_i['gt_labels_obj']
        gt_labels_prd = res_i['gt_labels_prd']
        gt_boxes_so = np.hstack((gt_boxes_sbj, gt_boxes_obj))
        gt_labels_spo = np.vstack((gt_labels_sbj, gt_labels_prd, gt_labels_obj)).transpose()
        det_labels_spo_top = np.vstack((det_labels_s_top, det_labels_p_top, det_labels_o_top)).transpose()
        
        if args.rel_class_recall:
            for i in gt_labels_prd:
                rel_class_gt_num[i] += 1
        
        det_boxes_so_top = np.hstack((det_boxes_s_top, det_boxes_o_top))
        pred_to_gt = _compute_pred_matches(
                    gt_labels_spo, det_labels_spo_top,
                    gt_boxes_so, det_boxes_so_top,
                    phrdet=args.phrdet)
        
        
        for k in recalls:
            gt_score = [0 for i in range(len(gt_boxes_sbj))]
            if len(pred_to_gt):
                match = reduce(np.union1d, pred_to_gt[:k]).astype(np.int)
                
                if args.rel_class_recall:
                    for gt_i in match:
                        rel_class_recalls[gt_labels_prd[gt_i]][k] += 1
                
                for p_id, pred_i in enumerate(pred_to_gt[:k]):
                    for gt_id in pred_i:
                        gt_score[gt_id] = max(gt_score[gt_id], det_scores_top[p_id])
                    
            else:
                match = []
                gt_score = []
            r_ans[k] = match
            r_score[k] = gt_score
        if len(gt_labels_prd) > args.lim and cnt <= args.cnt_lim:
            if not args.no_do_vis:
                saved_name = f_name.split('/')[-2:]
                saved_name = saved_name[0] + '/' + saved_name[1]
                img = mpimg.imread(f_name)
                for k in recalls:
                    rec_pos = {}
                    fig = plt.figure(figsize=(18, 12))
                    ax = plt.gca()
                    plt.imshow(img)
                    plt.axis('off')
                    det_title = plt.title('det')
                    plt.setp(det_title, color='b')
                    for i in range(len(gt_boxes_sbj)):
                        x, y, x1, y1 = gt_boxes_sbj[i].astype(np.int)
                        
                        s_name = obj_list[gt_labels_sbj[i]]
                        
                        s_cx, s_cy = (x+x1)//2, (y+y1)//2 
                        
                        srect = plt.Rectangle((x,y),x1-x,y1-y, fill=False, edgecolor='b', linewidth=3)
                        ax.add_patch(srect)
                        
                        ax.text(x, y,
                            s_name,
                            fontsize=font_size,
                            color='white',
                            bbox=dict(facecolor='orange', alpha=0.5, pad=0, edgecolor='none'))
                        
                        
                        x, y, x1, y1 = gt_boxes_obj[i].astype(np.int)
                        o_name = obj_list[gt_labels_obj[i]]
                        o_cx, o_cy = (x+x1)//2, (y+y1)//2 
                        
                        orect = plt.Rectangle((x,y),x1-x,y1-y, fill=False, edgecolor='b', linewidth=3)
                        ax.add_patch(orect)
                        
                        ax.text(x, y,
                            o_name,
                            fontsize=font_size,
                            color='white',
                            bbox=dict(facecolor='blue', alpha=0.5, pad=0, edgecolor='none'))
                        
                        p_name = rel_list[gt_labels_prd[i].astype(np.int)]
                        
                        rel_l = lines.Line2D([s_cx, o_cx], [s_cy, o_cy], color='purple', linewidth=3)
                        ax.add_line(rel_l)
                        
                        lx, ly = s_cx + 8*(o_cx - s_cx) / 9, s_cy + 8*(o_cy - s_cy) / 9
                        
                        if (lx, ly) in rec_pos:
                            rec_pos[(lx, ly)] += 10
                        else:
                            rec_pos[(lx, ly)] = 0
                        d = rec_pos[(lx, ly)]
                            
                        ax.text(lx, ly + d,
                                p_name,
                                fontsize=font_size,
                                color='white',
                                bbox=dict(facecolor='purple', alpha=0.5, pad=0, edgecolor='none'))
                    
                    saved_file_name = (saved_name + '_' +str(k)+'.png').replace('/', '_')
                    plt.savefig(os.path.join(saved_path, saved_file_name), bbox_inches='tight')
                    plt.close(fig)
            
        
    if args.rel_class_recall:
        print('=========== ' + 'Image_ver_rel_recalls' + ' ===========')
        for k in recalls:
            print('=========== {} ==========='.format(k))
            for i, gt_rel_num in enumerate(rel_class_gt_num):
                rel_class_recalls[i][k] = float(rel_class_recalls[i][k]) / (float(gt_rel_num) + 1e-12)
                print('%s: %.2f' % (rel_list[i], 100 * rel_class_recalls[i][k]))
    