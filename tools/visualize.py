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


from graphviz import Digraph


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
        '--st',
        help='Visualization Start',
        default=0, type=int)
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

    if not os.path.exists(os.path.join(args.output_dir, 'vis')):
        os.mkdir(os.path.join(args.output_dir, 'vis'))
    saved_path = os.path.join(args.output_dir, 'vis')
    
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
    
    print('Loading test_videos_list.json')
    if args.dataset.find('ag') >= 0:
        val_map_list_path = os.path.join('data', 'ag', 'annotations/test_videos_list.json')
        with open(val_map_list_path, 'r') as f:
            val_map_list = json.load(f)
            f.close()    
    elif args.dataset.find('vidvrd_train') >= 0:
        val_map_list_path = os.path.join('data', 'vidvrd', 'annotations/train_fname_list.json')
        with open(val_map_list_path, 'r') as f:
            val_map_list = json.load(f)
            f.close()
        val_map_list_ = set()
        for i, v in enumerate(val_map_list):
            ll = v.split('/')
            if len(ll) >= 2:
                val_map_list_.add(ll[-2].split('.')[-2])
        val_map_list = list(val_map_list_)
    elif args.dataset.find('vidvrd') >= 0:
        val_map_list_path = os.path.join('data', 'vidvrd', 'annotations/val_fname_list.json')
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
    
    all_gt_cnt = 0
    video_gt_cnt = [{} for r in range(len(rel_list))]
    video_recalls = [{} for r in range(len(rel_list))]
    video_len = len(val_map_list)
    for j, i in enumerate(val_map_list):
        file_real_name = i + '.mp4'
        for r in range(len(rel_list)):
            video_gt_cnt[r][file_real_name] = 0.
            video_recalls[r][file_real_name] = {10:0., 20: 0., 50: 0., 100: 0.}
    
    edge_width = 3
    font_size = 18
    
    rel_class_recalls = [{10:0, 20: 0, 50: 0, 100: 0} for i in range(len(rel_list))]
    tot_recalls = [{10:0, 20: 0, 50: 0, 100: 0} for i in range(len(rel_list))]
    rel_class_gt_num = [0 for i in range(len(rel_list))]
    recalls = [10, 20, 50, 100]
    print('total {} images. '.format(len(res)))
    args.num = min(args.num, len(res))
    print('Number is {}. '.format(args.num))
    cnt = 0
    for res_i in res[args.st:args.num]:
        r_ans = {10:0, 20: 0, 50: 0, 100: 0}
        r_score = {10:0, 20: 0, 50: 0, 100: 0}
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
        
        mm = res_i['image'].split('/')
        file_real_name = mm[-2]
        cur_frame_id = int(mm[-1].split('.')[0])
        
        gt_boxes_so = np.hstack((gt_boxes_sbj, gt_boxes_obj))
        gt_labels_spo = np.vstack((gt_labels_sbj, gt_labels_prd, gt_labels_obj)).transpose()
        det_labels_spo_top = np.vstack((det_labels_s_top, det_labels_p_top, det_labels_o_top)).transpose()
        
        if args.rel_class_recall:
            for i in gt_labels_prd:
                rel_class_gt_num[i] += 1
                video_gt_cnt[i][file_real_name] += 1.
        
        det_boxes_so_top = np.hstack((det_boxes_s_top, det_boxes_o_top))
        pred_to_gt = _compute_pred_matches(
                    gt_labels_spo, det_labels_spo_top,
                    gt_boxes_so, det_boxes_so_top,
                    phrdet=args.phrdet)
        
        gt_obj_set = set()
        gt_tri_info_set = set()
        for i in range(len(gt_boxes_sbj)):
            tri_info = []
            tri_info += list(gt_boxes_sbj[i, :].astype(np.int))
            tri_info += [gt_labels_sbj[i].astype(np.int), ]
            tri_info += list(gt_boxes_obj[i, :].astype(np.int))
            tri_info += [gt_labels_obj[i].astype(np.int), ]
            tri_info += [gt_labels_prd[i].astype(np.int), ]
            gt_tri_info_set.add(tuple(tri_info))
            if tuple(list(gt_boxes_sbj[i, :].astype(np.int)) + [gt_labels_sbj[i].astype(np.int), ]) not in gt_obj_set:
                gt_obj_set.add(tuple(list(gt_boxes_sbj[i, :].astype(np.int)) + [gt_labels_sbj[i].astype(np.int), ]))
            if tuple(list(gt_boxes_obj[i, :].astype(np.int)) + [gt_labels_obj[i].astype(np.int), ]) not in gt_obj_set:
                gt_obj_set.add(tuple(list(gt_boxes_obj[i, :].astype(np.int)) + [gt_labels_obj[i].astype(np.int), ]))
        
        for k in recalls:
            gt_score = [0 for i in range(len(gt_boxes_sbj))]
            if len(pred_to_gt):
                match = reduce(np.union1d, pred_to_gt[:k])
                match = np.array(match, dtype=np.int)
                
                if args.rel_class_recall:
                    for gt_i in match:
                        rel_class_recalls[gt_labels_prd[gt_i]][k] += 1
                        video_recalls[gt_labels_prd[gt_i]][file_real_name][k] += 1.
                
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
                cnt += 1
                for k in recalls:
                    if k < 20: continue
                    if k > 50: continue
                    preserve_set_obj = set()
                    preserve_set_rel = set()
                    rec_pos = {}
                    fig = plt.figure(figsize=(18, 12))
                    ax = plt.gca()
                    plt.imshow(img)
                    plt.axis('off')
                    det_title = plt.title('det')
                    plt.setp(det_title, color='b')
                    for gt_id in r_ans[k]:
                        x, y, x1, y1 = gt_boxes_sbj[gt_id].astype(np.int)
                        
                        s_name = obj_list[gt_labels_sbj[gt_id]]
                        
                        s_cx, s_cy = (x+x1)//2, (y+y1)//2 
                        
                        srect = plt.Rectangle((x,y),x1-x,y1-y, fill=False, edgecolor='b', linewidth=3)
                        ax.add_patch(srect)
                        
                        #ax.text(s_cx, s_cy,
                        ax.text(x, y,
                            s_name,
                            fontsize=font_size,
                            color='white',
                            bbox=dict(facecolor='orange', alpha=0.5, pad=0, edgecolor='none'))
                        
                        tri_info = [x, y, x1, y1, gt_labels_sbj[gt_id].astype(np.int)]
                        if tuple([x, y, x1, y1, gt_labels_sbj[gt_id].astype(np.int)]) not in preserve_set_obj:
                            preserve_set_obj.add(tuple([x, y, x1, y1, gt_labels_sbj[gt_id].astype(np.int)]))
                        
                        x, y, x1, y1 = gt_boxes_obj[gt_id].astype(np.int)
                        o_name = obj_list[gt_labels_obj[gt_id]]
                        o_cx, o_cy = (x+x1)//2, (y+y1)//2 
                        
                        orect = plt.Rectangle((x,y),x1-x,y1-y, fill=False, edgecolor='b', linewidth=3)
                        ax.add_patch(orect)
                        
                        ax.text(x, y,
                            o_name,
                            fontsize=font_size,
                            color='white',
                            bbox=dict(facecolor='blue', alpha=0.5, pad=0, edgecolor='none'))
                        p_name = rel_list[gt_labels_prd[gt_id].astype(np.int)]+ ' ' + str(r_score[k][gt_id])
                        
                        tri_info += [x, y, x1, y1, gt_labels_obj[gt_id].astype(np.int)]
                        if tuple([x, y, x1, y1, gt_labels_obj[gt_id].astype(np.int)]) not in preserve_set_obj:
                            preserve_set_obj.add(tuple([x, y, x1, y1, gt_labels_obj[gt_id].astype(np.int)]))
                            
                        tri_info += [gt_labels_prd[gt_id].astype(np.int), ]
                        preserve_set_rel.add(tuple(tri_info))
                        
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
                    
                    dot = Digraph(filename=(saved_name + '_' +str(k)).replace('/', '_'))
                    dot.body.append('size="16,16"')
                    dot.body.append('rankdir="LR"')
                    #dot.node_attr.update(style='filled')
                    
                    
                    map_obj_node = dict()
                    pn = 0
                    for gt_obj in gt_obj_set:
                        ol = obj_list[gt_obj[-1].astype(np.int)]
                        if gt_obj in preserve_set_obj:
                            dot.node(str(gt_obj), str(ol), color='green', shape='box')
                        else:
                            dot.node(str(gt_obj), str(ol), color='red', shape='box')
                        map_obj_node[gt_obj] = pn
                        pn += 1
                        
                    for gt_tri_info in gt_tri_info_set:
                        sn, on, pn = gt_tri_info[4], gt_tri_info[9], gt_tri_info[10]
                        sx, sy, sx1, sy1 = gt_tri_info[0],gt_tri_info[1],gt_tri_info[2],gt_tri_info[3]
                        ox, oy, ox1, oy1 = gt_tri_info[5],gt_tri_info[6],gt_tri_info[7],gt_tri_info[8]
                        st, ed = str(tuple([sx, sy, sx1, sy1, sn])), str(tuple([ox, oy, ox1, oy1, on]))
                        rl = rel_list[gt_tri_info[-1].astype(np.int)].replace('_', ' ')
                        #if gt_tri_info in preserve_set_rel:
                        #    dot.node(str(gt_tri_info), rl, color='lightblue2')
                        #else:
                        #    dot.node(str(gt_tri_info), rl, color='red')
                        #dot.edge(st, str(gt_tri_info))
                        #dot.edge(str(gt_tri_info), ed)
                        if gt_tri_info in preserve_set_rel:
                            dot.edge(st, ed, rl, color='green')
                        else:
                            dot.edge(st, ed, rl, fontcolor='red', color='red')
                    
                    
                    dot.render(os.path.join(
                                saved_path, 
                                (saved_name + '_' +str(k)).replace('/', '_')
                                ), cleanup=True)
                    
                    saved_file_name = (saved_name + '_' +str(k)+'.png').replace('/', '_')
                    plt.savefig(os.path.join(saved_path, saved_file_name), bbox_inches='tight')
                    plt.close(fig)
                
        
    if args.rel_class_recall:
        for r in range(len(rel_list)):
            for file_real_name, v in video_recalls[r].items():
                for k, vals in v.items():
                    video_recalls[r][file_real_name][k] = float(vals) / (float(video_gt_cnt[r][file_real_name]) + 1e-12)
                    tot_recalls[r][k] += video_recalls[r][file_real_name][k]
            for k in tot_recalls[r]:
                tot_recalls[r][k] = float(tot_recalls[r][k]) / (float(video_len) + 1e-12)
        
        print('=========== ' + 'Image_ver_rel_recalls' + ' ===========')
        mr_list = []
        for k in recalls:
            print('=========== {} ==========='.format(k))
            mrr = float(0.)
            for i, gt_rel_num in enumerate(rel_class_gt_num):
                rel_class_recalls[i][k] = float(rel_class_recalls[i][k]) / (float(gt_rel_num) + 1e-12)
                print('%s: %.2f' % (rel_list[i], 100 * rel_class_recalls[i][k]))
                mrr += rel_class_recalls[i][k]
            mr_list.append((k, 100*mrr/len(rel_class_gt_num)))
        for i in mr_list:    
            print('mR@{}: {}'.format(i[0], i[1]))
        
        print('=========== ' + 'Video_ver_rel_recalls' + ' ===========')
        mr_v_list = []
        for k in recalls:
            mrr = float(0.)
            for i, gt_rel_num in enumerate(rel_class_gt_num):
                #print('%s: %.2f' % (rel_list[i], 100 * tot_recalls[i][k]))
                mrr += tot_recalls[i][k]
            mr_v_list.append((k, 100*mrr/len(rel_class_gt_num)))
        for i in mr_v_list:    
            print('mR@{}: {}'.format(i[0], i[1]))
            
            