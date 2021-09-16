import numpy as np
import matplot

import pickle
import json
import argparse
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.lines as lines
from tqdm import tqdm

def batch_box_iou(boxes1, boxes2):
    """
        boxes1: [B, N, 4]
        boxes2: [B, M, 4]
        
        return: [B, N, M]
    """
    area1 = (boxes1[:, :, 3] - boxes1[:, :, 1] + 1) * (boxes1[:, :, 2] - boxes1[:, :, 0] + 1)
    area2 = (boxes2[:, :, 3] - boxes2[:, :, 1] + 1) * (boxes2[:, :, 2] - boxes2[:, :, 0] + 1)
    lt = np.maximum(boxes1[:, :, None, :2], boxes2[:, None, :, :2])  # [B, N,M,2]
    rb = np.minimum(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])  # [B, N,M,2]
    wh = (rb - lt) * (rb - lt>=0)  # [B, N,M,2]
    inter = wh[:, :, :, 0] * wh[:, :, :, 1]  # [B, N,M]
    union = area1[:, :, None] + area2[:, None, :] - inter
    iou = inter / union
    return iou

def cal_recall(predictions, k=100):
    if k is not None:
        predictions = predictions[:k]
    pred_sub_cls, pred_obj_cls, pred_rel_cls = [], [], []
    pred_sub_boxes, pred_obj_boxes = [], []
    for tri in predictions:
        pred_rel_cls.append(tri['rel_name'])
        pred_sub_cls.append(tri['sub_cls'])
        pred_obj_cls.append(tri['obj_cls'])
        pred_sub_boxes.append(tri['sub_box'])
        pred_obj_boxes.append(tri['obj_box'])
    pred_sub_cls = np.array(pred_sub_cls, dtype=np.int)
    pred_obj_cls = np.array(pred_obj_cls, dtype=np.int)
    pred_rel_cls = np.array(pred_rel_cls, dtype=np.int)
    pred_sub_boxes = np.array(pred_sub_boxes, dtype=np.float32)
    pred_obj_boxes = np.array(pred_obj_boxes, dtype=np.float32)
    return pred_sub_cls, pred_obj_cls, pred_rel_cls, \
        pred_sub_boxes, pred_obj_boxes

def cal_hit(gt_labels, predictions, k=100):
    pred_sub_cls, pred_obj_cls, pred_rel_cls, \
        pred_sub_boxes, pred_obj_boxes = cal_recall(predictions, k=k)
    gt_sub_cls, gt_obj_cls, gt_rel_cls, \
        gt_sub_boxes, gt_obj_boxes = cal_recall(gt_labels, k=None)
    
    pred_sub_cls = pred_sub_cls.reshape(-1, 1)
    pred_obj_cls = pred_obj_cls.reshape(-1, 1)
    pred_rel_cls = pred_rel_cls.reshape(-1, 1)
    hit_mat = (pred_sub_cls == gt_sub_cls) & \
        (pred_obj_cls == gt_obj_cls) & (pred_rel_cls == gt_rel_cls)
    
    pred_sub_boxes = torch.from_numpy(pred_sub_boxes).unsqueeze(0)
    pred_obj_boxes = torch.from_numpy(pred_obj_boxes).unsqueeze(0)
    gt_sub_boxes = torch.from_numpy(gt_sub_boxes).unsqueeze(0)
    gt_obj_boxes = torch.from_numpy(gt_obj_boxes).unsqueeze(0)
    siou = batch_box_iou(pred_sub_boxes, gt_sub_boxes).squeeze(0).data.cpu().numpy()
    oiou = batch_box_iou(pred_obj_boxes, gt_obj_boxes).squeeze(0).data.cpu().numpy()
    hit_mat = hit_mat & (siou >= 0.5) & (oiou >= 0.5)
    hit_id = np.where(hit_mat.sum(0) > 0)[0]
    hit_gt_labels = []
    for i in hit_id:
        hit_gt_labels.append(gt_labels[i])
    return hit_gt_labels
    
def get_predictions_in_current_frame(cur_frame_id, predictions, file_name):
    predictions_in_current_frame_list = []
    score_list = []
    for triplet in predictions[file_name]:
        st, ed = triplet['duration']
        sub_traj = triplet['sub_traj']
        obj_traj = triplet['obj_traj']
        if cur_frame_id < st or cur_frame_id >= ed: continue
        sub_box = sub_traj[cur_frame_id]
        obj_box = obj_traj[cur_frame_id]
        sub_cls, rel_name, obj_cls = triplet['triplet']
        score = triplet['score']
        score_list.append(score)
        predictions_in_current_frame_list.append(
            dict(sub_cls=sub_cls, sub_box=sub_box, 
                obj_cls=obj_cls, obj_box=obj_box, 
                rel_name=rel_name))
    score_list = np.array(score_list)
    sort_id = np.argsort(score_list)[::-1]
    ans = []
    for i in sort_id: 
        ans.append(predictions_in_current_frame_list[i])
    return ans
    
def plot_one_frame(instance_dict, rel_list, cur_frame_id, predictions, file_name, img):
    predictions_in_current_frame_list = \
        get_predictions_in_current_frame(cur_frame_id, predictions, file_name)
    gt_in_current_frame_list = []
    for rel in rel_list:
        sub_id, obj_id = rel['sub_id'], rel['obj_id']
        rel_name = rel['rel_name']
        sub_cls, sub_box = instance_dict[sub_id]['cls'], instance_dict[sub_id]['box']
        obj_cls, obj_box = instance_dict[obj_id]['cls'], instance_dict[obj_id]['box']
        gt_in_current_frame_list.append(
            dict(sub_cls=sub_cls, sub_box=sub_box, 
                obj_cls=obj_cls, obj_box=obj_box, 
                rel_name=rel_name))
    hit_gt_list = cal_hit(gt_in_current_frame_list, predictions_in_current_frame_list)
    
    fig = plt.figure(figsize=(18, 12))
    ax = plt.gca()
    plt.imshow(img)
    plt.axis('off')
    det_title = plt.title('')
    plt.setp(det_title, color='b')
    for idx, gt_hit_label in enumerate(hit_gt_list):
        s_name = gt_hit_label['sub_cls']
        x, y, x1, y1 = gt_hit_label['sub_box']
        s_cx, s_cy = (x+x1)//2, (y+y1)//2 
        srect = plt.Rectangle((x,y),x1-x,y1-y, fill=False, edgecolor='b', linewidth=3)
        ax.add_patch(srect)
        ax.text(x, y, s_name,
            fontsize=font_size,
            color='white',
            bbox=dict(facecolor='orange', alpha=0.5, pad=0, edgecolor='none'))
        
        o_name = gt_hit_label['obj_cls']
        x, y, x1, y1 = gt_hit_label['obj_box']
        o_cx, o_cy = (x+x1)//2, (y+y1)//2 
        orect = plt.Rectangle((x,y),x1-x,y1-y, fill=False, edgecolor='b', linewidth=3)
        ax.add_patch(orect)
        ax.text(x, y, o_name,
            fontsize=font_size,
            color='white',
            bbox=dict(facecolor='blue', alpha=0.5, pad=0, edgecolor='none'))
        
        p_name = gt_hit_label['rel_name']
        rel_l = lines.Line2D([s_cx, o_cx], [s_cy, o_cy], color='purple', linewidth=3)
        ax.add_line(rel_l)
        lx, ly = s_cx + 8*(o_cx - s_cx) / 9, s_cy + 8*(o_cy - s_cy) / 9
        if (lx, ly) in rec_pos: rec_pos[(lx, ly)] += 10
        else: rec_pos[(lx, ly)] = 0
        d = rec_pos[(lx, ly)]
        ax.text(lx, ly + d, p_name,
                fontsize=font_size,
                color='white',
                bbox=dict(facecolor='purple', alpha=0.5, pad=0, edgecolor='none'))
    saved_file_name = (file_name + '_' +str(cur_frame_id)+'.png').replace('/', '_')
    plt.savefig(os.path.join('./', saved_file_name), bbox_inches='tight')
    plt.close(fig)
    
    
def get_gt_rel_in_current_frame(rel_list, cur_frame_id):
    ans = []
    for rel in rel_list:
        st, ed = rel['begin_fid'], rel['end_fid']
        rel_name = rel['predicate']
        sub_id, obj_id = rel['subject_tid'], rel['object_tid']
        if cur_frame_id >= st and cur_frame_id < ed:
            ans.append(dict(sub_id=sub_id, obj_id=obj_id, rel_name=rel_name))
    return ans
    
def get_label(im_path, label_dict, file_name, predictions, selected_frames=None):
    w, h = label_dict['width'], label_dict['height']
    cls_dict = dict()
    for tracklet_cls in label_dict['subject/objects']:
        cls_dict[tracklet_cls['tid']] = tracklet_cls['category']
    for frame_id, boxes_list in enumerate(label_dict['trajectories']):
        cur_frame_instance_dict = dict()
        for xyxybox in boxes_list:
            tracklet_id = xyxybox['tid']
            x1 = xyxybox['bbox']['xmin']
            y1 = xyxybox['bbox']['ymin']
            y2 = xyxybox['bbox']['ymax']
            x2 = xyxybox['bbox']['xmax']
            xyxybox_coord = [x1, y1, x2, y2]
            xyxybox_cls = cls_dict[tracklet_id]
            cur_frame_instance_dict[tracklet_id] = dict(cls=xyxybox_cls, box=xyxybox_coord)
        cur_rel_list = get_gt_rel_in_current_frame(label_dict['relation_instances'], frame_id)
        if selected_frames is None or frame_id in selected_frames:
            img = mpimg.imread(im_path+'/'+'{:06d}'.format(frame_id)+'.png')
            plot_one_frame(cur_frame_instance_dict, cur_rel_list, frame_id, predictions, file_name, img)
        
if __name__ == '__main__':
    file_name = 'ILSVRC2015_val_00094001.mp4'
    gt_file_path = './annotations/test/'+file_name.split('.')[0]+'.json'
    predictions_file_path = './ours/' + 'baseline_relation_prediction.json'
    im_path = './frames/'+file_name
    with open(gt_file_path, 'r') as f:
        label_dict = json.load(f)
        f.close()
    with open(predictions_file_path, 'r') as f:
        predictions = json.load(f)
        f.close()
    predictions=predictions['results']
    get_label(im_path, label_dict, file_name, predictions, selected_frames=None)