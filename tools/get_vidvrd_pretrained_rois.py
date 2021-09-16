import os
import h5py
import pickle
import json
import numpy as np
import _init_paths
from tqdm import tqdm
import argparse

import utils.boxes as box_utils
from core.config import cfg


def get_obj_det(obj_anno_list):
    obj_det = [{'bbox': [], 'category_id': []} for _ in range(len(obj_anno_list)+1)]
    for obj_anno in obj_anno_list:
        obj_det[obj_anno['image_id']]['bbox'].append(obj_anno['bbox'])
        obj_det[obj_anno['image_id']]['category_id'].append(obj_anno['category_id'])
    for i in range(len(obj_det)):
        obj_det[i]['bbox'] = np.array(obj_det[i]['bbox'], dtype=np.float32)
        obj_det[i]['category_id'] = np.array(obj_det[i]['category_id'], dtype=np.float32)
    return obj_det
    
def get_gtlabel(obj_boxes, obj_gt_boxes, obj_gt_labels):
    xxyy_gt_box = obj_gt_boxes.copy()
    xxyy_gt_box[:, 2] += xxyy_gt_box[:, 0]
    xxyy_gt_box[:, 3] += xxyy_gt_box[:, 1]
    obj_to_gt_overlaps = box_utils.bbox_overlaps(
        obj_boxes.astype(dtype=np.float32, copy=False),
        xxyy_gt_box.astype(dtype=np.float32, copy=False)
    )
    obj_argmaxes = obj_to_gt_overlaps.argmax(axis=1)
    obj_maxes = obj_to_gt_overlaps.max(axis=1)
    
    
    fg_inds = np.where(obj_maxes >= cfg.TRAIN.FG_THRESH)[0]
    bg_inds = np.where(obj_maxes < cfg.TRAIN.BG_THRESH_LO)[0]
    obj_labels = (-1) * np.ones(len(obj_maxes), dtype=np.int32)
    obj_labels[fg_inds] = np.array(obj_gt_labels[obj_argmaxes[fg_inds]], dtype=np.int32) + 1
    obj_labels[bg_inds] = 0
    return obj_labels
    
def save_pretrain_rois(train_frame_mapping_dict, val_frame_mapping_dict, 
                        train_obj_det, test_obj_det, out_rpath='pre_processed_boxes', 
                        out_dir='data/vidvrd', anno_dir='data/vidvrd', rpath='traj_cls'):
    if not os.path.exists(os.path.join(out_dir, out_rpath)):
        os.mkdir(os.path.join(out_dir, out_rpath))
    if not os.path.exists(os.path.join(out_dir, out_rpath, 'val')):
        os.mkdir(os.path.join(out_dir, out_rpath, 'val'))
    if not os.path.exists(os.path.join(out_dir, out_rpath, 'train')):
        os.mkdir(os.path.join(out_dir, out_rpath, 'train'))
    train_list = []
    test_list = []
    init_path = os.path.join(anno_dir, 'features', rpath)
    video_anno_list = os.listdir(init_path)
    for file_name in tqdm(video_anno_list):
        anno_file_path = os.path.join(init_path, file_name)
        cur_seg_name = file_name
        frame_anno_list = os.listdir(anno_file_path)
        for frame_anno in frame_anno_list:
            all_st, all_ed = int(frame_anno.split('-')[-3]), int(frame_anno.split('-')[-2])
            if frame_anno.split('.')[-1] == 'json':
                #if all_st == 0:
                #    _mid = [all_st, (all_st + all_ed) // 2]
                #else:
                #    _mid = [(all_st + all_ed) // 2, ]
                
                #_mid = [all_st + idx for idx in [3, 11, 15, 19, 23]]
                _mid = [all_st + idx for idx in [3, 7, 11, 14, 15, 19, 23, 27]]
                    
                #to_save = [{'rois':[], 'labels_int32':[], 'score':[], 'prd_category':[]} for _ in range(all_st, all_ed)]
                to_save = [{'rois':[], 'labels_int32':[], 'score':[]} for _ in range(all_st, all_ed)]
                
                with open(os.path.join(anno_file_path, frame_anno), 'r') as f:
                    anno_list = json.load(f)
                    f.close()
                for anno in anno_list:
                    st, ed = all_st + anno['pstart'], all_st + anno['pend']
                    tracklet = anno['rois']
                    #prd_category = anno['category'] #
                    score = anno['classeme'] #
                    for frame_id in range(anno['pstart'], anno['pend']):
                        box = tracklet[frame_id]
                        x1, y1, x2, y2 = box
                        np_box = np.array([0., x1, y1, x2, y2], dtype=np.float32)
                        to_save[frame_id]['rois'].append(np_box)
                        to_save[frame_id]['score'].append(score)
                        #to_save[frame_id]['prd_category'].append(prd_category)
                
                for frame_id in _mid:
                    frame_name = cur_seg_name + '.mp4' + '/' + '{:06d}'.format(frame_id) + '.png'
                    if frame_name in train_frame_mapping_dict:
                        saved_id = str(train_frame_mapping_dict[frame_name]) + '.pkl'
                        frame_mapped_id = train_frame_mapping_dict[frame_name]
                        out_split = 'train'
                        obj_det_list = train_obj_det
                    elif frame_name in val_frame_mapping_dict:
                        saved_id = str(val_frame_mapping_dict[frame_name]) + '.pkl'
                        frame_mapped_id = val_frame_mapping_dict[frame_name]
                        out_split = 'val'
                        obj_det_list = test_obj_det
                    else:
                        #print('No annotation frame {} !'.format(frame_name))
                        continue
                
                    frame_id = frame_id - all_st
                    val_dict = to_save[frame_id]
                    val_dict['rois'] = np.vstack(val_dict['rois'])
                    val_dict['score'] = np.vstack(val_dict['score'])
                    val_dict['score'] = np.hstack((np.zeros(len(val_dict['score'])).reshape(-1,1), val_dict['score']))
                    #val_dict['prd_category'] = np.array(val_dict['prd_category'], dtype=np.int32) + 1
                    objlabels = get_gtlabel(val_dict['rois'][:, 1:], 
                                    obj_det_list[frame_mapped_id]['bbox'], 
                                    obj_det_list[frame_mapped_id]['category_id'])
                    val_dict['labels_int32'].append(objlabels)
                    
                    if not os.path.exists(os.path.join(out_dir, out_rpath, out_split, saved_id)):
                        with open(os.path.join(out_dir, out_rpath, out_split, saved_id), 'wb') as f: #save
                            pickle.dump(val_dict, f, pickle.HIGHEST_PROTOCOL)
                            f.close()
                    else:
                        raise Exception
            elif frame_anno.split('.')[-1] == 'h5':
                raise Exception
            else: raise Exception

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="get_vidvrd_pretrained_rois")
    parser.add_argument("--out_dir", default="detection_models/vidvrd",
                        help="out_dir.")
    parser.add_argument("--out_rpath", default="pre_processed_boxes",
                        help="out_rpath.")
    parser.add_argument("--rpath", default="traj_cls",
                        help="rpath.")
    args = parser.parse_args()
    
    
    with open(os.path.join('data/vidvrd/annotations', 'detections_val.json'), 'r') as f:
        test_obj_anno_list = json.load(f)
        f.close()
    with open(os.path.join('data/vidvrd/annotations', 'detections_train.json'), 'r') as f:
        train_obj_anno_list = json.load(f)
        f.close()
    with open(os.path.join('data/vidvrd/annotations', 'val_fname_mapping.json'), 'r') as f:
        val_frame_mapping_dict = json.load(f)
        f.close()    
    with open(os.path.join('data/vidvrd/annotations', 'train_fname_mapping.json'), 'r') as f:
        train_frame_mapping_dict = json.load(f)
        f.close()    
    train_obj_det = get_obj_det(train_obj_anno_list['annotations'])
    test_obj_det = get_obj_det(test_obj_anno_list['annotations'])
    
    #save_pretrain_rois(train_frame_mapping_dict, val_frame_mapping_dict, 
    #                    train_obj_det, test_obj_det, out_rpath='pre_processed_boxes_gt',               
    #                    out_dir='detection_models/vidvrd', anno_dir='data/vidvrd', rpath='traj_cls_gt')
                        
    save_pretrain_rois(train_frame_mapping_dict, val_frame_mapping_dict, 
                        train_obj_det, test_obj_det, out_rpath=args.out_rpath,
                        out_dir=args.out_dir, anno_dir='data/vidvrd', rpath=args.rpath)
                        
    
                        