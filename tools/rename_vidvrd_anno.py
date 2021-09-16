import os
import h5py
import pickle
import json
import numpy as np
import math
from tqdm import tqdm
import copy
from shutil import copyfile
from collections import OrderedDict

def get_name_mapping(video_id, frame_id, cnt):
    mapped_name = '{:012d}'.format(cnt) + '.png'
    return mapped_name

def get_class_id(class_list):
    ans_dict = {}
    for i, c in enumerate(class_list):
        ans_dict[c] = i
    return ans_dict
    
def obj_class2int(s, obj_dict):
    ans = obj_dict[s]
    return ans

def pred_class2int(s, pred_dict):
    ans = pred_dict[s]
    return ans

# [ymin, ymax, xmin, xmax] to [x, y, w, h]
def box_transform(box):
    x = box[2]
    y = box[0]
    w = box[3] - box[2] + 1
    h = box[1] - box[0] + 1
    return [x, y, w, h]


def txt2json(path, txt_path, json_path):
    with open(txt_path, 'r') as f:
        s = f.read().split()
        f.close()
    with open(json_path, 'w') as fj:
        fj.write(json.dumps(s))
        fj.close()
    print(len(s), s)
    return s

def get_box_from_track(tracklet_clip, obj_list, tid, obj_class_list, video_id):
    flg = False
    if tid >= len(obj_list) or tid != obj_list[tid]['tid']:
        #print('tid != obj_list[tid][\'tid\'] ! {} != {}'.format(tid, obj_list[tid]['tid']))
        for i in obj_list:
            if tid == i['tid']:
                category = i['category']
                flg = True
                break
    else:
        flg = True
        category = obj_list[tid]['category']
    if flg is not True:
        assert False
    x1 = tracklet_clip['bbox']['xmin']
    y1 = tracklet_clip['bbox']['ymin']
    x2 = tracklet_clip['bbox']['xmax']
    y2 = tracklet_clip['bbox']['ymax']
    ans = {"category": obj_class2int(category, obj_class_list), 
            "bbox": [y1, y2, x1, x2]}
    return ans

def process_vrd_split(pred_class_list, obj_class_list, out_split='train'):
    anno_dir = 'data/vidvrd/annotations/'
    init_path = os.path.join(anno_dir, out_split)
    video_anno_list = os.listdir(init_path)
    
    pred_class_list = get_class_id(pred_class_list)
    obj_class_list = get_class_id(obj_class_list)
    
    cnt = 1
    name_map = {}
    name_list = [' ', ]
    
    new_anns = dict()
    size_dict = dict()
    for video_anno_file in tqdm(video_anno_list):
        video_pure_id = video_anno_file.split('_')[-1].split('.')[0]
        with open(os.path.join(init_path, video_anno_file), 'r') as f:
            relation_anns = json.load(f)
            f.close()
        video_id = relation_anns['video_id']
        frame_count = relation_anns['frame_count']
        fps = relation_anns['fps']
        w = relation_anns['width']
        h = relation_anns['height']
        obj_list = relation_anns['subject/objects']
        tracks_list = relation_anns['trajectories']
        rel_list = relation_anns['relation_instances']
        #sgg = [[] for i in range(frame_count)]
        #sgg = dict()
        sgg = OrderedDict()
        for rel in rel_list:
            s_tid = rel['subject_tid']
            o_tid = rel['object_tid']
            pred = pred_class2int(rel['predicate'], pred_class_list)
            st = rel['begin_fid']
            ed = rel['end_fid']
            for t in range(st, ed):
                triplet_info = dict(predicate=pred)
                cur_frame_tracks_list = tracks_list[t]
                for tracklet_clip in cur_frame_tracks_list:
                    if tracklet_clip['tid'] == s_tid:
                        triplet_info['subject'] = get_box_from_track(tracklet_clip, 
                                                    obj_list, s_tid, obj_class_list, video_id)
                    if tracklet_clip['tid'] == o_tid:
                        triplet_info['object'] = get_box_from_track(tracklet_clip, 
                                                    obj_list, o_tid, obj_class_list, video_id)
                if t in sgg:
                    sgg[t].append(triplet_info)
                else:
                    sgg[t] = [triplet_info, ]
        for t, val in sgg.items():
            mapped_name = get_name_mapping(video_id, t, cnt)
            new_anns[mapped_name] = val
            size_dict[mapped_name] = (h, w)
            
            f_frames_path = video_id + '.mp4' + '/' + '{:06d}'.format(t) + '.png' #
            name_map[f_frames_path] = cnt
            name_list.append(f_frames_path)
            cnt += 1
    
    
    if out_split == 'test': out_split = 'val'
    
    with open('data/vidvrd/annotations/new_annotations_' + out_split + '.json', 'w') as outfile:
        json.dump(new_anns, outfile)
        outfile.close()
    
    name_map_fname = 'data/vidvrd/annotations/%s_fname_mapping.json' %(out_split)
    with open(name_map_fname, 'w') as f:
        json.dump(name_map, f, sort_keys=True, indent=4)
        f.close()
    name_list_fname = 'data/vidvrd/annotations/%s_fname_list.json' %(out_split)
    with open(name_list_fname, 'w') as f:
        json.dump(name_list, f, sort_keys=True, indent=4)
        f.close()
    
    convert_anno(out_split, new_anns, obj_class_list, size_dict)
    
def convert_anno(split, vrd_anns, obj_dict, size_dict):    
    print(len(vrd_anns)) #val: 45315; train: 75345
    new_imgs = []
    new_anns = []
    ann_id = 1
    for f, anns in tqdm(vrd_anns.items()):
        im_h, im_w = size_dict[f]
        
        image_id = int(f.split('.')[0])
        new_imgs.append(dict(file_name=f, height=im_h, width=im_w, id=image_id))
        # used for duplicate checking
        bbox_set = set()
        for ann in anns:
            # "area" in COCO is the area of segmentation mask, while here it's the area of bbox
            # also need to fake a 'iscrowd' which is always 0
            s_box = ann['subject']['bbox']
            bbox = box_transform(s_box)
            if not tuple(bbox) in bbox_set:
                bbox_set.add(tuple(bbox))
                area = bbox[2] * bbox[3]
                cat = ann['subject']['category']
                new_anns.append(dict(area=area, bbox=bbox, category_id=cat, id=ann_id, image_id=image_id, iscrowd=0))
                ann_id += 1

            o_box = ann['object']['bbox']
            bbox = box_transform(o_box)
            if not tuple(bbox) in bbox_set:
                bbox_set.add(tuple(bbox))
                area = bbox[2] * bbox[3]
                cat = ann['object']['category']
                new_anns.append(dict(area=area, bbox=bbox, category_id=cat, id=ann_id, image_id=image_id, iscrowd=0))
                ann_id += 1

    new_objs = []
    for obj, i in obj_dict.items():
        new_objs.append(dict(id=i, name=obj, supercategory=obj))


    new_data = dict(images=new_imgs, annotations=new_anns, categories=new_objs)

    with open('data/vidvrd/annotations/detections_' + split + '.json', 'w') as outfile:
        json.dump(new_data, outfile)
    
if __name__ == '__main__':
    path = './data/vidvrd/annotations'
    obj_txt_path = os.path.join(path, 'object.txt')
    obj_json_path = os.path.join(path, 'objects.json')
    pred_txt_path = os.path.join(path, 'predicate.txt')
    pred_json_path = os.path.join(path, 'predicates.json')
    if not os.path.exists(obj_json_path):
        obj_class_list = txt2json(path, obj_txt_path, obj_json_path) #35
    else:
        with open(obj_json_path, 'r') as f:
            obj_class_list = json.load(f)
            f.close()
        
    if not os.path.exists(pred_json_path):
        pred_class_list = txt2json(path, pred_txt_path, pred_json_path) #132
    else:
        with open(pred_json_path, 'r') as f:
            pred_class_list = json.load(f)
            f.close()
    
    rel_test_new_anno_json_path = os.path.join(path, 'new_annotations_val.json')
    if not os.path.exists(rel_test_new_anno_json_path):
        process_vrd_split(pred_class_list, obj_class_list, out_split='test')
        
    rel_train_new_anno_json_path = os.path.join(path, 'new_annotations_train.json')
    if not os.path.exists(rel_train_new_anno_json_path):
        process_vrd_split(pred_class_list, obj_class_list, out_split='train')
      