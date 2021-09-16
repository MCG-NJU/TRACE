#!/usr/bin/env python
# coding: utf-8

# In[23]:


import json
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import copy
from shutil import copyfile
import pickle
import cv2


# [ymin, ymax, xmin, xmax] to [x, y, w, h]
def box_transform(box):
    x = box[2]
    y = box[0]
    w = box[3] - box[2] + 1
    h = box[1] - box[0] + 1
    return [x, y, w, h]

def get_class_id(file_path):
    with open(file_path, 'r') as f:
        class_list = json.load(f)
        f.close()
    
    ans_dict = {}
    for i, c in enumerate(class_list):
        ans_dict[c] = i
        
    return ans_dict

def get_split_set(path):
    with open(path, 'r') as f:
        ans = json.load(f)
        f.close()
        
    ans = set(ans)
    return ans
    
def obj_class2int(s, obj_dict):
    s_no_split = s.split('/')
    ss = ''
    for v in s_no_split:
        ss = ss + v
    ans = obj_dict[ss]
    return ans

def pred_class2int(s, pred_dict):
    s_no_split = s.split('_')
    ss = ''
    for v in s_no_split:
        ss = ss + v
    ans = pred_dict[ss]
    return ans


# take the images from the sg_dataset folder and rename them

def process_vrd_split(in_split, out_split, split_set, obj_dict, pred_dict):
    ag_dir = 'data/ag/frames/'
    
    # load the original annotations OF BOTH TRAIN AND TEST DATASET!
    with open('data/ag/annotations/object_bbox_and_relationship.pkl', 'rb') as f:
        relation_anns = pickle.load(f)
        f.close()
    with open('data/ag/annotations/person_bbox.pkl', 'rb') as f:
        person_bbox_anns = pickle.load(f)
        f.close()
    
    
    cnt = 1
    name_map = {}
    name_list = [' ', ]
    for f in tqdm(sorted(os.listdir(ag_dir))):
        f_prefix = f.split('.')[0]
        if f_prefix not in split_set:
            continue
        f_frames_dir = os.path.join(ag_dir, f)
        for f_frames in sorted(os.listdir(f_frames_dir)):
            
            f_frames_path = f + '/' + f_frames
            
            if person_bbox_anns[f_frames_path]['bbox'].shape[0] <= 0:
                continue
            
            flg = False
            v = relation_anns[f_frames_path]
            for obj_info in v:
                t_flg = False
                if obj_info['attention_relationship'] is not None:
                    flg = True
                    t_flg = True
                if obj_info['spatial_relationship'] is not None:
                    flg = True
                    t_flg = True
                if obj_info['contacting_relationship'] is not None:
                    flg = True
                    t_flg = True
                
                if t_flg is False and obj_info['bbox'] is not None:
                    print(f_frames_path)
                    print(v)
                    assert False, 'some bbox have no relation!'
                
            if flg is False:
                continue
            
            name_map[f_frames_path] = cnt
            name_list.append(f_frames_path)
            cnt += 1

    print(len(name_map))
    fname_list = name_list
    
    name_list_fname = 'data/ag/annotations/%s_fname_list.json' %(out_split)
    #with open(name_list_fname, 'w') as f:
    #    json.dump(name_list, f, sort_keys=True, indent=4)
    #    f.close()
    
    # store the filename mappings here
    name_map_fname = 'data/ag/annotations/%s_fname_mapping.json' %(out_split)
    #with open(name_map_fname, 'w') as f:
    #    json.dump(name_map, f, sort_keys=True, indent=4)
    #    f.close()


    def get_triplet(v, person_bbox_anns, rel_class, obj_dict, pred_dict):
        # bbox: [ymin, ymax, xmin, xmax] !!!
        triplet_info = {}
        triplet_info['predicate'] = pred_class2int(rel_class, pred_dict)
        #triplet_info['object'] = {"category": obj_class2int(v['class'], obj_dict), "bbox": [int(x) for x in v['bbox']]}
        x = v['bbox'][0]
        y = v['bbox'][1]
        w = v['bbox'][2]
        h = v['bbox'][3]
        triplet_info['object'] = {"category": obj_class2int(v['class'], obj_dict), "bbox": [int(y), int(y+h), int(x), int(x+w)]}
        if person_bbox_anns['bbox_mode'] == 'xyxy':
            try:
                x = person_bbox_anns['bbox'][0, 0]
                y = person_bbox_anns['bbox'][0, 1]
                x1 = person_bbox_anns['bbox'][0, 2]
                y1 = person_bbox_anns['bbox'][0, 3]
                w = x1 - x
                h = y1 - y
            except Exception:
                print(person_bbox_anns)
        else:
            assert False
        #triplet_info['subject'] = {"category": 0, "bbox": [int(x), int(y), int(w), int(h)]}
        triplet_info['subject'] = {"category": 0, "bbox": [int(y), int(y1), int(x), int(x1)]}
        return triplet_info
    
    
    new_anns = {}
    for k, v in tqdm(relation_anns.items()):
        k_prefix = k.split('/')[0]
        k_prefix = k_prefix.split('.')[0]
        if k_prefix not in split_set:
            continue
        
        if person_bbox_anns[k]['bbox'].shape[0] <= 0:
            continue

        v_list = []
        for obj_info in v:
            if obj_info['attention_relationship'] is not None:
                for rel_class in obj_info['attention_relationship']:
                    tri = get_triplet(obj_info, person_bbox_anns[k], rel_class, obj_dict, pred_dict)
                    v_list.append(tri)
            if obj_info['spatial_relationship'] is not None:        
                for rel_class in obj_info['spatial_relationship']:
                    tri = get_triplet(obj_info, person_bbox_anns[k], rel_class, obj_dict, pred_dict)
                    v_list.append(tri)
            if obj_info['contacting_relationship'] is not None:        
                for rel_class in obj_info['contacting_relationship']:
                    tri = get_triplet(obj_info, person_bbox_anns[k], rel_class, obj_dict, pred_dict)
                    v_list.append(tri)
        
        if len(v_list) == 0:
            continue
        
        new_k = '{:012d}'.format(name_map[k]) + '.png'
        new_anns[new_k] = v_list

    vrd_anns = new_anns
    # create the new annotations 
    #with open('data/ag/annotations/new_annotations_' + out_split + '.json', 'w') as outfile:
    #    json.dump(new_anns, outfile)

    category_counter = convert_anno(out_split, vrd_anns, obj_dict, fname_list)
    return category_counter
        
def convert_anno(split, vrd_anns, obj_dict, fname_list):
    img_dir = 'data/ag/frames'
    print(len(vrd_anns))

    new_imgs = []
    new_anns = []
    ann_id = 1
    
    category_counter = [0 for i in range(36)]
    
    for f, anns in tqdm(vrd_anns.items()):
        #im_w, im_h = 720, 480
        
        f_path = fname_list[int(f.split('.')[0])]
        f_path = os.path.join(img_dir, f_path)
        im_h, im_w, C = 1, 1, 3
        
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
                
                category_counter[cat] += 1
                
                new_anns.append(dict(area=area, bbox=bbox, category_id=cat, id=ann_id, image_id=image_id, iscrowd=0))
                ann_id += 1

            o_box = ann['object']['bbox']
            bbox = box_transform(o_box)
            if not tuple(bbox) in bbox_set:
                bbox_set.add(tuple(bbox))
                area = bbox[2] * bbox[3]
                cat = ann['object']['category']
                
                category_counter[cat] += 1
                
                new_anns.append(dict(area=area, bbox=bbox, category_id=cat, id=ann_id, image_id=image_id, iscrowd=0))
                ann_id += 1

    new_objs = []
    for obj, i in obj_dict.items():
        new_objs.append(dict(id=i, name=obj, supercategory=obj))


    new_data = dict(images=new_imgs, annotations=new_anns, categories=new_objs)

    #with open('data/ag/annotations/detections_' + split + '.json', 'w') as outfile:
    #    json.dump(new_data, outfile)

    for i in range(0, 36):
        print('{} : {}'.format(i, category_counter[i]))
    return category_counter
        
if __name__ == '__main__':

    # using the test split as our val. We won't have a true test split for ag
    
    obj_dict = get_class_id('data/ag/annotations/objects.json')
    pred_dict = get_class_id('data/ag/annotations/predicates.json')
    
    test_split = get_split_set('data/ag/annotations/test_videos_list.json')
    train_split = get_split_set('data/ag/annotations/train_videos_list.json')
    
    category_counter_test = process_vrd_split('test', 'val', test_split, obj_dict, pred_dict)
    category_counter_train = process_vrd_split('train', 'train', train_split, obj_dict, pred_dict)
    
    freq_num = [0 for i in range(36)]
    for i in range(0, 36):
        freq_num[i] = category_counter_test[i]+category_counter_train[i]
    with open('data/ag/annotations/freq_num_list.json', 'w') as f:
        json.dump(freq_num, f)
        f.close()