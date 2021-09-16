import numpy as np
import cv2
import os
import json

from core.config import cfg



def get_frames_blob(img_list_path):    
    with open(img_list_path, 'r') as f:
        img_list = json.load(f)
        f.close()
    img_list = img_list[1:]
    num_images = len(img_list)
    all_frames_path_list = set()
    for i in range(num_images):
        video_path = img_list[i].split('/')[0]
        frame_full_name = img_list[i].split('/')[-1]
        frame_id = int(frame_full_name.split('.')[0])
        
        start_f_id = frame_id - (cfg.HALF_NUMBER_OF_FRAMES + 1) * cfg.FRAMES_INTERVAL
        end_f_id = frame_id + (cfg.HALF_NUMBER_OF_FRAMES + 1) * cfg.FRAMES_INTERVAL
        
        process_frames_id = []
        for j in range(frame_id, start_f_id, -cfg.FRAMES_INTERVAL):
            if j <= 0:
                break
            process_frames_id.append(j)
        process_frames_id = process_frames_id[::-1]
        process_frames_id = process_frames_id[:-1]
        for j in range(frame_id, end_f_id, cfg.FRAMES_INTERVAL):
            process_frames_id.append(j)    
        
        for cnt, j in enumerate(process_frames_id):
            if j <= 0:
                continue
            if j == frame_id:
                frame_path = './data/ag/frames/' + img_list[i]
            else:
                frame_path = './data/ag/all_frames/' + video_path + '/' + '{:06d}'.format(j) + '.jpg'
            if os.path.exists(frame_path):
                all_frames_path_list.add(frame_path)   
            else:
                break
    all_frames_path_list = list(all_frames_path_list) 
    return all_frames_path_list
    
if __name__ == 'main':
    img_list_path = './data/ag/annotations/val_fname_list.json'
    a = get_frames_blob(img_list_path)
    img_list_path = './data/ag/annotations/train_fname_list.json'
    b = get_frames_blob(img_list_path)
    a = a + b
    with open('./data/ag/annotations/all_frames_to_get_rois_list.json', 'w') as f:
        f.write(json.dumps(a))
        f.close()