import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import random
import time
import torchvision
import cv2

import dlib

import multiprocessing

def track_frames_from_featmap(img_list, bboxes, tracker_type='dsst', is_reverse=False):
    """
        illustration: tracking in one batch
        
        img_list: torch.tensor list: [img_0, img_1, ...]. img_i: cv2.imread + cv2.COLOR_BGR2RGB
        # frames_path: one video's path
        # frames_list: list: [frame_id0, frame_id1, frame_id2 ...], frame_id1 - frame_id0 = 15
        bboxes: torch.tensor: (xmin, ymin, w, h). size:(bbox_num_in_one_batch, 5)
        
        return:
            [[(x1,y1,x2,y2), (x1,y1,x2,y2), ...], [...], [...], ...].    : obj_num, segment_len, 4
    """
    obj_tracking_list = list()
    
    if is_reverse:
        anchor_frames = img_list[::-1]
    else:
        anchor_frames = img_list

    anchor_frames = anchor_frames[::3]
    for j in range(len(bboxes)):
        bbox = [bboxes[j, 0].item(), bboxes[j, 1].item(), bboxes[j, 2].item(), bboxes[j, 3].item()]
        tracklet_bboxes = list()
        tracking_bboxes = tracker(anchor_frames, tuple(bbox), is_multi_tracker=False, tracker_type=tracker_type)
        for i, each_track_bbox in enumerate(tracking_bboxes):
            if i + 1 < len(tracking_bboxes):
                pre_anchor = each_track_bbox
                back_anchor = tracking_bboxes[i + 1]

                dis_bbox = list()
                for bi in range(4):
                    dis_bbox.append((back_anchor[bi] - pre_anchor[bi]) / 3)
                first_bbox = list()
                second_bbox = list()

                for di in range(4): 
                    first_bbox.append(pre_anchor[di] + dis_bbox[di])
                    second_bbox.append(back_anchor[di] - dis_bbox[di])
                tracklet_bboxes.append(pre_anchor)
                tracklet_bboxes.append(tuple(first_bbox))
                tracklet_bboxes.append(tuple(second_bbox))
            else:
                if i + 1 == len(tracking_bboxes):
                    tracklet_bboxes.append(tracking_bboxes[i])
                    tracklet_bboxes.append(tracklet_bboxes[-1])
                    tracklet_bboxes.append(tracklet_bboxes[-1])
                else:
                    tracklet_bboxes.append(tracking_bboxes[i])
                    tracklet_bboxes.append(tracking_bboxes[i])
                    tracklet_bboxes.append(tracking_bboxes[i])

        obj_tracking_list.append(tracklet_bboxes)
    return obj_tracking_list




def track_frames(frames_path, anchor_bboxes, start_frame, max_frame_id, segment_len=30, tracker_type='KCF', is_reverse=False, is_multi_tracker=False, is_collective=False):
    """
    frames_path: one video's path
    # frames_list: list: [frame_id0, frame_id1, frame_id2 ...], frame_id1 - frame_id0 = 15
    # anchor_bboxes: [{class0: [bbox0, bbox1, ...], class1: [bbox2, bbox3, ...]}, {...}, ...]. len(anchor_bboxes) == len(frames_list). box: {'score':..., 'bbox':[x,y,w,h]}
    anchor_bboxes: {class0: [bbox0, bbox1, ...], class1: [bbox2, bbox3, ...]}. box: {'score':..., 'bbox':[x,y,w,h]}
    """
    obj_tracking_list = list()
    each_frame = start_frame
    
    anchor_frames = list()
    
    if is_reverse:
        next_each_frame = max(1, start_frame - segment_len)
        for frame_idx in range(int(next_each_frame)-1, int(each_frame)-1, -1):
            if is_collective:
                anchor_frames.append(cv2.imread(os.path.join(frames_path, 'frame' + str(frame_idx).zfill(4) + '.jpg')))
            else:
                anchor_frames.append(cv2.imread(os.path.join(frames_path, str(frame_idx).zfill(6) + '.png')))
    else:
        next_each_frame = min(max_frame_id, start_frame + segment_len)
        for frame_idx in range(int(each_frame), int(next_each_frame)):
            if is_collective:
                anchor_frames.append(cv2.imread(os.path.join(frames_path, 'frame' + str(frame_idx).zfill(4) + '.jpg')))
            else:
                anchor_frames.append(cv2.imread(os.path.join(frames_path, str(frame_idx).zfill(6) + '.png')))

    anchor_frames = anchor_frames[::3]
    if is_multi_tracker is False:
        for each_class, bboxes in anchor_bboxes.items():
            for each_bbox in bboxes:
                score = each_bbox['score']
                bbox = each_bbox['bbox']
                tracklet_bboxes = list()
                tracking_bboxes = tracker(anchor_frames, tuple(bbox), is_multi_tracker=is_multi_tracker, tracker_type=tracker_type)
                for i, each_track_bbox in enumerate(tracking_bboxes):
                    if i + 1 < len(tracking_bboxes):
                        pre_anchor = each_track_bbox
                        back_anchor = tracking_bboxes[i + 1]

                        dis_bbox = list()
                        for bi in range(4): 
                            dis_bbox.append((back_anchor[bi] - pre_anchor[bi]) / 3)
                        first_bbox = list()
                        second_bbox = list()
                        
                        for di in range(4): 
                            first_bbox.append(pre_anchor[di] + dis_bbox[di])
                            second_bbox.append(back_anchor[di] - dis_bbox[di])
                        tracklet_bboxes.append(pre_anchor)
                        tracklet_bboxes.append(tuple(first_bbox))
                        tracklet_bboxes.append(tuple(second_bbox))
                    else:
                        if i + 1 == len(tracking_bboxes):
                            tracklet_bboxes.append(tracking_bboxes[i])
                            tracklet_bboxes.append(tracklet_bboxes[-1])
                            tracklet_bboxes.append(tracklet_bboxes[-1])
                        else:
                            tracklet_bboxes.append(tracking_bboxes[i])
                            tracklet_bboxes.append(tracking_bboxes[i])
                            tracklet_bboxes.append(tracking_bboxes[i])

                obj_tracking_list.append({
                    'obj_cls': each_class,
                    'start_frame': int(each_frame),
                    'score': score,
                    'tracklet': tracklet_bboxes
                })
    else:
        multiTracker = cv2.MultiTracker_create()
        bboxes = []
        bbox_score = []
        bbox_class = []
        for each_class, bboxes_per_class in anchor_bboxes.items():
            for each_bbox in bboxes_per_class:
                score = each_bbox['score']
                bbox = each_bbox['bbox']
                bboxes.append(bbox)
                bbox_score.append(score)
                bbox_class.append(each_class)
        
        init_frame = anchor_frames[0]
        for bbox in bboxes:
            multiTracker.add(tracker_select(is_multi_tracker=is_multi_tracker, tracker_type=tracker_type), init_frame, tuple(bbox))
        
        tracklet_bboxes = []
        for j, sampled_each_frame in enumerate(anchor_frames):
            if j == 0:
                tracklet_bboxes.append(np.array(bboxes))
            else:
                ok, bbox = multiTracker.update(sampled_each_frame)
                if ok:
                    tracklet_bboxes.append(bbox)
        obj_tracking_list = {
            'obj_tracking': tracklet_bboxes,
            'bboxes_class': bbox_class,
            'bbox_score': bbox_score,
            
        }
    
    return obj_tracking_list
    


def tracker_select(is_multi_tracker=True, tracker_type='KCF'):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    if tracker_type == 'dsst' and (is_multi_tracker is False):
        tracker = dlib.correlation_tracker()
    elif tracker_type not in tracker_types:
        tracker_type = tracker_types[2]
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    return tracker
    
def tracker(frames, init_bbox, is_multi_tracker=False, tracker_type='KCF'): 
    tracker = tracker_select(is_multi_tracker=is_multi_tracker, tracker_type=tracker_type)

    bboxes = list()
    if tracker_type == 'dsst':
        first_bbox = list(init_bbox)
        first_bbox[2] = first_bbox[2] + first_bbox[0]
        first_bbox[3] = first_bbox[3] + first_bbox[1]
        for i, each_frame in enumerate(frames):
            img = cv2.cvtColor(each_frame,cv2.COLOR_BGR2RGB)
            if i == 0:
                tracker.start_track(img, dlib.rectangle(first_bbox[0], first_bbox[1], first_bbox[2], first_bbox[3]))
                bboxes.append(init_bbox)
            else:
                tracker.update(img)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                bboxes.append((startX, startY, endX-startX, endY-startY))
    else:
        init_frame = frames[0]
        ok = tracker.init(init_frame, init_bbox)
        if not ok:
            print("Cannot initiate!")
            
        for i, each_frame in enumerate(frames):
            if i == 0:
                bboxes.append(init_bbox)
            else:
                ok, bbox = tracker.update(each_frame)
                if ok:
                    bboxes.append(bbox)

    return bboxes    

    
    