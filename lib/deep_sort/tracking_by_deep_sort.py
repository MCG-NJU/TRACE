import os
import time

import cv2
import numpy as np

from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker

#from deep_sort import nn_matching
#from deep_sort.detection import Detection
#from deep_sort.tracker import Tracker

def create_detections(detection_mat, min_height=0):
    detection_list = []
    for row in detection_mat:
        bbox, feature = row[1:5], row[5:]
        confidence = 1.0
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list

def track_frames_from_featmap(img_list, input_bboxes, is_reverse=False, max_cosine_distance = 0.2, nn_budget = None, min_detection_height=1):
    """
        illustration: tracking in one batch
        
        img_list: torch.tensor list: [img_0, img_1, ...]. img_i: cv2.imread + cv2.COLOR_BGR2RGB
        # frames_path: one video's path
        # frames_list: list: [frame_id0, frame_id1, frame_id2 ...], frame_id1 - frame_id0 = 15
        bboxes: torch.tensor list: [bboxi, ...]. bboxi:(batch_id, xmin, ymin, w, h, ...)
        
        return:
            [[(x1,y1,x2,y2), (x1,y1,x2,y2), ...], [...], [...], ...].    : obj_num, segment_len, 4
    """
    #print('Better to divide the rois according to their class!!!!!!!!!!!!!!!!!')
    #print('Have Freezed the BN')
    #print('Speed')
    #print('Loss')
    obj_tracking_list = list()
    if is_reverse:
        anchor_frames = img_list[::-1]
        bboxes = input_bboxes[::-1].copy()
    else:
        anchor_frames = img_list
        bboxes = input_bboxes.copy()
        
    
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=3, n_init=1)
    #tracker = Tracker(metric, max_age=5, n_init=1)
    results = []
    res_mat = np.zeros((len(bboxes[0]), len(img_list), 4), dtype=np.float32)
    
    #b = time.time()
    
    for frame_idx, img in enumerate(anchor_frames):
        #c = time.time()
        
        ###!!!detections = create_detections(bboxes[frame_idx], min_detection_height)
        detections = bboxes[frame_idx][:, 1:]
        
        #print('c: '+str(time.time()-c))
        #o = time.time()
        
        tracker.predict()
        
        #print('o: '+str(time.time()-o))
        #oo = time.time()
        
        tracker.update(detections)
        
        #print('oo: '+str(time.time()-oo))
        #d = time.time()
        
        if frame_idx == 0:
            res_mat[:, 0, :] = bboxes[0][:, 1:5].copy()
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            
            if track.track_id < len(bboxes[0]):
                res_mat[track.track_id, frame_idx, :] = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=np.float32)
        
        #print('d: '+str(time.time()-d))
        #print('dddddddddddddddddddddddddddddddddddddddddddd')
        
    if is_reverse:
        res_mat = res_mat[:, ::-1, :]            
    
    #print('b: '+str(time.time()-b))
    
    return results, res_mat
    
    
if __name__ == '__main__':
    N = 3
    roi_num = 300
    feat_num = 2048+4
    img_list = range(0, N)
    input_bboxes = []
    for i in range(len(img_list)):
        roi = 500 * np.random.rand(roi_num, feat_num) + 1
        idx = 0 * np.ones(roi_num, dtype=np.float32).reshape(-1, 1)
        roi = np.hstack((idx, roi))
        input_bboxes.append(roi)
    
    a = time.time()
    
    track_frames_from_featmap(img_list, input_bboxes)
    track_frames_from_featmap(img_list, input_bboxes, is_reverse=True)
    print(time.time() - a)