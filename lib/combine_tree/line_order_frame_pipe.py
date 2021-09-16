import numpy as np
import torch
import torch.nn as nn

class line_order_frame_pipe:
    def __init__(self, tot_length=21, max_time=18):
        self.tot_length = tot_length
        self.max_time = max_time
        self.time_count = dict()
        self.img_pipe = dict()
        
    def saving_frame(self, new_img_id, new_img):
        del_list = []
        cur_max_time = 0
        cur_max_time_id = ''
        for frame_id in self.time_count:
            self.time_count[frame_id] += 1
            if self.time_count[frame_id] >= self.max_time:
                del_list.append(frame_id)
            elif cur_max_time <= self.time_count[frame_id]:
                cur_max_time = self.time_count[frame_id]
                cur_max_time_id = frame_id
            
        if new_img_id not in self.time_count:
            self.time_count[new_img_id] = 0
            self.img_pipe[new_img_id] = new_img
        
        for i in del_list:
            del self.time_count[i]
            del self.img_pipe[i]
        
        if len(self.time_count) > self.tot_length and cur_max_time_id != '':
            del self.time_count[cur_max_time_id]
            del self.img_pipe[cur_max_time_id]
    def frame_in(self, img_id):
        if img_id not in self.time_count:
            return False
        else:
            return True
    def get_frame(self, img_id):
        if img_id not in self.time_count:
            return None
        else:
            return self.img_pipe[img_id]