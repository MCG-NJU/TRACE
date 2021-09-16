# Adapted by Ji Zhang for this project in 2019
#
# Based on Detectron.Pytorch/lib/roi/loader.py by Roy Tseng

import math
import numpy as np
import numpy.random as npr

import torch
import torch.utils.data as data
import torch.utils.data.sampler as torch_sampler
from torch.utils.data.dataloader import default_collate
from torch._six import int_classes as _int_classes

from core.config import cfg
from roi_data_rel.minibatch_rel import get_minibatch
import utils.blob as blob_utils


class RoiDataLoader(data.Dataset):
    def __init__(self, roidb, num_classes, training=True):
        self._roidb = roidb
        self._num_classes = num_classes
        self.training = training
        self.DATA_SIZE = len(self._roidb)

    def __getitem__(self, index_tuple):
        index, ratio = index_tuple
        single_db = [self._roidb[index]]
        blobs, valid = get_minibatch(single_db)
        #TODO: Check if minibatch is valid ? If not, abandon it.
        # Need to change _worker_loop in torch.utils.data.dataloader.py.

        # Squeeze batch dim
        for key in blobs:
            if key != 'roidb':
                blobs[key] = blobs[key].squeeze(axis=0)

        if self._roidb[index]['need_crop']:
            self.crop_data(blobs, ratio)
            # Check bounding box
            entry = blobs['roidb'][0]
            boxes = entry['boxes']
            invalid = (boxes[:, 0] == boxes[:, 2]) | (boxes[:, 1] == boxes[:, 3])
            valid_inds = np.nonzero(~ invalid)[0]
            if len(valid_inds) < len(boxes):
                for key in ['boxes', 'gt_classes', 'seg_areas', 'gt_overlaps', 'is_crowd',
                            'box_to_gt_ind_map', 'gt_keypoints']:
                    if key in entry:
                        entry[key] = entry[key][valid_inds]
                entry['segms'] = [entry['segms'][ind] for ind in valid_inds]
            # for rel sanity check
            sbj_gt_boxes = entry['sbj_gt_boxes']
            obj_gt_boxes = entry['obj_gt_boxes']
            sbj_invalid = (sbj_gt_boxes[:, 0] == sbj_gt_boxes[:, 2]) | (sbj_gt_boxes[:, 1] == sbj_gt_boxes[:, 3])
            obj_invalid = (obj_gt_boxes[:, 0] == obj_gt_boxes[:, 2]) | (obj_gt_boxes[:, 1] == obj_gt_boxes[:, 3])
            rel_valid = sbj_invalid | obj_invalid
            rel_valid_inds = np.nonzero(~ rel_invalid)[0]
            if len(rel_valid_inds) < len(sbj_gt_boxes):
                for key in ['sbj_gt_boxes', 'sbj_gt_classes', 'obj_gt_boxes', 'obj_gt_classes', 'prd_gt_classes',
                            'sbj_gt_overlaps', 'obj_gt_overlaps', 'prd_gt_overlaps', 'pair_to_gt_ind_map',
                            'width', 'height', 'file_name', 'pre_processed_temporal_roi', 'pre_processed_frames_rpn_ret']: ###!!!
                    if key in entry:
                        entry[key] = entry[key][rel_valid_inds]

        blobs['roidb'] = blob_utils.serialize(blobs['roidb'])  # CHECK: maybe we can serialize in collate_fn

        return blobs

    def crop_data(self, blobs, ratio):
        data_height, data_width = map(int, blobs['im_info'][:2])
        boxes = blobs['roidb'][0]['boxes']
        if ratio < 1:  # width << height, crop height
            size_crop = math.ceil(data_width / ratio)  # size after crop
            min_y = math.floor(np.min(boxes[:, 1]))
            max_y = math.floor(np.max(boxes[:, 3]))
            box_region = max_y - min_y + 1
            if min_y == 0:
                y_s = 0
            else:
                if (box_region - size_crop) < 0:
                    y_s_min = max(max_y - size_crop, 0)
                    y_s_max = min(min_y, data_height - size_crop)
                    y_s = y_s_min if y_s_min == y_s_max else \
                        npr.choice(range(y_s_min, y_s_max + 1))
                else:
                    # CHECK: rethinking the mechnism for the case box_region > size_crop
                    # Now, the crop is biased on the lower part of box_region caused by
                    # // 2 for y_s_add
                    y_s_add = (box_region - size_crop) // 2
                    y_s = min_y if y_s_add == 0 else \
                        npr.choice(range(min_y, min_y + y_s_add + 1))
            # Crop the image
            blobs['data'] = blobs['data'][:, y_s:(y_s + size_crop), :,] 
            blobs['all_frames'] = blobs['all_frames'][:, :, y_s:(y_s + size_crop), :,] ###!!!
            # Update im_info
            blobs['im_info'][0] = size_crop
            # Shift and clamp boxes ground truth
            boxes[:, 1] -= y_s
            boxes[:, 3] -= y_s
            np.clip(boxes[:, 1], 0, size_crop - 1, out=boxes[:, 1])
            np.clip(boxes[:, 3], 0, size_crop - 1, out=boxes[:, 3])
            blobs['roidb'][0]['boxes'] = boxes
        else:  # width >> height, crop width
            size_crop = math.ceil(data_height * ratio)
            min_x = math.floor(np.min(boxes[:, 0]))
            max_x = math.floor(np.max(boxes[:, 2]))
            box_region = max_x - min_x + 1
            if min_x == 0:
                x_s = 0
            else:
                if (box_region - size_crop) < 0:
                    x_s_min = max(max_x - size_crop, 0)
                    x_s_max = min(min_x, data_width - size_crop)
                    x_s = x_s_min if x_s_min == x_s_max else \
                        npr.choice(range(x_s_min, x_s_max + 1))
                else:
                    x_s_add = (box_region - size_crop) // 2
                    x_s = min_x if x_s_add == 0 else \
                        npr.choice(range(min_x, min_x + x_s_add + 1))
            # Crop the image
            blobs['data'] = blobs['data'][:, :, x_s:(x_s + size_crop)]
            blobs['all_frames'] = blobs['all_frames'][:, :, :, x_s:(x_s + size_crop)]
            # Update im_info
            blobs['im_info'][1] = size_crop
            # Shift and clamp boxes ground truth
            boxes[:, 0] -= x_s
            boxes[:, 2] -= x_s
            np.clip(boxes[:, 0], 0, size_crop - 1, out=boxes[:, 0])
            np.clip(boxes[:, 2], 0, size_crop - 1, out=boxes[:, 2])
            blobs['roidb'][0]['boxes'] = boxes

    def __len__(self):
        return self.DATA_SIZE


def cal_minibatch_ratio(ratio_list):
    """Given the ratio_list, we want to make the RATIO same for each minibatch on each GPU.
    Note: this only work for 1) cfg.TRAIN.MAX_SIZE is ignored during `prep_im_for_blob` 
    and 2) cfg.TRAIN.SCALES containing SINGLE scale.
    Since all prepared images will have same min side length of cfg.TRAIN.SCALES[0], we can
     pad and batch images base on that.
    """
    DATA_SIZE = len(ratio_list)
    ratio_list_minibatch = np.empty((DATA_SIZE,))
    num_minibatch = int(np.ceil(DATA_SIZE / cfg.TRAIN.IMS_PER_BATCH))  # Include leftovers
    for i in range(num_minibatch):
        left_idx = i * cfg.TRAIN.IMS_PER_BATCH
        right_idx = min((i+1) * cfg.TRAIN.IMS_PER_BATCH - 1, DATA_SIZE - 1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        ratio_list_minibatch[left_idx:(right_idx+1)] = target_ratio
    return ratio_list_minibatch


class MinibatchSampler(torch_sampler.Sampler):
    def __init__(self, ratio_list, ratio_index):
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.num_data = len(ratio_list)

        if cfg.TRAIN.ASPECT_GROUPING:
            # Given the ratio_list, we want to make the ratio same
            # for each minibatch on each GPU.
            self.ratio_list_minibatch = cal_minibatch_ratio(ratio_list)

    def __iter__(self):
        if cfg.TRAIN.ASPECT_GROUPING:
            # indices for aspect grouping awared permutation
            n, rem = divmod(self.num_data, cfg.TRAIN.IMS_PER_BATCH)
            round_num_data = n * cfg.TRAIN.IMS_PER_BATCH
            indices = np.arange(round_num_data)
            npr.shuffle(indices.reshape(-1, cfg.TRAIN.IMS_PER_BATCH))  # inplace shuffle
            if rem != 0:
                indices = np.append(indices, np.arange(round_num_data, round_num_data + rem))
            ratio_index = self.ratio_index[indices]
            ratio_list_minibatch = self.ratio_list_minibatch[indices]
        else:
            rand_perm = npr.permutation(self.num_data)
            ratio_list = self.ratio_list[rand_perm]
            ratio_index = self.ratio_index[rand_perm]
            # re-calculate minibatch ratio list
            ratio_list_minibatch = cal_minibatch_ratio(ratio_list)

        return iter(zip(ratio_index.tolist(), ratio_list_minibatch.tolist()))

    def __len__(self):
        return self.num_data


class BatchSampler(torch_sampler.BatchSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, torch_sampler.Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)  # Difference: batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size



def collate_minibatch(list_of_blobs):
    """Stack samples seperately and return a list of minibatches
    A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
    Hence, we need to stack smaples from each minibatch seperately.
    """
    Batch = {key: [] for key in list_of_blobs[0]}
    # Because roidb consists of entries of variable length, it can't be batch into a tensor.
    # So we keep roidb in the type of "list of ndarray".
    list_of_roidb = [blobs.pop('roidb') for blobs in list_of_blobs]
    for i in range(0, len(list_of_blobs), cfg.TRAIN.IMS_PER_BATCH):
        mini_list = list_of_blobs[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        # Pad image data
        mini_list = pad_image_data(mini_list)
        minibatch = default_collate(mini_list)
        minibatch['roidb'] = list_of_roidb[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        for key in minibatch:
            Batch[key].append(minibatch[key])

    return Batch


def pad_image_data(list_of_blobs):
    max_shape = blob_utils.get_max_shape([blobs['data'].shape[1:] for blobs in list_of_blobs])
    f_max_shape = blob_utils.get_max_shape([blobs['all_frames'].shape[2:] for blobs in list_of_blobs])
    output_list = []
    for blobs in list_of_blobs:
        data_padded = np.zeros((3, max_shape[0], max_shape[1]), dtype=np.float32)
        #all_frames_padded = np.zeros((2*cfg.HALF_NUMBER_OF_FRAMES+1, 3, max_shape[0], max_shape[1]), dtype=np.float32) #
        #all_frames_padded = np.zeros((2*cfg.HALF_NUMBER_OF_FRAMES, 3, max_shape[0], max_shape[1]), dtype=np.float32) # 7->6
        all_frames_padded = np.zeros((blobs['all_frames'].shape[0], 3, f_max_shape[0], f_max_shape[1]), dtype=np.float32) # 7->6
        _, h, w = blobs['data'].shape
        _, _, fh, fw = blobs['all_frames'].shape
        data_padded[:, :h, :w] = blobs['data']
        all_frames_padded[:, :, :fh, :fw] = blobs['all_frames']
        blobs['data'] = data_padded
        blobs['all_frames'] = all_frames_padded
        output_list.append(blobs)
    return output_list
