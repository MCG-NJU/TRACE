# Adapted by Ji Zhang in 2019
#
# Based on Detectron.pytorch/lib/roi_data/minibatch.py written by Roy Tseng

import numpy as np
import cv2
import os

from core.config import cfg
import utils.blob as blob_utils
import roi_data.rpn


def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data', 'all_frames', 'bf_cur_len', 'f_scale']
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster R-CNN
        blob_names += roi_data.rpn.get_rpn_blob_names(is_training=is_training)
    elif cfg.RETINANET.RETINANET_ON:
        raise NotImplementedError
    else:
        # Fast R-CNN like models trained on precomputed proposals
        blob_names += roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=is_training
        )
    return blob_names


def get_minibatch(roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob, im_scales, all_frames_blob, bf_cur_len, f_scale = _get_image_blob(roidb)
    blobs['data'] = im_blob
    blobs['all_frames'] = all_frames_blob
    blobs['bf_cur_len'] = bf_cur_len
    blobs['f_scale'] = f_scale
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster/Mask R-CNN
        valid = roi_data.rpn.add_rpn_blobs(blobs, im_scales, roidb)
    elif cfg.RETINANET.RETINANET_ON:
        raise NotImplementedError
    else:
        # Fast R-CNN like models trained on precomputed proposals
        valid = roi_data.fast_rcnn.add_fast_rcnn_blobs(blobs, im_scales, roidb)
    
    # add relpn blobs
    add_relpn_blobs(blobs, im_scales, roidb)
    return blobs, valid


def add_relpn_blobs(blobs, im_scales, roidb):
    
    assert 'roidb' in blobs
    valid_keys = ['dataset_name',
                  'sbj_gt_boxes', 'sbj_gt_classes', 'obj_gt_boxes', 'obj_gt_classes', 'prd_gt_classes',
                  'sbj_gt_overlaps', 'obj_gt_overlaps', 'prd_gt_overlaps', 'pair_to_gt_ind_map',
                  'width', 'height', 'file_name', 'pre_processed_temporal_roi', 'pre_processed_frames_rpn_ret'] ###!!!
    for i, e in enumerate(roidb):
        for k in valid_keys:
            if k in e:
                blobs['roidb'][i][k] = e[k]
    
    # Always return valid=True, since RPN minibatches are valid by design
    return True


def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    im_scales = []
    
    #roidb_file_name = []
    for i in range(num_images):
        #im = cv2.imread(roidb[i]['image'])
        im = cv2.imread(roidb[i]['image'], cv2.IMREAD_COLOR)
        #print(roidb[i]['image'], im.shape)
        #roidb_file_name.append(int(roidb[i]['file_name'].split('.')[0]))
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        # If NOT using opencv to read in images, uncomment following lines
        # if len(im.shape) == 2:
        #     im = im[:, :, np.newaxis]
        #     im = np.concatenate((im, im, im), axis=2)
        # # flip the channel, since the original one using cv2
        # # rgb -> bgr
        # im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)
    
    if (len(cfg.TRAIN.DATASETS) > 0 and \
        cfg.TRAIN.DATASETS[0].find('vidvrd') >= 0) or \
        (len(cfg.TEST.DATASETS) > 0 and \
        cfg.TEST.DATASETS[0].find('vidvrd') >= 0):
        im_id_st = 0
    elif (len(cfg.TRAIN.DATASETS) > 0 and \
        cfg.TRAIN.DATASETS[0].find('ag') >= 0) or \
        (len(cfg.TEST.DATASETS) > 0 and \
        cfg.TEST.DATASETS[0].find('ag') >= 0):
        im_id_st = 1
    else:
        im_id_st = 1
    all_frames_blob, bf_cur_len, f_scale = get_frames_blob(roidb, \
        num_images, scale_inds, im_id_st=im_id_st, half_frame_relative_path=cfg.HALF_FRAME_RELATIVE_PATH) ###!
    #print(blob.shape, all_frames_blob.shape)
    return blob, im_scales, all_frames_blob, bf_cur_len, f_scale

def get_frames_blob(roidb, num_images, scale_inds, im_id_st=1, half_frame_relative_path=''):    
    all_frames_blob = []
    bf_cur_len = []
    f_scale = []
    if half_frame_relative_path == 'sampled_frames': sufix_class = '.jpg'
    else: sufix_class = '.png'
    for i in range(num_images):
        frame_full_name = roidb[i]['image'].split('/')[-1]
        frame_id = int(frame_full_name.split('.')[0])
        tot_video_path_list = roidb[i]['image'].split('/')
        video_path_list = tot_video_path_list[:-3]
        video_path = '/'
        for j in video_path_list:
            video_path = os.path.join(video_path, j)
        #video_path = os.path.join(video_path, 'all_frames')
        video_path = os.path.join(video_path, half_frame_relative_path) ###!!!
        video_path = os.path.join(video_path, tot_video_path_list[-2])
        
        processed_frames = []
        start_f_id = frame_id - (cfg.HALF_NUMBER_OF_FRAMES + 1) * cfg.FRAMES_INTERVAL
        end_f_id = frame_id + (cfg.HALF_NUMBER_OF_FRAMES + 1) * cfg.FRAMES_INTERVAL
        
        process_frames_id = []
        for j in range(frame_id, start_f_id, -cfg.FRAMES_INTERVAL):
            if j < im_id_st:
                break
            process_frames_id.append(j)
        process_frames_id = process_frames_id[::-1]
        process_frames_id = process_frames_id[:-1]
        for j in range(frame_id, end_f_id, cfg.FRAMES_INTERVAL):
            process_frames_id.append(j)    
        
        off_set_f = 0
        off_set_b = cfg.HALF_NUMBER_OF_FRAMES
        k = 0
        for cnt, j in enumerate(process_frames_id):
            if j < im_id_st:
                continue
            
            frame_path = os.path.join(video_path, '{:06d}'.format(j)+sufix_class)
            if j == frame_id:
                off_set_f = k
                k = 0
                #k = k+1 #
                #continue #
                frame_path = roidb[i]['image']
            
            if os.path.exists(frame_path):
                im = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                if roidb[i]['flipped']:
                    im = im[:, ::-1, :]
                #target_size = cfg.TRAIN.SCALES[scale_inds[i]]    
                target_size = cfg.TEMPORAL_SCALES   
                im, f_scale_i = blob_utils.prep_im_for_blob(
                    im, cfg.PIXEL_MEANS, [target_size], 1000)    
                processed_frames.append(im[0])   
                k = k + 1
            else:
                off_set_b = k - 1
                break
        
        st = cfg.HALF_NUMBER_OF_FRAMES - off_set_f
        ed = cfg.HALF_NUMBER_OF_FRAMES + off_set_b
        if cfg.FPN.REL_FPN_ON:
            frames_blob = blob_utils.im_list_to_blob(processed_frames)
        else:
            #frames_blob = np.stack(processed_frames)
            frames_blob = np.array(processed_frames, dtype=np.float32)
            channel_swap = (0, 3, 1, 2)
            frames_blob = frames_blob.transpose(channel_swap)
            
        
        
        
        ##got_frames = np.zeros((2*cfg.HALF_NUMBER_OF_FRAMES+1, frames_blob.shape[1], frames_blob.shape[2], frames_blob.shape[3]), dtype=np.float32)
        ##got_frames[st:ed] = frames_blob.astype(np.float32)
        
        pad_st = max(0, st)
        #pad_ed = max(0, 2*cfg.HALF_NUMBER_OF_FRAMES + 1 - ed)
        pad_ed = max(0, 2*cfg.HALF_NUMBER_OF_FRAMES - ed) #
        
        
        
        f_scale.append(f_scale_i[0])
        
        if (pad_st == 0 and pad_ed == 0) or num_images == 1:
            got_frames = frames_blob
        elif num_images != 1:
            got_frames = np.pad(frames_blob, ((pad_st,pad_ed), (0,0), (0,0), (0,0)),'constant',constant_values=0)
            
        if num_images != 1:
            bf_cur_len.append(cfg.HALF_NUMBER_OF_FRAMES)
            all_frames_blob.append(got_frames)
        else:
            bf_cur_len.append(off_set_f)
            all_frames_blob = np.expand_dims(got_frames, axis=0)
    
    if num_images != 1:
        all_frames_blob = np.stack(all_frames_blob)
    bf_cur_len = np.array(bf_cur_len, dtype=np.int32)
    f_scale = np.array(f_scale, dtype=np.float32)
    
    return all_frames_blob, bf_cur_len, f_scale