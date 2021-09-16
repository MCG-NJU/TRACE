import os
import json
import argparse
import warnings
from tqdm import tqdm


def get_sampled_frames(f_id_list, half_len, frames_store_type, st_id):
    tmp_ans = set()
    for i, f_name in enumerate(f_id_list):
        f_id = int(f_name.split('.')[0])
        tmp_ans.add(f_name)
        
        ii = 0
        st = f_id
        a1, a2, a3 = st, st, st
        while True:
            a1 -= 2
            if a1 >= st_id:
                tmp_ans.add(('{:06d}'.format(a1)) + '.' + frames_store_type)
            a2 -= 3
            if a2 >= st_id:
                tmp_ans.add(('{:06d}'.format(a2)) + '.' + frames_store_type)
            a3 -= 4
            if a3 >= st_id:
                tmp_ans.add(('{:06d}'.format(a3)) + '.' + frames_store_type)
            ii += 1
            if (a1 < st_id and a2 < st_id and a2 < st_id) or ii >= half_len:
                break
        
        ii = 0
        st = f_id
        a1, a2, a3 = st, st, st
        while True:
            a1 += 2
            tmp_ans.add(('{:06d}'.format(a1)) + '.' + frames_store_type)
            a2 += 3
            tmp_ans.add(('{:06d}'.format(a2)) + '.' + frames_store_type)
            a3 += 4
            tmp_ans.add(('{:06d}'.format(a3)) + '.' + frames_store_type)
            ii += 1
            if ii >= half_len:
                break
    tmp_ans = list(tmp_ans)
    return tmp_ans

def dump_frames(args, frame_list_file='frame_list.txt', st_id=1):
    video_dir = args.video_dir
    frame_dir = args.frame_dir
    annotation_dir = args.annotation_dir
    all_frames = args.all_frames
    frames_store_type = args.frames_store_type
    ignore_editlist = args.ignore_editlist

    # Load the list of annotated frames
    frame_list = []
    if frame_list_file.find(',')<0 and frame_list_file.split('.')[-1] == 'txt':
        with open(os.path.join(annotation_dir, frame_list_file), 'r') as f:
            for frame in f:
                frame_list.append(frame.rstrip('\n'))
            f.close()
    elif frame_list_file.find(',')<0 and frame_list_file.split('.')[-1] == 'json':
        with open(os.path.join(annotation_dir, frame_list_file), 'r') as f:
            frame_list = json.load(f)
            f.close()
        frame_list = frame_list[1:]
    elif frame_list_file.find(',')>=0:
        print('list.')
        frame_list_file = frame_list_file.split(',')
        for fl in frame_list_file:
            part_frame_list = []
            if fl.split('.')[-1] == 'json':
                with open(os.path.join(annotation_dir, fl), 'r') as f:
                    part_frame_list = json.load(f)
                    f.close()
                part_frame_list = part_frame_list[1:]
            elif fl.split('.')[-1] == 'txt':
                with open(os.path.join(annotation_dir, fl), 'r') as f:
                    for frame in f:
                        part_frame_list.append(frame.rstrip('\n'))
                    f.close()
            frame_list += part_frame_list
    else: raise Exception
    # Create video to frames mapping
    video2frames = {}
    for path in frame_list:
        video, frame = path.split('/')
        if video not in video2frames:
            video2frames[video] = []
        video2frames[video].append(frame)
    
    # For each video, dump frames.
    for v in tqdm(video2frames):
        curr_frame_dir = os.path.join(frame_dir, v)
        
        if args.sampled_frames:
            f_id_list = video2frames[v]
            tmp_ans = get_sampled_frames(f_id_list, args.half_len, frames_store_type, st_id)
            #print(float(len(tmp_ans)) / (float(len(f_id_list))+1e-12))
            if args.rid_of_half_len > 0:
                rid_tmp_ans = get_sampled_frames(f_id_list, args.rid_of_half_len, frames_store_type, st_id)
                tmp_ans = set(tmp_ans) - set(rid_tmp_ans)
                tmp_ans = list(tmp_ans)
        
        if not os.path.exists(curr_frame_dir):
            os.makedirs(curr_frame_dir)
            # Use ffmpeg to extract frames. Different versions of ffmpeg may generate slightly different frames.
            # We used **ffmpeg 2.8.15** to dump our frames.
            # Note that the frames are extracted according to their original video FPS, which is not always 24.
            # Therefore, our frame indices are different from Charades extracted frames' indices.
            if frames_store_type == 'jpg' and args.high_quality:
                if ignore_editlist:
                    os.system('ffmpeg -ignore_editlist 1 -loglevel panic -i %s/%s -start_number %d -qscale:v 2 %s/%%06d.%s' % (video_dir, v, st_id, curr_frame_dir, frames_store_type))
                else:
                    os.system('ffmpeg -loglevel panic -i %s/%s -start_number %d -qscale:v 2 %s/%%06d.%s' % (video_dir, v, st_id, curr_frame_dir, frames_store_type))

            else:
                if ignore_editlist:
                    os.system('ffmpeg -ignore_editlist 1 -loglevel panic -i %s/%s -start_number %d %s/%%06d.%s' % (video_dir, v, st_id, curr_frame_dir, frames_store_type))
                else:
                    os.system('ffmpeg -loglevel panic -i %s/%s -start_number %d %s/%%06d.%s' % (video_dir, v, st_id, curr_frame_dir, frames_store_type))
            

            # if not keeping all frames, only keep the annotated frames included in frame_list.txt
            if not all_frames:
                if args.sampled_frames:
                    keep_frames = tmp_ans
                else:
                    keep_frames = video2frames[v]
                frames_to_delete = set(os.listdir(curr_frame_dir)) - set(keep_frames)
                for frame in frames_to_delete:
                    os.remove(os.path.join(curr_frame_dir, frame))
        else:
            warnings.warn('Frame directory %s already exists. Skipping dumping into this directory.' % curr_frame_dir,
                          RuntimeWarning)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump frames")
    parser.add_argument("--video_dir", default="data/ag/videos",
                        help="Folder containing Charades videos.")
    parser.add_argument("--frame_dir", default="data/ag/frames",
                        help="Root folder containing frames to be dumped.")
    parser.add_argument("--annotation_dir", default="data/ag/annotations",
                        help=("Folder containing annotation files, including object_bbox_and_relationship.pkl, "
                              "person_bbox.pkl and frame_list.txt."))
    parser.add_argument("--all_frames", action="store_true",
                        help="Set if you want to dump all frames, rather than the frames listed in frame_list.txt")
    parser.add_argument("--frames_store_type", default="png",
                        help="The type for storing frames.")
    parser.add_argument("--ignore_editlist", action="store_true",
                        help="Old versions of ffmpeg do not support edit list, such as ffmpeg 2.8.15. Using edit list may drop some frames.")
    parser.add_argument(
        '--high_quality', help='high_quality, 6 times bigger (jpg_low: 3kb, jpg_high: 20kb, png: 170kb).',
        action='store_true')
    parser.add_argument(
        '--half_len',
        help='half_len',
        default=13, type=int)
        #default=8, type=int)
    parser.add_argument("--sampled_frames", action="store_true",
                        help="dump sampled_frames")
    parser.add_argument("--frame_list_file", default="frame_list.txt",
                        help="frame_list_file.")
    parser.add_argument("--st_id", default=1,
                        help="st_id.")
    parser.add_argument(
        '--rid_of_half_len',
        help='rid_of_half_len',
        default=-1, type=int)
    args = parser.parse_args()
    
    try:
        dump_frames(args, args.frame_list_file, int(args.st_id))
    except Exception:
        print('stop')