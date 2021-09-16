import os
import math
import pickle


def load_annotations(annotation_dir):
    with open(os.path.join(annotation_dir, 'object_bbox_and_relationship.pkl'), 'rb') as f:
        object_anno = pickle.load(f)

    with open(os.path.join(annotation_dir, 'person_bbox.pkl'), 'rb') as f:
        person_anno = pickle.load(f)

    frame_list = []
    with open(os.path.join(annotation_dir, 'frame_list.txt'), 'r') as f:
        for frame in f:
            frame_list.append(frame.rstrip('\n'))

    return object_anno, person_anno, frame_list

def show_person_bbox(person_anno, max_show_num=2):
    cnt = 0
    for key, val in person_anno.items():
        if cnt > max_show_num:
            break
        print(key)
        print(val)
        cnt = cnt + 1

def show_object_bbox_and_relationship(object_anno, max_show_num=2):
    cnt = 0
    for key, val in object_anno.items():
        if cnt > max_show_num:
            break
        print(key)
        print(val)
        cnt = cnt + 1

def count_object_bbox_and_relationship(object_anno):
    cnt_spatial_relationship = 0
    cnt_contacting_relationship = 0
    cnt_attention_relationship = 0
    cnt_visible = 0
    cnt_obj = 0
    cnt_visible_max = 0
    cnt_frame = 0
    flg = False
    for key, val in object_anno.items():
        cnt_frame = cnt_frame + 1
        i = 0
        cnt_obj = cnt_obj + len(val)
        for obj in val:
            if obj['visible'] is not None:
                cnt_visible = cnt_visible + 1
                i = i + 1
            if obj['spatial_relationship'] is not None:
                cnt_spatial_relationship = max(cnt_spatial_relationship, len(obj['spatial_relationship']))
                if len(obj['spatial_relationship']) > 3:
                    if flg is False:
                        flg = True
                        print(key, val)    # Q8XEE.mp4/000111.png [{'class': 'sofa/couch', 'bbox': (57.07142857142854, 226.5, 212.4332564325793, 132.5), 'attention_relationship': ['not_looking_at'], 'spatial_relationship': ['behind', 'beneath', 'on_the_side_of', 'in_front_of'], 'contacting_relationship': ['sitting_on'], 'metadata': {'tag': 'Q8XEE.mp4/sofa_couch/000111', 'set': 'train'}, 'visible': True}, {'class': 'laptop', 'bbox': (162.94264069264077, 275.38492063492026, 42.33333333333334, 9.0), 'attention_relationship': ['looking_at'], 'spatial_relationship': ['in_front_of'], 'contacting_relationship': ['touching'], 'metadata': {'tag': 'Q8XEE.mp4/laptop/000111', 'set': 'train'}, 'visible': True}, {'class': 'dish', 'bbox': (204.93649065228874, 276.9772712476055, 44.572616404055246, 9.258509950993453), 'attention_relationship': ['looking_at'], 'spatial_relationship': ['in_front_of'], 'contacting_relationship': ['touching'], 'metadata': {'tag': 'Q8XEE.mp4/dish/000111', 'set': 'train'}, 'visible': True}]
            if obj['contacting_relationship'] is not None:
                cnt_contacting_relationship = max(cnt_contacting_relationship, len(obj['contacting_relationship']))
            if obj['attention_relationship'] is not None:
                cnt_attention_relationship = max(cnt_attention_relationship, len(obj['attention_relationship']))
        cnt_visible_max = max(cnt_visible_max, i)
    print('cnt_spatial_relationship: '+str(cnt_spatial_relationship))        #cnt_spatial_relationship: 5
    print('cnt_contacting_relationship: '+str(cnt_contacting_relationship))        #cnt_contacting_relationship: 4
    print('cnt_attention_relationship: '+str(cnt_attention_relationship))        #cnt_attention_relationship: 1
    print('cnt_visible: '+str(cnt_visible))        #cnt_visible: 737427
    print('cnt_visible_per_frame: '+str(cnt_visible / cnt_frame))        #cnt_visible_per_frame: 2.553576746473118
    print('cnt_visible_per_frame_max: '+str(cnt_visible_max))        #cnt_visible_per_frame_max: 9
    print('cnt_obj_per_frame: '+str(cnt_obj / cnt_frame))        #cnt_obj_per_frame: 2.553576746473118
        
if __name__ == "__main__":
    annotation_dir = '../data/ag/annotations'
    object_anno, person_anno, frame_list = load_annotations(annotation_dir)
    assert set(object_anno.keys()) == set(person_anno.keys())
    assert len(object_anno) == len(frame_list)
    #show_person_bbox(person_anno)
    #show_object_bbox_and_relationship(object_anno, max_show_num=10)
    count_object_bbox_and_relationship(object_anno)
    
'''

001YG.mp4/000089.png
{'bbox': array([[ 24.29774 ,  71.443954, 259.23602 , 268.20288 ]], dtype=float32), 'bbox_score': array([0.9960979], dtype=float32), 'bbox_size': (480, 270), 'bbox_mode': 'xyxy', 'keypoints': array([[[149.51952 , 120.54931 ,   1.      ],
        [146.48587 , 111.43697 ,   1.      ],
        [141.09274 , 115.824394,   1.      ],
        [111.76759 , 123.58676 ,   1.      ],
        [112.44173 , 124.26174 ,   1.      ],
        [ 82.10537 , 154.6362  ,   1.      ],
        [113.45295 , 168.47343 ,   1.      ],
        [153.56436 , 207.96022 ,   1.      ],
        [162.66527 , 247.44699 ,   1.      ],
        [146.48587 , 149.91127 ,   1.      ],
        [216.59659 , 229.22232 ,   1.      ],
        [112.10466 , 243.73456 ,   1.      ],
        [163.3394  , 267.69662 ,   1.      ],
        [237.83205 , 202.56032 ,   1.      ],
        [239.18031 , 202.56032 ,   1.      ],
        [186.93436 , 219.0975  ,   1.      ],
        [220.9785  , 227.87234 ,   1.      ]]], dtype=float32), 'keypoints_logits': array([[11.073427  , 10.578527  , 10.863391  ,  3.6263876 , 11.451177  ,
         4.500312  ,  6.419147  ,  3.4865067 ,  7.920906  ,  5.6766253 ,
         9.343614  , -0.7024717 , -0.36381796,  1.039403  ,  1.1701871 ,
        -0.03817523, -2.2913933 ]], dtype=float32)}
001YG.mp4/000093.png
{'bbox': array([[ 30.254154,  72.110634, 250.53336 , 267.7752  ]], dtype=float32), 'bbox_score': array([0.9933429], dtype=float32), 'bbox_size': (480, 270), 'bbox_mode': 'xyxy', 'keypoints': array([[[150.8511 , 121.53281,   1.     ],
        [148.8271 , 111.07487,   1.     ],
        [141.06842, 116.80987,   1.     ],
        [112.73235, 123.89428,   1.     ],
        [113.40702, 124.56898,   1.     ],
        [ 81.69761, 154.59337,   1.     ],
        [113.40702, 169.43689,   1.     ],
        [151.52577, 207.8951 ,   1.     ],
        [160.97113, 249.38948,   1.     ],
        [144.7791 , 149.19572,   1.     ],
        [214.94458, 229.14832,   1.     ],
        [145.11642, 242.30508,   1.     ],
        [167.04315, 267.26917,   1.     ],
        [186.94586, 187.99129,   1.     ],
        [236.19666, 251.4136 ,   1.     ],
        [186.94586, 219.70245,   1.     ],
        [215.61926, 228.13626,   1.     ]]], dtype=float32), 'keypoints_logits': array([[11.193577  , 10.757453  , 11.36148   ,  3.7291524 , 11.204956  ,
         4.613439  ,  6.7478156 ,  3.3264844 ,  7.550535  ,  5.9024115 ,
         9.554751  , -1.0433159 ,  0.16355455, -0.98884803, -1.2457311 ,
        -0.14376768, -3.0215602 ]], dtype=float32)}
001YG.mp4/000264.png
{'bbox': array([[ 22.108408,  65.39506 , 297.0553  , 266.9599  ]], dtype=float32), 'bbox_score': array([0.9903499], dtype=float32), 'bbox_size': (480, 270), 'bbox_mode': 'xyxy', 'keypoints': array([[[135.62943 , 100.28127 ,   1.      ],
        [132.25584 ,  93.877045,   1.      ],
        [129.55698 ,  95.2253  ,   1.      ],
        [106.616615, 109.04497 ,   1.      ],
        [109.9902  , 108.7079  ,   1.      ],
        [ 65.45893 , 133.31364 ,   1.      ],
        [119.436226, 166.34601 ,   1.      ],
        [144.40073 , 171.402   ,   1.      ],
        [158.56978 , 242.85976 ,   1.      ],
        [140.01508 , 141.06615 ,   1.      ],
        [209.8482  , 224.99532 ,   1.      ],
        [180.83542 , 223.64706 ,   1.      ],
        [173.41353 , 266.45428 ,   1.      ],
        [232.11386 , 194.3224  ,   1.      ],
        [286.4285  , 242.52269 ,   1.      ],
        [247.96968 , 226.68063 ,   1.      ],
        [262.13873 , 234.7702  ,   1.      ]]], dtype=float32), 'keypoints_logits': array([[10.689097  ,  7.2150955 , 11.540033  ,  2.4815173 ,  9.973262  ,
         3.6367958 ,  6.7345824 ,  3.0724719 ,  7.9117594 ,  6.9296474 ,
         8.430188  , -0.23265168,  1.0760294 , -0.20095463,  0.7929864 ,
        -1.1699445 , -2.1476836 ]], dtype=float32)}
50N4E.mp4/000682.png
[{'class': 'light', 'bbox': None, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': '50N4E.mp4/light/000682', 'set': 'train'}, 'visible': False}, {'class': 'dish', 'bbox': None, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': '50N4E.mp4/dish/000682', 'set': 'train'}, 'visible': False}]
50N4E.mp4/000680.png
[{'class': 'light', 'bbox': None, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': '50N4E.mp4/light/000680', 'set': 'train'}, 'visible': False}, {'class': 'dish', 'bbox': (68.13486896878712, 124.97725912325058, 22.36461147520882, 19.541735491260912), 'attention_relationship': ['looking_at'], 'spatial_relationship': ['in_front_of'], 'contacting_relationship': ['holding'], 'metadata': {'tag': '50N4E.mp4/dish/000680', 'set': 'train'}, 'visible': True}]
50N4E.mp4/000689.png
[{'class': 'light', 'bbox': None, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': '50N4E.mp4/light/000689', 'set': 'train'}, 'visible': False}, {'class': 'dish', 'bbox': (43.742838401034604, 136.05715197323664, 28.901881651624752, 17.77797792459259), 'attention_relationship': ['not_looking_at'], 'spatial_relationship': ['in_front_of'], 'contacting_relationship': ['holding'], 'metadata': {'tag': '50N4E.mp4/dish/000689', 'set': 'train'}, 'visible': True}]
50N4E.mp4/000719.png
[{'class': 'light', 'bbox': None, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': '50N4E.mp4/light/000719', 'set': 'train'}, 'visible': False}, {'class': 'dish', 'bbox': (421.52164502164464, 137.38428238428236, 15.666666666666629, 24.66666666666663), 'attention_relationship': ['not_looking_at'], 'spatial_relationship': ['behind'], 'contacting_relationship': ['not_contacting'], 'metadata': {'tag': '50N4E.mp4/dish/000719', 'set': 'train'}, 'visible': True}]
50N4E.mp4/000701.png
[{'class': 'light', 'bbox': None, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': '50N4E.mp4/light/000701', 'set': 'train'}, 'visible': False}, {'class': 'dish', 'bbox': (24.201217747501087, 126.30262249827456, 25.517015054768596, 12.073582425550045), 'attention_relationship': ['looking_at'], 'spatial_relationship': ['in_front_of'], 'contacting_relationship': ['holding'], 'metadata': {'tag': '50N4E.mp4/dish/000701', 'set': 'train'}, 'visible': True}]
50N4E.mp4/000756.png
[{'class': 'light', 'bbox': None, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': '50N4E.mp4/light/000756', 'set': 'train'}, 'visible': False}, {'class': 'dish', 'bbox': None, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': '50N4E.mp4/dish/000756', 'set': 'train'}, 'visible': False}]
E546V.mp4/000067.png
[{'class': 'light', 'bbox': None, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': 'E546V.mp4/light/000067', 'set': 'train'}, 'visible': False}]
50N4E.mp4/000737.png
[{'class': 'light', 'bbox': None, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': '50N4E.mp4/light/000737', 'set': 'train'}, 'visible': False}, {'class': 'dish', 'bbox': None, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': '50N4E.mp4/dish/000737', 'set': 'train'}, 'visible': False}]
E546V.mp4/000199.png
[{'class': 'light', 'bbox': None, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': 'E546V.mp4/light/000199', 'set': 'train'}, 'visible': False}]
E546V.mp4/000463.png
[{'class': 'light', 'bbox': None, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': 'E546V.mp4/light/000463', 'set': 'train'}, 'visible': False}]
E546V.mp4/000331.png
[{'class': 'light', 'bbox': None, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': 'E546V.mp4/light/000331', 'set': 'train'}, 'visible': False}]

'''