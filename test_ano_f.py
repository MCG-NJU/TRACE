import os
import math
import pickle
import json
import numpy as np

R = 100
KK = 26
N = 55

def cal_select_prob(n, k):
    if n*KK < R:
        return 1. * k
    b = 0
    #for k in range(0, N):
    #    a = np.random.choice(n*26, 50)
    #    b += (a <= k).astype(np.int).sum()
    a = np.random.choice(n*KK*N, R*N, replace=False)
    b += (a <= k*N).astype(np.int).sum()
    c = float(b) / float(N)
    return c

def cal_inter_area(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if (x <= x1) and (y <= y1):
        return (x1-x+1)*(y1-y+1)
    else:
        return -1

        
        
with open('./new_annotations_val.json', 'r') as f:
    an_list = json.load(f)
    f.close()
cnt = 0
all_cnt = 0
max_tri = 0
max_pair = 0
max_r_pair = 0.
mean_r_pair = 0.
mean_r_pair2 = 0.
valid_g_cnt = 0.
pair_count = dict()
sum_prob = []
sum_len_g = 0.

NN = 26
hit_num = [0 for i in range(0, NN)]


print(len(an_list))
for i, v in an_list.items():
    g = dict()
    for trip in v:
        y, y1, x, x1 = trip['object']['bbox']
        oc = trip['object']['category']
        obox = [x, y, x1, y1, oc]
        y, y1, x, x1 = trip['subject']['bbox']
        sc = trip['subject']['category']
        sbox = [x, y, x1, y1, sc]
        p = trip['predicate']
        a = cal_inter_area(sbox, obox)
        if a >= 0:
            cnt += 1
        all_cnt += 1
        if tuple(sbox + obox) not in g: g[tuple(sbox + obox)] = 0
        g[tuple(sbox + obox)] += 1
    
    max_tri = max(max_tri, len(v))
    max_pair = max(max_pair, len(g))
    sum_len_g += 1. * len(g)
    
    if len(g) not in pair_count: pair_count[len(g)] = 0.
    pair_count[len(g)] += 1.
    
    
    for ii in range(1, NN):
        s50 = 0
        for kk,vv in g.items():
            if vv > 0:
                mean_r_pair += vv
                mean_r_pair2 += vv**2
                max_r_pair = max(max_r_pair, vv)
                valid_g_cnt += 1
                s50 += min(vv, ii) * 1.0
        hit_num[ii] += min(s50, 50)
    #hitnum = cal_select_prob(len(g), len(v))
    #sum_prob.append(hitnum)
    
r = float(cnt) / (1e-12 + float(all_cnt)) * 100
print('{} val relations iou>0. '.format(r)) # 85.92895152573377 val relations iou>0.
print('{} val relations density. '.format(float(all_cnt / len(an_list)))) # 8.451416826238955 val relations density.
print(max_tri) #31
print(max_pair) #9
print(max_r_pair) #9
print(mean_r_pair / valid_g_cnt) #3.28
print(math.sqrt(mean_r_pair2 / valid_g_cnt - (mean_r_pair / valid_g_cnt)**2)) #0.56

print('number of pairs in scene:')
tot = 0.
n_allcountpair = 0.
for k,v in pair_count.items():
    if k * 26 <= 105:
        n_allcountpair += v
    tot += v
for k,v in pair_count.items():
    print('{}: {}'.format(k, 100*v/tot))
print('number of all count pairs: {}, Expect recall: {}'.format(n_allcountpair /tot * 100, n_allcountpair /tot * 100 + (1 - n_allcountpair /tot) * 50))
print()

#print('Expect hit: {}'.format(np.array(sum_prob,dtype=np.float32).sum() / float(all_cnt)))

for ii in range(1, NN):
    print('must hit one (k={}), recall: {}'.format(ii, hit_num[ii] / float(all_cnt)))
#must hit one (k=1), recall: 0.3048474266234306
#must hit one (k=2), recall: 0.6096948532468612
#must hit one (k=3), recall: 0.9145422798702918
#must hit one (k=4), recall: 0.9844558077658602
#must hit one (k=5), recall: 0.9973476344890663
#must hit one (k=6), recall: 0.9999002244948865
#must hit one (k=7), recall: 0.9999937640309304
#must hit one (k=8), recall: 0.9999979213436435
print()


with open('./new_annotations_train.json', 'r') as f:
    an_list = json.load(f)
    f.close()
cnt = 0
all_cnt = 0
max_tri = 0
max_pair = 0
mean_r_pair = 0.
mean_r_pair2 = 0.
max_r_pair = 0.
valid_g_cnt = 0.
pair_count = dict()
sum_prob = []
for i, v in an_list.items():
    g = dict()
    for trip in v:
        y, y1, x, x1 = trip['object']['bbox']
        oc = trip['object']['category']
        obox = [x, y, x1, y1, oc]
        y, y1, x, x1 = trip['subject']['bbox']
        sc = trip['subject']['category']
        sbox = [x, y, x1, y1, sc]
        p = trip['predicate']
        a = cal_inter_area(sbox, obox)
        if a >= 0:
            cnt += 1
        all_cnt += 1
        if tuple(sbox + obox) not in g: g[tuple(sbox + obox)] = 0
        g[tuple(sbox + obox)] += 1
        
    max_tri = max(max_tri, len(v))
    max_pair = max(max_pair, len(g))
    
    if len(g) not in pair_count: pair_count[len(g)] = 0.
    pair_count[len(g)] += 1.
    
    for kk,vv in g.items():
        if vv > 0:
            mean_r_pair += vv
            mean_r_pair2 += vv**2
            max_r_pair = max(max_r_pair, vv)
            valid_g_cnt += 1
    hitnum = cal_select_prob(len(g), len(v))
    sum_prob.append(hitnum)        
r = float(cnt) / (1e-12 + float(all_cnt)) * 100
print('{} train relations iou>0. '.format(r)) # 87.28015177142264 train relations iou>0.
print('{} train relations density. '.format(float(all_cnt / len(an_list)))) # 6.96152935205549 train relations density.
print(max_tri) #29
print(max_pair) #8
print(max_r_pair) #8
print(mean_r_pair / valid_g_cnt) #3.28
print(math.sqrt(mean_r_pair2 / valid_g_cnt - (mean_r_pair / valid_g_cnt)**2)) #0.57

print('number of pairs in scene:')
tot = 0.
n_allcountpair = 0.
for k,v in pair_count.items():
    if k * 26 <= 105:
        n_allcountpair += v
    tot += v
for k,v in pair_count.items():
    print('{}: {}'.format(k, 100*v/tot))
print('number of all count pairs: {}, Expect recall: {}'.format(n_allcountpair /tot * 100, n_allcountpair /tot * 100 + (1 - n_allcountpair /tot) * 50))
print()

print('Expect hit: {}'.format(np.array(sum_prob,dtype=np.float32).sum() / float(all_cnt)))


#with open(os.path.join('./', 'object_bbox_and_relationship.pkl'), 'rb') as f:
#    object_anno = pickle.load(f)
#    f.close()
#
#cnt = 0
#c = 0
#
#tr = set()
#va = set()
#for i, v in object_anno.items():
#    if i == '00607.mp4/000016.png':
#        print(v)
#        assert False
#    for j in v:
#        if j['metadata']['set'] != 'train':
#            cnt += 1
#            a = j['metadata']['tag'].split('/')[0].split('.')[0]
#            va.add(a)
#            break
#        else:
#            c += 1
#            a = j['metadata']['tag'].split('/')[0].split('.')[0]
#            tr.add(a)
#            break
#print(len(tr))
#print(len(va))
#
#
#with open(os.path.join('./', 'train_videos_list.json'), 'r') as f:
#    object_anno = json.load(f)
#    print(len(object_anno))
#    tl = set(object_anno)
#    f.close()
#    
#with open(os.path.join('./', 'test_videos_list.json'), 'r') as f:
#    object_anno = json.load(f)
#    print(len(object_anno))
#    vl = set(object_anno)
#    f.close()
#    
#
#print(len(tr-tl))
#print(len(tl-tr))
#print(len(va-vl))
#print(list(vl-va))



