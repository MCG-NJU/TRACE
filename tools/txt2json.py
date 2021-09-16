import os
import json

def class_coco2ag(ag_p, coco_p):
    with open(ag_p, 'r') as f:
        ag_c = json.load(f)
        f.close()
    with open(coco_p, 'r') as f:
        coco_c = json.load(f)
        f.close()
    
    ans = [0, ]
    for i in ag_c:
        ii = i
        if ii not in coco_c:
            if i == 'closetcabinet': #
                ii = 'cabinet'
            if i == 'cupglassbottle':
                ii = 'bottle'
            if i == 'papernotebook': #
                ii = 'notebook'
            if i == 'phonecamera':
                ii = 'cell_phone'
            if i == 'sofacouch':
                ii = 'couch'
            if i == 'table':
                ii = 'dining_table'
            if ii not in coco_c:
                print('fail on {}.'.format(ii))
                ans.append(-1)
                continue
        
        id = coco_c.index(ii) + 1
        ans.append(id)

    return ans
    
def txt2json(path, txt_path, json_path):
    with open(txt_path, 'r') as f:
        s = f.read().split()
        f.close()
    with open(json_path, 'w') as fj:
        fj.write(json.dumps(s))
        fj.close()
    print(len(s))
if __name__ == '__main__':
    path = './data/ag/annotations'
    obj_txt_path = os.path.join(path, 'object_classes.txt')
    obj_json_path = os.path.join(path, 'objects.json')
    pred_txt_path = os.path.join(path, 'relationship_classes.txt')
    pred_json_path = os.path.join(path, 'predicates.json')
    if not os.path.exists(obj_json_path):
        txt2json(path, obj_txt_path, obj_json_path) #36
    if not os.path.exists(pred_json_path):
        txt2json(path, pred_txt_path, pred_json_path) #26
        
    cobj_txt_path = os.path.join(path, 'COCO_object_class.txt')
    cobj_json_path = os.path.join(path, 'COCO_object_class.json')
    if not os.path.exists(cobj_json_path):
        txt2json(path, cobj_txt_path, cobj_json_path) #80

    c2a = class_coco2ag(obj_json_path, cobj_json_path)
    print(c2a)
        