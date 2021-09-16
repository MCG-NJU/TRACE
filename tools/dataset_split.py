import csv
import json
import os

def get_set(path, file_name, save_path, save_name):
    ans = set()
    with open(os.path.join(path, file_name), 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            ans.add(row[0])
        f.close()
    
    ans = list(ans)
    with open(os.path.join(save_path, save_name), 'w') as f:
        json.dump(ans, f)
        f.close()
    
    return ans
        
if __name__ == '__main__':
    path = 'data/ag/Charades_annotations'
    save_path = 'data/ag/annotations'
    ans = get_set(path, 'Charades_v1_test.csv', save_path, 'test_videos_list.json')
    print(len(ans))
    ans = get_set(path, 'Charades_v1_train.csv', save_path, 'train_videos_list.json')
    print(len(ans))