import os
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

def get_json(root, dir_name:list, output_path='./data/data.json', flag=False):
    data = {}
    total_length = 0
    valid_length = 0
    for dir in dir_name:
        path = os.path.join(root, dir)
        path = Path(path)
        data[str(dir)] = []
        files = []
        for file in os.listdir(path):
            file = str(file).split('/')[-1]
            file_time = file.split('_')[0]
            files.append((int(file_time), file))
        files.sort(key=lambda x:x[0])
        total_length += len(files)
        for (file_time, file) in files:
            if flag:
                if str(file_time)[-2:] != '00':
                    continue
            valid_length += 1
            data[str(dir)].append(file)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=1)
    return (total_length, valid_length)

def check_in(labels_path='./data/labels.json', images_path='./data/images.json'):
    with open(images_path, 'r') as f:
        images = json.load(f)
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    data_images = []
    data_labels = []
    for i in images.values():
        for j in i:
            data_images.append(j.split('_')[0] + '_' + j.split("_")[1] + '.png')
    for i in labels.values():
        for j in i:
            data_labels.append(j)    
    for i in data_labels:
        if not (i in data_images):
            print(i)
            return False
    return True

def check_seqs(time1, time2, interval=15):
    time1 = time1.split('_')[0]
    time2 = time2.split('_')[0]
    minute1 = int(time1[-4:-2]) + int(time1[-6:-4]) * 60 + int(time1[-8:-6]) * 24 * 60
    minute2 = int(time2[-4:-2]) + int(time2[-6:-4]) * 60 + int(time2[-8:-6]) * 24 * 60
    return (minute2 - minute1) == interval

def get_seqs(labels_path='./data/labels.json', seqs_path='./data/seqs.json', frames=6, interval=15):
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    seqs = {}
    length = 0
    for key, tmp in labels.items():
        seqs[key] = []
        for l in range(len(tmp)):
            t=[]
            for r in range(l,len(tmp),int(interval/15)):
                if len(t) == 0:
                    t.append(tmp[r])
                else:
                    if check_seqs(t[-1], tmp[r], interval):
                        t.append(tmp[r])
                    else:
                       break
                if len(t) == frames:
                    break
            if len(t) == frames:
                length += 1
                seqs[key].append(t)
    with open(seqs_path, 'w') as f:
        json.dump(seqs, f, indent=1)
    return length

def split(seqs_path='./data/seqs.json', train_path='./data/train.json', valid_path='./data/valid.json', test_path='./data/test.json'):
    with open(seqs_path, 'r') as f:
        seqs = json.load(f)
    final_train = []
    final_valid = []
    final_test = []
    for value in seqs.values():
        train_valid, test = train_test_split(value, test_size=0.15, random_state=0)
        train, valid = train_test_split(train_valid, test_size=0.15/0.85, random_state=0)
        final_train.extend(train)
        final_valid.extend(valid)
        final_test.extend(test)
    with open(train_path, 'w') as f:
        json.dump(final_train, f, indent=1)
    with open(valid_path, 'w') as f:
        json.dump(final_valid, f, indent=1)
    with open(test_path, 'w') as f:
        json.dump(final_test, f, indent=1)
    return (len(final_train), len(final_valid), len(final_test))

print(split(seqs_path='./data/labels.json'))

#if __name__ == 'data_preprocessing':
    # month_list = [f"2018{str(i).zfill(2)}" for i in range(1, 13)]
    # print(get_json('/mnt/md1/ConvectionAirport/Datasets/Satellite/FY4A/MCSRSI/bright_images',
    #         dir_name=month_list, output_path='./data/images.json', flag=False))
    # print(get_json('/mnt/md1/ConvectionAirport/Datasets/Satellite/FY4A/MCSRSI/labels_v2/all',
    #         dir_name=month_list, output_path='./data/labels.json', flag=True))
    # if check_in():
    #     print(get_seqs())
    #     print(split())
    
    