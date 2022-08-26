from torch_geometric.data import InMemoryDataset, Data
import os.path as osp
import torch
import pandas as pd
from glob import glob
import json
from tqdm import tqdm
from intervaltree import *
import numpy as np
import torch_geometric.transforms as T
import cv2
from yolo_detector import OfflineObjectDetectionYolo

LUT = {'background': 0, 'challenge': 1, 'play': 2, 'throwin': 3}

# yolo_feat = OfflineObjectDetectionYolo(
#     model_path='/media/oussama/60d0458f-2f1f-4c73-bfe4-93757a0b94c5/home/oussama/workspace/reeplayer/github/reeplayer-AI---Tools/yolo_training/weights/best_ckpt.pt',
#     config_path='/media/oussama/60d0458f-2f1f-4c73-bfe4-93757a0b94c5/home/oussama/workspace/reeplayer/github/reeplayer-AI---Action-Tracking---Python-GPU/conf/coco.yaml',
#     device="cuda",
#     skip_frames=1,
# )

def get_graph_attr(cap_vid, start_frame, end_frame, step, video_id=None):
    x = []
    pos = []
    labels = []
    for fn in range(start_frame, end_frame, step):
        cap_vid.set(cv2.CAP_PROP_POS_FRAMES, fn)
        # while True:
        ret, frame = cap_vid.read()
        if frame is not None:
            image = cv2.resize(frame, (224, 224), cv2.INTER_LINEAR)
            norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)
            # x.append(yolo_feat.predict(frame)[0])
            x.append(norm_image)
            pos.append([0 + fn, 0])
            # TODO comment for forward
            # y = extract_label_from_frame_num(get_video_action_tree(video_id), fn / 25)
            # cv2.putText(frame, str(y), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
            # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            # cv2.imshow('test', frame)
            # cv2.waitKey(0)
            # labels.append(y)
    return x, pos, labels


def get_video_action_tree(video_id):
    df = pd.read_csv('dfl-bundesliga-data-shootout/train.csv')
    df_video = df[df.video_id == video_id]
    arr = df_video[['time', 'event']].values
    tree = IntervalTree()
    idx = 0
    event_time_list = []
    while idx < len(arr) - 1:
        start_time = arr[idx, 0]
        idx += 1
        crr_event = arr[idx, 1]
        while crr_event in ['start', 'end']:
            idx += 1
            crr_event = arr[idx, 1]
        idx += 1
        end_time = arr[idx, 0]
        idx += 1

        event_time_list.append(end_time - start_time)
        tree[start_time: end_time] = crr_event
    return tree


def extract_label_from_frame_num(tree, frame_time):
    if len(tree[frame_time]) == 0:
        return LUT['background']
    else:
        assert len(tree[frame_time]) == 1
        for interval_set in tree[frame_time]:
            return LUT[interval_set.data]


class FramesGraph(InMemoryDataset):
    def __init__(self, root, split, cross_val_idx, transform=None, pre_transform=None):
        self.split = split
        self.crossVal = cross_val_idx
        super(FramesGraph, self).__init__(root, transform, pre_transform)

        if self.split == "train":
            path = self.processed_paths[0]
        elif self.split == "val":
            path = self.processed_paths[1]
        elif self.split == "test":
            path = self.processed_paths[2]

        else:
            raise ValueError(
                (f"Split {split} found, but expected either " "train, val, trainval or test"))

    #         self.data, self.slices = torch.load(path)

    @property
    def processed_dir(self):
        return osp.join("./", 'processed')

    #     @property
    #     def processed_file_names(self):
    #         return [osp.join("data_{}_{}.pt".format(split, self.video_idx)) for split in [self.split]]
    @property
    def processed_file_names(self):
        return ['data_' + str(idx) + '.pt' for idx in range(244)]

    def process(self):
        for i, split in enumerate([self.split]):
            video_paths = glob(self.root + '/' + self.split + '/*.mp4')
            #             video_paths =  glob(self.root + '/' +self.split +'/' + self.video_idx + '.mp4')
            data_list = []
            idx = 0
            for video_path in tqdm(video_paths):
                step = 240
                cap_vid = cv2.VideoCapture(video_path)
                video_id = video_path.split('/')[-1].split('.')[0]
                # for k in tqdm(range(25000, 40000, step)):
                for k in tqdm(range(42000, 50000, step)):
                    x, pos, labels = get_graph_attr(cap_vid, k, k + step, 12, video_id)
                    print(labels, video_id)
                    if len(np.unique(np.array(labels))) == 1 and 0 in labels:
                                 continue

                    # x = torch.stack(x, dim=0)
                    x = torch.tensor(np.array(x), dtype=torch.float)
                    y = torch.tensor(labels, dtype=torch.long)
                    pos = torch.tensor(pos, dtype=torch.float)
                    data = Data(x=x, pos=pos, y=y)
                    data.name = video_id
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
                    idx += 1

    #                     data_list.append(data)
    #                     torch.save(self.collate(data_list), self.processed_paths[i])
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


if __name__ == "__main__":

    # train_dataset = FramesGraph('./dfl-bundesliga-data-shootout', split='train', cross_val_idx=0)
    #                             #transform=T.Compose([T.KNNGraph()]))
    print()
