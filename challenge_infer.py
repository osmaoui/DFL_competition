import glob
import os.path

from torch_geometric.data import InMemoryDataset, Data
import cv2
from tqdm import tqdm
import torch
from graph_frame_dataset import get_graph_attr
import numpy as np
from itertools import repeat, product
from torch import Tensor
import torch_geometric.transforms as T
from graph_model import GraphModel
from torch_geometric.data import DataLoader
import csv


def collate_forward(data_list):
    keys = data_list[0].keys
    data = data_list[0].__class__()

    for key in keys:
        data[key] = []
    slices = {key: [0] for key in keys}

    for item, key in product(data_list, keys):
        data[key].append(item[key])
        if isinstance(item[key], Tensor) and item[key].dim() > 0:
            cat_dim = item.__cat_dim__(key, item[key])
            cat_dim = 0 if cat_dim is None else cat_dim
            s = slices[key][-1] + item[key].size(cat_dim)
        else:
            s = slices[key][-1] + 1
        slices[key].append(s)

    if hasattr(data_list[0], '__num_nodes__'):
        data.__num_nodes__ = []
        for item in data_list:
            data.__num_nodes__.append(item.num_nodes)

    for key in keys:
        item = data_list[0][key]
        if isinstance(item, Tensor) and len(data_list) > 1:
            if item.dim() > 0:
                cat_dim = data.__cat_dim__(key, item)
                cat_dim = 0 if cat_dim is None else cat_dim
                data[key] = torch.cat(data[key], dim=cat_dim)
            else:
                data[key] = torch.stack(data[key])
        elif isinstance(item, Tensor):  # Don't duplicate attributes...
            data[key] = data[key][0]
        elif isinstance(item, int) or isinstance(item, float):
            data[key] = torch.tensor(data[key])

        slices[key] = torch.tensor(slices[key], dtype=torch.long)
    return data, slices


class ForwardFramesGraph(InMemoryDataset):
    def __init__(self, video_path, skip_frames=12, video_length=240, pre_transform=None):
        self.video_path = video_path
        self.skip_frames = skip_frames
        self.video_length = video_length
        super(ForwardFramesGraph, self).__init__(None, None, pre_transform)

    def process_forward(self):
        cap = cv2.VideoCapture(self.video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        data_list = []
        for k in tqdm(range(0, num_frames, self.video_length)):
            x, pos, _ = get_graph_attr(cap, k, k+self.video_length, self.skip_frames)
            x = torch.tensor(np.array(x), dtype=torch.float)
            pos = torch.tensor(pos, dtype=torch.float)
            data = Data(x=x, pos=pos)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        self.data, self.slices = collate_forward(data_list)


def get_key_from_value(val):
    LUT = {'background': 0, 'challenge': 1, 'play': 2, 'throwin': 3}
    keys = [k for k, v in LUT.items() if v == val]
    if keys:
        return keys[0]
    return None


if __name__ == "__main__":

    device = torch.device('cuda')
    model = GraphModel(input_features=1536, num_classes=4, device=device)
    print(model)
    checkpoint_path = "checkpoints/epoch_34.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    output_list = []
    for video_path in tqdm(glob.glob('./dfl-bundesliga-data-shootout/test/*')):
        video_id = os.path.basename(video_path).split('.mp4')[0]
        dataset = ForwardFramesGraph("dfl-bundesliga-data-shootout/test/2f54ed1c_0.mp4",
                                     pre_transform=T.Compose([T.KNNGraph(k=2), T.Distance()]))
        dataset.process_forward()
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        ### EVAL MODEL

        out = model(dataset.data)
        # pred = out.argmax(dim=1).tolist()
        prob = np.exp(out.detach().cpu().numpy())
        prob[prob < 0.95] = 0
        pred = np.argmax(prob, axis=1)

        for i, label in enumerate(pred):
            if label != 0:
                output_list.append([video_id, str(i*12/25), get_key_from_value(label), max(prob[i])])


    with open('submission.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(['video_id','time','event','score'])
        write.writerows(output_list)

    print("Done")