from graph_frame_dataset import FramesGraph
from torch_geometric.data import DataLoader
import torch
from graph_model import GraphModel
import torch_geometric.transforms as T
from train import test
import numpy as np
from tqdm import tqdm


train_dataset = FramesGraph('./dfl-bundesliga-data-shootout', split='train', cross_val_idx=0,
                            transform=T.Compose([T.KNNGraph(k=2), T.Distance()]))

device = 'cpu'
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
model = GraphModel(input_features=1536, num_classes=4, device='cpu')
# model = otherModel(input_features=1536, num_classes=16)
model.to(device)

if __name__ == "__main__":
    checkpoint = torch.load('checkpoints/epoch_20.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    TAR = np.array([])
    PRE = np.array([])
    for j, data_element in tqdm(enumerate(train_loader)):
        with torch.no_grad():
            model.eval()
            out = model(data_element)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            test_correct = pred[:] == data_element.y[:]  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / int(data_element.y.size()[0])

            TAR = np.hstack((TAR, data_element.y.tolist()))
            PRE = np.hstack((PRE, pred.tolist()))
    from sklearn.metrics import classification_report, f1_score, accuracy_score

    print("classification report= \n{}".format(classification_report(TAR, PRE, digits=4)))