import random
from graph_frame_dataset import FramesGraph
from torch_geometric.data import DataLoader
import torch
from graph_model import GraphModel
import torch.nn as nn
import torch_geometric.transforms as T
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.functional as F


writer = SummaryWriter(comment="test", filename_suffix='GNN')
device = 'cuda'
train_dataset = FramesGraph('./dfl-bundesliga-data-shootout', split='train', cross_val_idx=0,
                            transform=T.Compose([T.KNNGraph(k=2), T.Distance()]))
batch_size = 4
random.seed(112)
train_dataset = train_dataset.shuffle()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(train_dataset[800:], batch_size=batch_size, shuffle=True)
model = GraphModel(input_features=1536, num_classes=4, device='cuda')
# model = otherModel(input_features=1536, num_classes=16)
model.to(device)
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=1e-10, momentum=0.9, weight_decay=1e-11)
optimizer = torch.optim.Adam(params, lr=1e-5, weight_decay=1e-5)  # TODO e-4
criterion = nn.CrossEntropyLoss().to(device)  # Update parameters bas


def train(data):
    model.train()
    optimizer.zero_grad()  # zero the gradient buffers
    data = data.to(device)
    output = model(data)
    loss = criterion(output, data.y)
    loss.backward()
    optimizer.step()
    return loss


def test(data_element):
    model.eval()
    out = model(data_element)
    loss = criterion(out, data_element.y)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[:] == data_element.y[:]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data_element.y.size()[0])
    return test_acc, loss, pred


if __name__ == "__main__":
    checkpoint = torch.load('checkpoints/epoch_20.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    # last_epoch = -1
    n_epochs = 500
    output_dir = './checkpoints'
    os.makedirs(output_dir, exist_ok=True)
    for epoch in range(last_epoch + 1, n_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            loss = train(data)
            running_loss += loss.item()
            writer.add_scalar("Loss/train_step", loss.item(), epoch * len(train_loader) + i)
        print("epoch : ", epoch, "loss : ", running_loss/len(train_loader))
        if epoch % 4 == 0:
            TAR = np.array([])
            PRE = np.array([])
            runnning_loss_val = 0.0
            for j, data_eval in enumerate(train_loader):
                with torch.no_grad():
                    acc_, loss_val, pred = test(data_eval.to(device))
                    runnning_loss_val += loss_val
                    writer.add_scalar("Loss/val_step", loss_val, epoch * len(train_loader) + j)
                    TAR = np.hstack((TAR, data_eval.y.tolist()))
                    PRE = np.hstack((PRE, pred.tolist()))
            from sklearn.metrics import classification_report, f1_score, accuracy_score

            print("epoch: ", epoch, "classification report= \n{}".format(classification_report(TAR, PRE, digits=4)))
            print("=============================================================")
            print('average accuracy for epoch {} is {}'.format(epoch, accuracy_score(TAR, PRE)))
            print("=============================================================")
            writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)
            writer.add_scalar("Loss/Val", runnning_loss_val / len(train_loader), epoch)
            writer.add_scalar("Acc/Val", accuracy_score(TAR, PRE), epoch)
        if epoch % 2 == 0 and epoch >= 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(output_dir, 'epoch_{}.pth'.format(epoch)))
