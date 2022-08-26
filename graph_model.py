import torch
import torch
from torchvision.models import resnet50, resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torch_geometric.nn import GCNConv, global_mean_pool, max_pool_x, graclus
import torch.nn.functional as F


class GraphModel(torch.nn.Module):
    def __init__(self,  input_features: int, num_classes: int, device: str):
        super(GraphModel, self).__init__()
        self.input_features = input_features
        self.device = torch.device(device)
        self.is_cuda = True
        # Create a New Graph instead of PointNet
        # # Get a resnet50 backbone
        # m = resnet50()
        # # Extract 4 main layers (note: MaskRCNN needs this particular name
        # # mapping for return nodes)
        # self.resnet = create_feature_extractor(
        #     m, return_nodes={f'layer{k}': str(v)
        #                      for v, k in enumerate([1, 2, 3, 4])})
        # # Dry run to get number of channels for FPN
        # inp = torch.randn(2, 3, 224, 224)
        # with torch.no_grad():
        #     out = self.resnet(inp)
        # in_channels_list = [o.shape[1] for o in out.values()]
        # # Build FPN
        # self.out_channels = 256
        # self.fpn = FeaturePyramidNetwork(
        #     in_channels_list, out_channels=self.out_channels,
        #     extra_blocks=LastLevelMaxPool())
        self.resnet = resnet18(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.conv0 = GCNConv(512, 64)
        self.conv1 = GCNConv(64, num_classes)
        # self.conv2 = GCNConv(64, num_classes)

    def forward(self, data):
    #
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        # x = self.resnet(torch.permute(data.x[:9], dims=[0,3,1,2]))

        # x_final = []
        # for i in range(data.x.size()[0]):
        #     x = self.feature_extractor(torch.permute(data.x[i:i+1], dims=[0, 3, 1, 2])).squeeze()
        #     x_final.append(x)
        # x = torch.stack(x_final, dim=0)
        x = self.feature_extractor(torch.permute(data.x, dims=[0, 3, 1, 2])).squeeze()
        # x = F.relu(x)
        x = self.conv0(x, data.edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.conv2(x, edge_index)
        # x = x.permute(0, 2, 1).max(dim=2)
        return F.log_softmax(x, dim=1)


    #     if not isinstance(tooth_data, list):
    #         tooth_data = [tooth_data]
    #     # data_loader = DataListLoader(tooth_data, batch_size=len(tooth_data))
    #     for j, tooth in enumerate(tooth_data):
    #     # for j, tooth in enumerate(data_loader):
    #         loader = DataLoader(tooth, batch_size=len(tooth))
    #         for i, tt in enumerate(loader):
    #             tt = tt.to(self.device)
    #             x = tt.x.view(-1, 1).to(self.device)
    #             #x = torch.cat((x, tt.pos), 1)
    #             l = self.conv0(x, tt.edge_index.to(self.device))#.view(-1, 512)
    #             l = l.relu()
    #             l = self.conv00(l, tt.edge_index.to(self.device))
    #             l = global_max_pool(l, tt.batch)
    #             if j == 0:
    #                 feat = l
    #             if j >= 1:
    #                 feat = torch.cat((feat, l), 0)
    #     edge_attr = torch.index_select(edge_attr, 1, torch.tensor([0, 1]).to(self.device))
    #     x = self.conv1(feat, edge_index, edge_attr)
    #     x = F.relu(x)
    #     x = F.dropout(x, training=self.training)
    #     x = self.conv2(x, edge_index, edge_attr)
    #     return F.log_softmax(x, dim=1)


# class Resnet_GNN(torch.nn.Module):
#     def __init__(self, num_classes, device):
#         super(Resnet_GNN, self).__init__()
#         self.num_classes = num_classes
#         self.device = device
#         # spline x,edge_index,edge_features
#         self.resnet = models.resnet18(pretrained=False)
#         self.feature_extractor = torch.nn.Sequential(*list(self.resnet.children())[:-1])
#         # function defined earlier for fearture extraction output tensor zize 512)
#         self.conv1 = SplineConv(512, 128, dim=2, kernel_size=5, aggr="max")
#         self.conv3 = SplineConv(128, 64, dim=2, kernel_size=5, aggr="max")
#         self.conv2 = SplineConv(64, self.num_classes, dim=2, kernel_size=5, aggr="max")
#
#     def forward(self, x, edge_index, edge_attr):
#         x = self.feature_extractor(x).squeeze()
#         x = self.conv1(x, edge_index, edge_attr)
#         x = self.conv3(x, edge_index, edge_attr)
#         x = x.relu()
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index, edge_attr)
#         return x
