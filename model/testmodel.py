import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierModel(nn.Module):
    '''
    Cluster information(numClusterFeature, 1)               ->  Linear  
    Slice(numSlice*4, 50, 50)   ->  CNN  ->  Flatten        ->  Linear
    Other Slice information(numSlice*numSliceFeature, 1)    ->  Linear  
    '''
    def __init__(self, numClusterFeature = 21, numSlice = 4, numSliceFeature = 20):
        super().__init__()
        self.Slice_CNNmodel = nn.Sequential(
                nn.Conv2d(numSlice*4, 128, 2),
                nn.MaxPool2d(2,2),
                nn.Conv2d(128, 64, 2),
                nn.MaxPool2d(2,2),
                nn.Conv2d(64, 32, 2),
                nn.MaxPool2d(2,2),
                nn.Conv2d(32, 16, 2),
                nn.MaxPool2d(2,2),                
                )
        self.Cluster_model = nn.Sequential(
                nn.Linear(numClusterFeature, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                )
        self.Slice_info_model =  nn.Sequential(
                nn.Linear(numSliceFeature*numSlice, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                )
        self.linear1 = nn.Linear(96, 48)
        self.linear2 = nn.Linear(48, 24)
        self.linear3 = nn.Linear(24, 2)


    def forward(self, Slice_img, Cluster_info, Slice_info):
        x = torch.flatten(self.Slice_CNNmodel(Slice_img), start_dim=1)
        x = torch.cat((x, self.Cluster_model(Cluster_info), self.Slice_info_model(Slice_info)), 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


