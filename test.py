# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.testmodel as testmodel
import utils.myDataLoader as myDataLoader
import os
# %%
numClusterFeature = 21
numSlice = 4
numSliceFeature = 20

# %%
savedir='output/865/'
os.makedirs(savedir, exist_ok=True)
myDataLoader.createDataSet(eventfile='event/865.txt', num_slices=numSlice, SaveDir=savedir, num_cluster=10)

# %%
model=testmodel.ClassifierModel(numClusterFeature, numSlice, numSliceFeature)
# %%
torch.onnx.export(model, 
            (torch.randn(1, numSlice*4, 50, 50), 
                torch.randn(1, numClusterFeature), 
                torch.randn(1, numSliceFeature*numSlice)), 
            'model/testmodel.onnx',
            input_names = ['Slice IMGs', 'Cluster info', 'Slice info'],   # the model's input names
            output_names = ['output'],)
