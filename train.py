# %%
import sys
sys.path.append('utils')
import numpy as np
#import utils.select_id as select_id
#import utils.cluster as clst
#import utils.slice as slc
import os
import torch
import json
import model.testmodel as testmodel
# %%
# load normalization parameters
normalization = json.load(open('utils/normalization.json'))
ADC_index = [0,2,4,6,8,10,12,14,]
time_index = [1,3,5,7,9,11,13,15,]
# %%
# load all data file directory
from pathlib import Path
filelist=[]
for path in Path('output').rglob('*.npz'):
    filelist.append(path.__str__())
# %%
from torch.utils.data import Dataset, DataLoader, TensorDataset
class MyDataset(Dataset):
    def __init__(self, file_dir_list_):
        self.file_list = file_dir_list_
        self.len = len(file_dir_list_)

    def __getitem__(self, index):
        file_dir = self.file_list[index]
        data = np.load(file_dir)
        _Slice_img = data['Slice_img']
        _Cluster_info = data['Cluster_info']
        _Slice_info = data['Slice_info']
        _supernova = data['supernova']
        _Slice_img[ADC_index, :, :] = (_Slice_img[ADC_index, :, :] - normalization['ADC img']['mean']) / normalization['ADC img']['std']
        _Slice_img[time_index, :, :] = (_Slice_img[time_index, :, :] - normalization['time img']['mean']) / normalization['time img']['std']
        _Cluster_info = (_Cluster_info - normalization['Cluster_info']['mean']) / normalization['Cluster_info']['std']
        _Slice_info = (_Slice_info - normalization['Slice_info']['mean']) / normalization['Slice_info']['std']
        return torch.from_numpy(_Slice_img).to(torch.float), \
                torch.from_numpy(_Cluster_info).to(torch.float), \
                torch.from_numpy(_Slice_info).to(torch.float), \
                torch.from_numpy(_supernova).to(torch.long)
    def __len__(self):
        return self.len
# %%
# create a dataset
newDataset = MyDataset(filelist)

# define the batch size
batch_size_train = 10
batch_size_test = 10

train_size = int(0.8 * len(newDataset))
test_size = len(newDataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(newDataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train,
                                            shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test,
                                            shuffle=False)  
# %%
len(trainloader.dataset)
# %%
len(testloader.dataset)
# %%
net=testmodel.ClassifierModel(numClusterFeature = 19, numSlice = 4, numSliceFeature = 20).cuda()
# %%
import torch.optim as optim
import torch.nn as nn
criterion = nn.CrossEntropyLoss()

# Increasing the learning rate helps some models learn faster, so that may be worth trying
# Also, a learning rate schedule, or using an optimizer with a built in learning rate schedule (e.g. Adagrad)
# may be beneficial to helping the model learn.
optimizer = optim.Adam(net.parameters(), lr=0.01)
# %%
sum(p.numel() for p in net.parameters())

# %%
torch.cuda.get_device_name(0)

# %%
# Training
loss_list = []
epochs = 100
accuracy_list = []

for i in range(epochs):

    net.train() # begin training

    for (batch_idx, batch) in enumerate(trainloader):
        Slice_img_train_batch = batch[0].cuda() # remove .cuda() if you don't have a GPU
        Cluster_train_batch = batch[1].cuda() # remove .cuda() if you don't have a GPU
        Slice_info_train_batch = batch[2].cuda() # remove .cuda() if you don't have a GPU
        sig_train_batch = batch[3].cuda() # remove .cuda() if you don't have a GPU

        Netout = net.forward(Slice_img_train_batch, 
                                Cluster_train_batch, 
                                Slice_info_train_batch) 
        # This will call the forward function, usually it returns tensors.
        
        #print(F.softmax(Netout))


        loss = criterion(Netout, sig_train_batch) # classification loss        
        
        # Zero the gradients before running the backward pass.
        optimizer.zero_grad() 
        
        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()
        
        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

        loss_list.append(loss)
        if batch_idx % 50 == 0 or True:
            #print("Epoch: {}, batch: {} Loss: {} label_loss:{}".format(i, batch_idx, loss, label_loss_))
            print("Epoch: {}, batch: {} Loss: {:0.4f}".format(i, batch_idx, loss))
    
    net.eval() # begin testing
    preds = np.array([])
    reals = np.array([])


# %%
with torch.no_grad():
    for (batch_idx, batch) in enumerate(testloader):
        XZ_test_batch = batch[0].cuda() # remove .cuda() if you don't have a GPU
        YZ_test_batch = batch[1].cuda() # remove .cuda() if you don't have a GPU
        sig_test_batch = batch[2].cuda() # remove .cuda() if you don't have a GPU

        Netout = net.forward(XZ_test_batch, YZ_test_batch) # This will call the forward function, usually it returns tensors.
        #print(Netout.shape)
        prediction=F.softmax(Netout, dim=1).argmax(dim=1)
        

        preds=np.concatenate((preds, prediction.cpu().detach().numpy().flatten()))
        reals=np.concatenate((reals, sig_test_batch.cpu().detach().numpy().flatten()))
    preds=np.array(preds)
    reals=np.array(reals)
    accuracy=np.mean(preds==reals)
    accuracy_list.append(accuracy)
    print("Test accuracy: {}".format(accuracy))

# %%

# %%
