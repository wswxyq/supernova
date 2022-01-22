# %%
from pathlib import Path
filelist=[]
for path in Path('output').rglob('*.npz'):
    filelist.append(path.__str__())
# %%
print(filelist[0])
# %%
import numpy as np
import tqdm as tqdm
import random as random

# %%
ADCimglist=[]
for file in tqdm.tqdm(filelist):
    data=np.load(file)
    ADCimglist.append(data['Slice_img'][[0,2,4,6,8,10,12,14,], :, :])
ADCimglist=np.array(ADCimglist)
# %%
ADCimglist.mean()
# %%
ADCimglist.std()

# %%
import utils.sizeof as sizeof
sizeof.sizeof_fmt(ADCimglist.size)
# %%
np.array(ADCimglist).reshape((-1, 50,50)).size


# %%
timeimglist=[]
for file in tqdm.tqdm(filelist):
    data=np.load(file)
    timeimglist.append(data['Slice_img'][[1,3,5,7,9,11,13,15], :, :])
timeimglist=np.array(timeimglist)
# %%
timeimglist.mean()
# %%
timeimglist.std()


# %%
cluster_feature_list=[]
for file in tqdm.tqdm(filelist):
    data=np.load(file)
    cluster_feature_list.append(data['Cluster_info'])
cluster_feature_list=np.array(cluster_feature_list)
# %%
cluster_feature_list.mean(axis=0)
# %%
cluster_feature_list.std(axis=0)
# %%
cluster_feature_list.shape

# %%
slice_feature_list=[]
for file in tqdm.tqdm(filelist):
    data=np.load(file)
    slice_feature_list.append(data['Slice_info'])
slice_feature_list=np.array(slice_feature_list)
# %%
slice_feature_list.mean(axis=0)
# %%
slice_feature_list.std(axis=0)
# %%
slice_feature_list.shape
# %%
