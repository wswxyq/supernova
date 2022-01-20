# need to do a stat on how many hits of the clusters are supernova hits
# I find no clusters
# %%
import numpy as np
import sys
sys.path.append('D:/supernova/utils')
import utils.select_id as select_id
import utils.cluster as clst
import utils.slice as slc
from os import listdir
import os
from multiprocessing import Manager, Pool
import random
import tqdm

# %%
# count the number of supernova hits in the cluster
eventfolder='event'
filelist=listdir(eventfolder)
shared_list=[]
for f in tqdm.tqdm(filelist):
    events=select_id.quick_select_id(os.path.join(eventfolder, f))
    ids=events.ids
    for i in ids:
        if i<0:
            #print(i)
            event=events.select(i)
            b=clst.cluster_info(event)
            temp=np.sum(b.event[:,4])
            if b.both==1 and temp>0:
                shared_list.append([temp, b.numhit])
# %%
shared_list=np.array(shared_list)
# %%
shared_list
# %%
np.mean(shared_list[:,0])/np.mean(shared_list[:,1])
# %%
import numpy as np
from matplotlib import pyplot as plt
shared_list=np.load('shared_list.npy')
info=np.divide(shared_list[:,0], shared_list[:,1])
plt.hist(info, bins=np.arange(0,1,0.05))
plt.show()

# %%
set(info)

# %%
(info>=0.5).sum()/info.size
# %%
np.mean(shared_list[:,0])
# %%
np.mean(shared_list[:,1])
# %%
ids
# %%
shared_list.shape
# %%
np.save('shared_list', shared_list)
# %%
info.size
# %%
shared_list[:,0].mean()
# %%
shared_list[:,1].mean()
# %%
