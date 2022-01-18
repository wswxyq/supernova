# need to do a stat on how many of the clusters are supernova clusters
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
eventfolder='event'
filelist=listdir(eventfolder)
shared_list=[]
for f in tqdm.tqdm(random.sample(filelist, 100)):
    
    supernovacount=0
    totalcount=0
    events=select_id.quick_select_id(os.path.join(eventfolder, f))
    ids=events.ids
    xyzclusterlst=[] # list of clusters with xz and yz planes
    for i in ids:
        if i<0:
            #print(i)
            event=events.select(i)
            b=clst.cluster_info(event)
            if b.both==1:
                totalcount+=1
                if b.is_supernova==1:
                    supernovacount+=1
    shared_list.append([supernovacount, totalcount])
# %%
shared_list=np.array(shared_list)
# %%
shared_list
# %%
np.mean(shared_list[:,0])/np.mean(shared_list[:,1])
# %%
from matplotlib import pyplot as plt
plt.hist(shared_list[:,0], bins=np.arange(0,100,10))
# %%
np.mean(shared_list[:,0])
# %%
np.mean(shared_list[:,1])
# %%
ids
# %%
