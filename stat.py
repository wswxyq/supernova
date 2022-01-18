# %%
import numpy as np
import sys
sys.path.append('D:/supernova/utils')
import utils.select_id as select_id
import utils.cluster as clst
import utils.slice as slc
testfile = 'event/890.txt'
events=select_id.quick_select_id(testfile)


# %%
ids=events.idmap.keys() # get all ids
slice_lst=[] # list of slices with xz and yz planes
for i in ids:
    if i>0:
        temp=slc.slice_base(events.select(i))
        if temp.both==1:
            #print(i)
            slice_lst.append(slc.slice_image(events.select(i)))

# %%
xyzclusterlst=[] # list of clusters with xz and yz planes
for i in ids:
    if i<0:
        #print(i)
        event=events.select(i)
        b=clst.cluster_info(events.select(i))
        if b.both==1:
            xyzclusterlst.append(i)
            print(b.is_supernova)
# %%
# do a stat on the number of close slices of each cluster
num=[]
for j in xyzclusterlst:
    if j<0:
        b=clst.xyz_cluster(events.select(j))
        b.get_close_slice_num(slice_lst)
        print(j, b.close_slice_num)
        num.append(b.close_slice_num)

# %%
np.mean(num)
# %%
len(slice_lst)/len([i for i in ids if i>0])
# %%
len(xyzclusterlst)/len([i for i in ids if i<0])

# %%
slc.slice_image(events.select(91))# %%

# %%
events.select(91)

# %%
import uuid
str(uuid.uuid4('dasfsf'))
# %%
b=clst.xyz_cluster(events.select(-10))
b.get_close_slice_num(slice_lst)
b.close_slice_list
# %%
