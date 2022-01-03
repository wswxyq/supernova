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
b=clst.xyz_cluster(events.select(-1))
# %%
ids=events.idmap.keys()
slice_lst=[]
for i in ids:
    if i>0:
        print(i)
        slice_lst.append(slc.slice_image(events.select(i)))
# %%
b.get_close_slice_num(slice_lst))


# %%
temp=[]
for i in ids:
    if i<0:
        #print(i)
        event=events.select(i)
        b=clst.cluster_info(events.select(i))
        if b.both==1:
            temp.append(i)
# %%
len(ids)
# %%
len(ids)
# %%
len(temp)
# %%

# %%
slc.slice_image(events.select(91))# %%

# %%
events.select(91)
# %%
