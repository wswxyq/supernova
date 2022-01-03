# test utils/slice.py

# %%
import numpy as np
import matplotlib.pyplot as plt
import utils.slice as slice
import sys

# %%
f=np.loadtxt('event2', dtype=int)


# %%
f[f[:,0]%2==0].shape


# %%
slice2=slice.slice_image(f)

# %%
slice2.crop()
# %%
plt.imshow(slice2.original_map[0,:,:], cmap='jet')
# %%
plt.imshow(slice2.original_map[1,:,:], cmap='jet')
# %%
plt.imshow(slice2.original_map[2,:,:], cmap='jet')
# %%
plt.imshow(slice2.original_map[3,:,:], cmap='jet')
# %%
np.sum(slice2.original_map[1,:,:])
# %%
slice2.numhit
# %%
(slice2.img_size-(slice2.xmax - slice2.xmin))//2
# %%
(slice2.img_size-(slice2.ymax - slice2.ymin))//2
# %%
(slice2.img_size-(slice2.zmax - slice2.zmin))//2

# %%
slice2.resize()
# %%
plt.imshow(slice2.resized_map[0,:,:], cmap='jet')
# %%
plt.imshow(slice2.resized_map[1,:,:], cmap='jet')
# %%
plt.imshow(slice2.resized_map[2,:,:], cmap='jet')
# %%
plt.imshow(slice2.resized_map[3,:,:], cmap='jet')
# %%
sys.getsizeof(slice2.original_map)
# %%
sys.getsizeof(slice2.resized_map)
# %%
type(slice2.original_map[0,:,:])
# %%
type(slice2.resized_map[0,:,:])
# %%
slice2.resized_map.shape
# %%
