import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))

a=np.load('fd-1kpc-9.6sm-0-overlaid.csv/bkg_reduced_0/1067.npz')
b=np.load('fd-1kpc-9.6sm-0-overlaid.csv/npy0_reduced/1067.npz')
plt.imshow(a['imxz']-b['imxz'])
plt.show()
plt.figure(figsize=(10,10))

plt.imshow(a['imyz']-b['imyz'])
plt.show()