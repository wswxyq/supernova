import numpy as np
import matplotlib.pyplot as plt
a=np.loadtxt('fd-1kpc-9.6sm-0-overlaid.csv/csv0/1112.txt')

z_size = 448
xy_size = 384
def image_gen(event):
    """
    generate npy from event
    """
    #print(event)
    imgxz=np.zeros((z_size, xy_size), dtype=int)
    imgyz=np.zeros((z_size, xy_size), dtype=int)

    for i in range(event.shape[0]):
        if event[i,0]%2 == 0:
            imgxz[int(event[i,0]/2), int(event[i, 1])] = event[i, 3]
            if event[i, 4] != 0:
                print(int(event[i,0]/2), int(event[i, 1]))
        else:
            imgyz[int((event[i, 0]-1)/2), int(event[i, 1])] = event[i, 3]
            if event[i, 4] != 0:
                print(int((event[i, 0]-1)/2), int(event[i, 1]))
    return imgxz, imgyz, event[:, 4].max()

b=a[a[:, 2].argsort()]
c=b[330000:340000, :]
print(c.shape[0], 'hits')
imxz, imyz, max_val = image_gen(c)
plt.figure(figsize=(10,10))
plt.imshow(imxz.T, cmap='jet')
plt.title('supernova neutrino number: '+str(c[:, 4].sum())+'\n time window:'+str((c[-1, 2]-c[0, 2]))+'ns', fontsize=20)
plt.show()
plt.figure(figsize=(10,10))
plt.imshow(imyz.T, cmap='jet')
plt.title('supernova neutrino number: '+str(c[:, 4].sum())+'\n time window:'+str((c[-1, 2]-c[0, 2]))+'ms', fontsize=20)
plt.show()