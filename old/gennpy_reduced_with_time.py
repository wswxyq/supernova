from os import listdir
import sys
import numpy as np
import os
from os.path import isfile, join

# argv[1] is the path to the directory containing the .txt files
# argv[2] is the path to the directory to write the .npy files to

# image size
z_size = 448
xy_size = 384

def image_gen(event):
    """
    generate npy from event
    """
    #print(event)
    imgxz   =   np.zeros((2, z_size, xy_size), dtype=float)
    imgyz   =   np.zeros((2, z_size, xy_size), dtype=float)


    for i in range(event.shape[0]):
        if event[i,0]%2 == 0:
            imgxz[0, int(event[i,0]/2), event[i, 1]] = event[i, 3]/4096
            imgxz[1, int(event[i,0]/2), event[i, 1]] = (event[i, 2]/5000000)-0.5
        else:
            imgyz[0, int((event[i, 0]-1)/2), event[i, 1]] = event[i, 3]/4096
            imgyz[1, int((event[i, 0]-1)/2), event[i, 1]] = (event[i, 2]/5000000)-0.5
    return imgxz, imgyz, event[:, 4].max()


def npygen(filename):
    print(filename)
    imxz, imyz, max_val = image_gen(np.loadtxt(os.path.join(sys.argv[1], filename), dtype=int))
    np.savez(os.path.join(sys.argv[2], filename.replace(".txt", "")), imxz=imxz, imyz=imyz, sig=max_val)


import multiprocessing as mp

if __name__=="__main__": # simple python multiprocessing
    p=mp.Pool(os.cpu_count())
    p.map(npygen, listdir(sys.argv[1])) 
    p.close()
    p.join()