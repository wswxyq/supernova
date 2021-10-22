from os import listdir
import sys
import numpy as np
import os
from os.path import isfile, join
import random

# argv[1] is the path to the directory containing the .txt files
# argv[2] is the path to the directory to write the .npy files to

# image size
z_size = 448
xy_size = 384

divs=40

def image_gen(event):
    """
    generate npy from event
    """
    event=event[event[:, 2].argsort()]
    subevents=np.array_split(event, divs)
    #print(event)
    imgxz   =   np.zeros((2, z_size, xy_size), dtype=float)
    imgyz   =   np.zeros((2, z_size, xy_size), dtype=float)

    imgxzlist = []
    imgyzlist = []
    num_vallist = []

    for j in range(divs):
        for i in range(subevents[j].shape[0]):
            if subevents[j][i,0]%2 == 0:
                imgxz[0, int(subevents[j][i,0]/2), subevents[j][i, 1]] = subevents[j][i, 3]/4096
                imgxz[1, int(subevents[j][i,0]/2), subevents[j][i, 1]] = (subevents[j][i, 2]/5000000)-0.5
            else:
                imgyz[0, int((subevents[j][i, 0]-1)/2), subevents[j][i, 1]] = subevents[j][i, 3]/4096
                imgyz[1, int((subevents[j][i, 0]-1)/2), subevents[j][i, 1]] = (subevents[j][i, 2]/5000000)-0.5
        imgxzlist.append(imgxz)
        imgyzlist.append(imgyz)
        num_vallist.append(np.sum(subevents[j][:, 4]))
    return imgxzlist, imgyzlist, num_vallist


def npygen(filename):
    print(filename)
    imxzlist, imyzlist, max_vallist = image_gen(np.loadtxt(os.path.join(sys.argv[1], filename), dtype=int))
    for i in range(divs):
        np.savez(os.path.join(sys.argv[2], filename[:-4]+'_'+str(i)), imxz=imxzlist[i], imyz=imyzlist[i], sig=max_vallist[i])


import multiprocessing as mp
eventfiles=listdir(sys.argv[1])[0:40]
random.shuffle(eventfiles)
if __name__=="__main__": # simple python multiprocessing
    p=mp.Pool(os.cpu_count())
    p.map(npygen, eventfiles) 
    p.close()
    p.join()