from os import listdir
import sys
import numpy as np
import os
from os.path import isfile, join
import random

# argv[1] is the path to the directory containing the .csv files
# argv[2] is the path to the directory to write the .npy files to

# image size
z_size = 448
xy_size = 384

divs=200

def shadow(pixel_loc, matrix, step=1):
    for i in range( np.max([pixel_loc[0]-step, 0]), np.min([pixel_loc[0]+step+1, matrix.shape[0]]) ):
        for j in range( np.max([pixel_loc[1]-step, 0]), np.min([pixel_loc[1]+step+1, matrix.shape[1]])):
            matrix[i, j] = 1
    # in place modification of the matrix



def image_gen(event):

    event=event[event[:, 2].argsort()] # sort the hits by time in nanoseconds

    ADC_mean = np.mean(event[:, 3])
    ADC_std = np.std(event[:, 3])

    subevents=np.array_split(event, divs) # divide the events into subevents
    #print(event)

    imgxzlist = [] # list of xz images
    imgyzlist = []  # list of yz images
    num_hit_list = [] # list of the number of hits in each image
    labelxzlst =  []
    labelyzlst =  []

    for j in range(divs):
        imgxz   =   np.zeros((2, z_size, xy_size), dtype=float)
        imgyz   =   np.zeros((2, z_size, xy_size), dtype=float)
        labelxz=[]
        labelyz=[]
        time_mean = np.mean(subevents[j][:, 2])
        time_std = np.std(subevents[j][:, 2])
        for i in range(subevents[j].shape[0]):
            if subevents[j][i,0]%2 == 0:
                imgxz[0, int(subevents[j][i, 0]/2), subevents[j][i, 1]] = (subevents[j][i, 3]-ADC_mean)/ADC_std
                imgxz[1, int(subevents[j][i, 0]/2), subevents[j][i, 1]] = (subevents[j][i, 2]-time_mean)/time_std
                if subevents[j][i, 4] == 1:
                    labelxz.append( [ int(subevents[j][i, 0]/2), subevents[j][i, 1] ] )
            else:
                imgyz[0, int((subevents[j][i, 0]-1)/2), subevents[j][i, 1]] = (subevents[j][i, 3]-ADC_mean)/ADC_std
                imgyz[1, int((subevents[j][i, 0]-1)/2), subevents[j][i, 1]] = (subevents[j][i, 2]-time_mean)/time_std
                if subevents[j][i, 4] == 1:
                    labelyz.append( [ int((subevents[j][i, 0]-1)/2), subevents[j][i, 1] ] )
        imgxzlist.append(imgxz)
        imgyzlist.append(imgyz)
        labelxzlst.append(np.array(labelxz, dtype=int))
        labelyzlst.append(np.array(labelyz, dtype=int))
        num_hit_list.append(np.sum(subevents[j][:, 4]))
    return imgxzlist, imgyzlist, num_hit_list, labelxzlst, labelyzlst


def npygen(filename):
    print(filename)
    imxzlist, imyzlist, num_hitlist, labelxzlst, labelyzlst = image_gen(np.loadtxt(os.path.join(sys.argv[1], filename), dtype=int))
    for i in range(divs):
        np.savez(os.path.join(sys.argv[2], filename[:-4]+'_'+str(i)), imxz=imxzlist[i], imyz=imyzlist[i], numhit=num_hitlist[i], labelxz=labelxzlst[i], labelyz=labelyzlst[i])


import multiprocessing as mp
all_files=listdir(sys.argv[1])
random.shuffle(all_files)
eventfiles=all_files[0:10]
if __name__=="__main__": # simple python multiprocessing
    p=mp.Pool(os.cpu_count())
    p.map(npygen, eventfiles) 
    p.close()
    p.join() 