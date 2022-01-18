import numpy as np
import sys
sys.path.append('D:/supernova/utils')
import utils.select_id as select_id
import utils.cluster as clst
import utils.slice as slc
import os
import uuid
import random

def createDataSet(eventfile:str, num_slices:int, SaveDir:str, num_cluster:int=-1):
    events=select_id.quick_select_id(eventfile) # read an event file
    ids=events.ids # get all ids

    slice_lst=[] # list of slices with xz and yz planes
    for i in ids:
        if i>0:
            temp=slc.slice_base(events.select(i))
            if temp.both==1:
                #print(i)
                slice_lst.append(slc.slice_image(events.select(i)))

    cluster_lst=[] # list of clusters with xz and yz planes
    for i in ids:
        if i<0:
            temp=clst.cluster_info(events.select(i))
            if temp.both==1:
                #print(i)
                cluster_lst.append(clst.xyz_cluster(events.select(i)))

    loop_cluster_lst=cluster_lst
    if num_cluster!=-1 and num_cluster<=len(cluster_lst):
        loop_cluster_lst=random.sample(cluster_lst, num_cluster)
    elif num_cluster>len(cluster_lst):
        print('Warning: num_cluster is larger than the number of clusters in the event file. \n\
                Set num_cluster to -1 to use all clusters. \n\
                Setting num_cluster = len(cluster_lst) now.')

    for x in loop_cluster_lst:
        x.get_close_slice_num(slice_lst)
        if x.close_slice_num>=num_slices:
            cluster_features=[
                x.zmin,
                x.zmax,
                x.total_ADC,
                x.numhit,
                x.std_time,
                x.maxtime-x.avg_time,
                x.mintime-x.avg_time,
                x.xmin,
                x.xmax,
                x.ymin,
                x.ymax,
                x.start_xloc_in_xz,
                x.end_xloc_in_xz,
                x.start_zloc_in_xz,
                x.end_zloc_in_xz,
                x.start_yloc_in_yz,
                x.end_yloc_in_yz,
                x.start_zloc_in_yz,
                x.end_zloc_in_yz,
            ]
            slice_features=[]
            Sliceimg=[]
            for j in range(num_slices):
                Sliceimg.append(slice_lst[x.close_slice_list[j,0]].resized_map)
                temp_slice=slice_lst[x.close_slice_list[j,0]]
                slice_features=slice_features+[
                        temp_slice.zmin,
                        temp_slice.zmax,
                        temp_slice.total_ADC,
                        temp_slice.numhit,
                        temp_slice.std_time,
                        temp_slice.avg_time-x.avg_time,
                        temp_slice.maxtime-temp_slice.avg_time,
                        temp_slice.mintime-temp_slice.avg_time,
                        temp_slice.xmin,
                        temp_slice.xmax,
                        temp_slice.ymin,
                        temp_slice.ymax,
                        temp_slice.start_xloc_in_xz,
                        temp_slice.end_xloc_in_xz,
                        temp_slice.start_zloc_in_xz,
                        temp_slice.end_zloc_in_xz,
                        temp_slice.start_yloc_in_yz,
                        temp_slice.end_yloc_in_yz,
                        temp_slice.start_zloc_in_yz,
                        temp_slice.end_zloc_in_yz,
                        ]
            cluster_features=np.array(cluster_features)
            slice_features=np.array(slice_features)
            Sliceimg=np.concatenate(Sliceimg, axis=0)
            np.savez(os.path.join(SaveDir, str( uuid.uuid4() ) ), Slice_img=Sliceimg, 
                    Cluster_info=cluster_features, Slice_info=slice_features)
 

