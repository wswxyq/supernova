import numpy as np
import slice as slice
import cluster as cluster

def space_cut(slice_obj: slice.slice_image, cluster_obj:cluster.cluster_info):
    # the NOvA detector cell is 6cm deep in z direction, and 3.9cm wide in x and y direction.
    # If any of the hits in clusters are within the given distance to end of slice, return True.
    xydis=100
    zdis=100
    for i in range(len(cluster_obj.xzevent)):
        if np.abs(cluster_obj.xzevent[i, 0] - slice_obj.end_zloc_in_xz) < xydis and np.abs(cluster_obj.xzevent[i, 1] - slice_obj.end_xloc_in_xz) < zdis:
            return True

    for i in range(len(cluster_obj.yzevent)):
        if np.abs(cluster_obj.yzevent[i, 0] - slice_obj.end_zloc_in_yz) < xydis and np.abs(cluster_obj.yzevent[i, 1] - slice_obj.end_yloc_in_yz) < zdis:
            return True
    return False