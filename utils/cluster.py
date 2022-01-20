import numpy as np
import cuts as cuts

class cluster_info:
    '''
    This class will contains information of a candidate cluster.
    Note that a cluster has much less hits than a slice.
    Usually it has around 2 hits.
    XZ, YZ parts can be empty.
    '''
    def __init__(self, event):
        self.event = event # read event
        # split event into xz and yz
        self.xzevent = event[event[:,0]%2==0] # xz event
        self.yzevent = event[event[:,0]%2==1] # yz event
        self.xz = int(self.xzevent.shape[0]!=0) # xz part is empty or not
        self.yz = int(self.yzevent.shape[0]!=0) # yz part is empty or not
        self.both = self.xz * self.yz # 1 if both parts are not empty

        # get information that may be used as input of model.
        self.zmin = np.min(self.event[:,0])
        self.zmax = np.max(self.event[:,0])

        self.total_ADC = np.sum(self.event[:,3]) # total ADC
        self.numhit = self.event.shape[0] # number of hits

        self.avg_time = np.mean(self.event[:,2]) # average time
        self.std_time = np.std(self.event[:,2]) # standard deviation of time
        self.maxtime = np.max(self.event[:,2]) # maximum time
        self.mintime = np.min(self.event[:,2]) # minimum time
        self.close_slice_list=[]

        # is this a supernova neutrino hits cluster?
        self.is_supernova = ( np.sum(self.event[:,4])/self.numhit >= 0.5 )

class xyz_cluster(cluster_info):
    '''
    A special class for clusters which contain hits in both XZ and YZ plan.
    '''
    def __init__(self, event):
        super().__init__(event)
        # get information that may be used as input of model.
        self.xmin = np.min(self.xzevent[:,1]) # minimal x position
        self.xmax = np.max(self.xzevent[:,1]) # maximal x position
        self.ymin = np.min(self.yzevent[:,1])
        self.ymax = np.max(self.yzevent[:,1])

        self.start_xloc_in_xz = self.xzevent[np.argmin(self.xzevent[:,2])][1] # starting(early) x location in xz plane
        self.end_xloc_in_xz = self.xzevent[np.argmax(self.xzevent[:,2])][1] # ending(late) x location in xz plane
        self.start_zloc_in_xz = self.xzevent[np.argmin(self.xzevent[:,2])][0]
        self.end_zloc_in_xz = self.xzevent[np.argmax(self.xzevent[:,2])][0]

        self.start_yloc_in_yz = self.yzevent[np.argmin(self.yzevent[:,2])][1]
        self.end_yloc_in_yz = self.yzevent[np.argmax(self.yzevent[:,2])][1]
        self.start_zloc_in_yz = self.yzevent[np.argmin(self.yzevent[:,2])][0]
        self.end_zloc_in_yz = self.yzevent[np.argmax(self.yzevent[:,2])][0]

        self.close_slice_num = 0 # number of close slices

    def get_close_slice_num(self, slices_list: list):
        '''
        Get the slice that is close to this cluster. Use space_cut in cuts.py.
        '''
        for i in range(len(slices_list)):
            if cuts.space_cut(slices_list[i], self):
                self.close_slice_num += 1
                # save the minimum difference of this slice's hits time to this cluster's average time
                self.close_slice_list.append([i, np.min(np.abs( slices_list[i].event[:,2]-self.avg_time))])
        if self.close_slice_list!=[]:
            self.close_slice_list=np.array(self.close_slice_list, dtype=np.int)
            self.close_slice_list=self.close_slice_list[self.close_slice_list[:,-1].argsort()]
        
 