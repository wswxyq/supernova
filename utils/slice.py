import numpy as np
import matplotlib.pyplot as plt
import cv2

# image size
z_size = 50
xy_size = 50

# padding size
pad_size = 20

class slice_base:
    '''
    This is the base class for all the slice classes.
    It does not require that the slice contains both xz and yz part.
    '''
    def __init__(self, event):
        self.event = event # read event
        # split event into xz and yz
        self.xzevent = event[event[:,0]%2==0] # xz event
        self.yzevent = event[event[:,0]%2==1] # yz event
        self.xz = int(self.xzevent.shape[0]!=0) # xz part is empty or not
        self.yz = int(self.yzevent.shape[0]!=0) # yz part is empty or not
        self.both = self.xz * self.yz # 1 if both parts are not empty

        self.zmin = np.min(self.event[:,0])
        self.zmax = np.max(self.event[:,0])

        self.total_ADC = np.sum(self.event[:,3]) # total ADC
        self.numhit = self.event.shape[0] # number of hits

        self.avg_time = np.mean(self.event[:,2]) # average time
        self.std_time = np.std(self.event[:,2]) # standard deviation of time
        self.maxtime = np.max(self.event[:,2]) # maximum time
        self.mintime = np.min(self.event[:,2]) # minimum time


class slice_image(slice_base):
    """
    This class requires that the slice contains both xz and yz part.
    [ A class that contains the event data and processes it. ]
    Event columns:(notes from Matt)
    0) Plane number (z position)
    1) Cell number (x or y position depending on even or odd plane)
    2) Time in nanoseconds
    3) ADC (kinda like energy, but depends on position)
    4) 1 if simulated hit, 0 if data hit
    5) depends on signs:
    Negative: Supernova-like cluster number -n, where the first cluster is
    numbered 1.

    Zero: Not in either a slice or cluster. This usually happens because
    the hit is completely isolated; I don't form 1-hit clusters. It can
    also happen if the cluster wanted to be larger than 7 hits, or if the
    hit is very high ADC, or if the hit is in the very beginning or end of
    the 5.005ms block that overlaps with the adjacent block.

    Positive: Slice number n, where the first slice is numbered 1.
    """
    #print(event)
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

        self.img_size = np.max((self.xmax - self.xmin, self.ymax - self.ymin, self.zmax - self.zmin )) + 1 + pad_size
        self.original_map=np.zeros((4, self.img_size, self.img_size)) # 4 channels: xz ADC, xz time, yz ADC, yz time
        self.resized_map=np.zeros((4, z_size, xy_size)) # 4 channels: xz ADC, xz time, yz ADC, yz time
        # fill xz img:
        self.shifted_xzevent = self.xzevent - np.array([self.zmin-(self.img_size-(self.zmax - self.zmin + 1))//2, 
                                self.xmin-(self.img_size-(self.xmax - self.xmin))//2, 0, 0, 0, 0])[None,:] # put the slice in the center of the image
        
        # fill yz img:
        self.shifted_yzevent = self.yzevent - np.array([self.zmin-(self.img_size-(self.zmax - self.zmin + 1))//2,
                                self.ymin-(self.img_size-(self.ymax - self.ymin))//2, 0, 0, 0, 0])[None,:] # put the slice in the center of the image
        

    def crop(self):
        '''return a numpy array of the cropped image. Slices are centered.'''
        for i in range(self.xzevent.shape[0]):
            self.original_map[0, self.shifted_xzevent[i,0], self.shifted_xzevent[i,1]] = self.shifted_xzevent[i,3] # ADC
            self.original_map[1, self.shifted_xzevent[i,0], self.shifted_xzevent[i,1]] = self.shifted_xzevent[i,2] # time
        
        for i in range(self.yzevent.shape[0]):
            self.original_map[2, self.shifted_yzevent[i,0], self.shifted_yzevent[i,1]] = self.shifted_yzevent[i,3] # ADC
            self.original_map[3, self.shifted_yzevent[i,0], self.shifted_yzevent[i,1]] = self.shifted_yzevent[i,2] # time

    def resize(self):
        '''resize the image to the desired size for CNN'''
        self.resized_map = cv2.resize(np.moveaxis(self.original_map, 0, -1), (z_size, xy_size), interpolation = cv2.INTER_AREA)
        self.resized_map = np.moveaxis(self.resized_map, -1, 0)
