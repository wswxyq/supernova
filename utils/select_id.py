import numpy as np

class quick_select_id:
    '''
    This class will read a txt file,
    sort the rows by the last column(slice/cluster number),
    and return the nth cluster/slice in select().
    '''
    
    def __init__(self, txtfile):
        self.txtfile = txtfile # string of txt file directory
        self.events=np.loadtxt(txtfile, dtype=int) # read txt file
        self.events=self.events[self.events[:,-1].argsort()] # sort by last column
        
        # map of id's to index locations. This allows us to find the indexes of the id very quickly.
        self.idmap={} # define an empty dictionary
        _id=self.events[0,-1] # set initial id to the first id in the first row
        _begin=0 # set initial begin to the first index
        _end=0 # set initial end to the first index
        for i in range(1, self.events.shape[0]): # Skip the first row because it is already set.
            _end=i # update end to the current index
            if self.events[i,-1]!=_id: # if the current id is different from the previous id
                self.idmap[_id]=[_begin, _end] # add the previous id and its indexes to the dictionary
                _id=self.events[i,-1] # update the id to the current id
                _begin=i # update the begin to the current index
        self.idmap[_id]=[_begin, _end+1] # add 1 to the end of the last index because the last index is exclusive.

    def select(self, n):
        return self.events[self.idmap[n][0]:self.idmap[n][1],:] # return the indexes of the id

