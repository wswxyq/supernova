# %%
import os
import utils.myDataLoader as myDataLoader

eventfolder='event'
filelist=os.listdir(eventfolder)
numSlice=4
numCluster=100

def gen_from_eventfile(eventfile:str):
    savedir=os.path.join('output', eventfile.replace(".txt", "")) # output folder
    os.makedirs(savedir, exist_ok=True)
    myDataLoader.createDataSet(eventfile=os.path.join(eventfolder, eventfile), num_slices=numSlice, SaveDir=savedir, num_cluster=numCluster)

# gen_from_eventfile('1020.txt')
# %%
import multiprocessing as mp

if __name__=="__main__": # simple python multiprocessing
    p=mp.Pool(os.cpu_count())
    p.map(gen_from_eventfile, filelist) 
    p.close()
    p.join()
# %%
