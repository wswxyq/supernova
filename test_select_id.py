
# %%
import numpy as np
import sys
import utils.select_id as select_id
testfile = 'event/890.txt'
# %%
import time
'''This code verifies that the select_id.py code is working properly.'''
start = time.time()
events=select_id.quick_select_id(testfile)
valevent=np.loadtxt(testfile, dtype=int)

ids=np.unique(valevent[:,-1])

for i in ids:
    if len(valevent[valevent[:,-1]==i])!=len(events.select(i)):
        print('Error')
        print(i)
        print(len(valevent[valevent[:,-1]==i]))
        print(len(events.select(i)))
        #break

end = time.time()
print(end - start)

# %%
import time

start = time.time()
valevent=np.loadtxt(testfile, dtype=int)
end = time.time()
print("init quick:", end - start)

start = time.time()
events=select_id.quick_select_id(testfile)
end = time.time()
print("init st:", end - start)


ids=np.unique(valevent[:,-1])

start = time.time()
for i in ids:
    b=valevent[valevent[:,-1]==i]
end = time.time()
print("standard selector:", end - start)

start = time.time()
for i in ids:
    b=events.select(i)
end = time.time()
print("quick selector:", end - start)




# %%
