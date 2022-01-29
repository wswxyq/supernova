## splitdata.py
A Python script to split a large CSV file into smaller txt files for every 5 ms window.

## utils\cluster.py
A Python script contains the class for cluster informations, which contains the function to find close slices of a cluster.

## utils\cuts.py
A Python script contains the cuts function to select the close slices of a cluster.

## utils\slice.py
A Python script contains the class for slice informations. The class contains the function to center-crop and resize the slice.

## utils\select_id.py
A Python script contains the code to initialize a class which contains the information of a single 5 ms event file. It allows you to select a cluster/slice by its ID.

## utils\myDataLoader.py
A Python script contains the function to produce the dataset for training by reading a single 5 ms event file. It saves every cluster with {num_slices} close slices in a single npz file.

## gendata.py
A Python script contains the multi-threaded code to generate the dataset for training by running the function in utils\myDataLoader.py over all the 5 ms event files.

## train.py
A Python script contains the code to train the model.