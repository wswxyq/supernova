import sys

# This function splits the large csv file into smaller files. Each file contains only one event.
# args: large_csv_file_name, output_dir_name
def split_data(filename, outdir: str):
    """
    split data file into smaller files
    """
    data=[]
    with open(filename, 'r') as f:
        eventnum = 0
        for line in f:
            if 'Event ' in line or line=='':
                if data!=[]:
                    textfile = open(outdir+str(eventnum)+".txt", "w")
                    for element in data:
                        textfile.write(element)
                    textfile.close()
                eventnum += 1
                data=[]
            else:
                data.append(line)
        textfile = open(outdir+str(eventnum)+".txt", "w") # I add this line because the last event is not saved. Anyway it works :)
        for element in data:
            textfile.write(element)
        textfile.close()

split_data(sys.argv[1], sys.argv[2])