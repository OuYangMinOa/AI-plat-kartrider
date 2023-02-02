# process  data

import glob, os
import numpy as np
from random import shuffle

DATAFLODER = "dataset"
SAVED_DATAFLODER = "processed_data"

for i in glob.glob(DATAFLODER+"\\*.npy"):
    data = [[] for i in range(6)]
    data_this = np.load(i,allow_pickle=True)
    for each_data in data_this:

        data[np.argmax(each_data[1])].append(each_data)


    left_num = min( (len(data[1]),len(data[2]),len(data[3])) )
    final_data = []
    for z in range(6):
        print(len(data[z]), end=" ")
        final_data = final_data +  data[z][:left_num]
    print(len(final_data))
    shuffle(final_data)
    np.save(os.path.join(SAVED_DATAFLODER,os.path.basename(i)),final_data)





