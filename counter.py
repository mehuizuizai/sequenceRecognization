import os
import numpy as np
dir = 'E:/胎号/338_2/338_2/'
list = os.listdir(dir)

num = np.zeros(255)
for sub_dir in list:
    sub_list =os.listdir(dir + sub_dir)
    num[ord(sub_dir)] += len(sub_list)

for i in range(255):
    if num[i] != 0:
        print(chr(int(i))+':',int(num[i]))