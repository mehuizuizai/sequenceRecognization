import os
import numpy as np
from PIL import Image
import  shutil
RANGE_DIR = [0,1,2,3,4,5,6,7,8,9,"A","C","D","E","F","K","L","M","R","S","U","V","X","Y"]
sdir = 'E:/胎号所有/训练_图片大于50resize/'#来源
ddir = 'E:/胎号所有/训练_图片大于50_过采样/'#目的
for  i in RANGE_DIR :
    sdir1 = sdir+'%s/' % i
    ddir1 = ddir +'%s/' %i
    a = os.listdir(sdir1)
    if not os.path.exists(ddir1):
        os.makedirs(ddir1)
    if len(a)>=200:
        for j in a:
            shutil.copyfile(sdir1 + j, ddir1+ j)
    else:

        times = 1
        while(len(a)*(times-1)<200):
            for j in a:
                shutil.copyfile(sdir1 + j, ddir1 +str(times)+j)
            times += 1





