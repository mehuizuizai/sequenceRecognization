import os
import numpy as np
from PIL import Image

sdir = 'E:/胎号/338-2_new/'#来源
ddir = 'E:/胎号/338_new_cut/'#目的

list = os.listdir(sdir)

count = 0
for i in list:
    index = i.find('(')
    if index == -1:#若没找到
        index = i.find('.')

    img = np.array(Image.open(sdir+i))
    len = img.shape[1]
    for j in range(index):
        tmp = img[:,int(len * j/index):int(len * (j+1)/index)]
        tmp = Image.fromarray(tmp)
        tmp.save(ddir+i[:-4]+'_'+i[j]+str(j)+'.jpg')
        count += 1
print(count)