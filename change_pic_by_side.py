import os
import numpy as np
from PIL import Image

sdir = 'E:/胎号/338-2_new/'  # 来源
ddir = 'E:/胎号/338-2_new/分割后/'  # 目的
if not os.path.exists(ddir):
    os.makedirs(ddir)
list = os.listdir(sdir)
num = 9  # 序列号字符个数
count = 0
for i in list:
    img = np.array(Image.open(sdir + i))
    len = img.shape[1]
    mean_len = int(len / num)
    erro = int(mean_len * 0.1)
    len =len+erro*2

