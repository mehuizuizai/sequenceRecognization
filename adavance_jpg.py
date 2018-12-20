from PIL import Image, ImageEnhance
import pytesseract
import cv2
import os
import numpy as np
RANGE_DIR = ["0","1","2","3","4","5","6","7","8","9","A","C","D","E","F","K","L","M","R","S","U","V","X","Y"]
kernel_sharpen_3 = np.array([
        [-1,-1,-1,-1,-1],
        [-1,2,2,2,-1],
        [-1,2,8,2,-1],
        [-1,2,2,2,-1],
        [-1,-1,-1,-1,-1]])/8.0
def advance_jpg(num):
    outfile = "E:/胎号所有/训练_图片大于50resize_28_advance/" + str(num)
    if not os.path.exists(outfile):
        os.makedirs(outfile)
    for rt, dirs, files in os.walk('E:/胎号所有/训练_图片大于50resize_28/'+str(num)+'/'):
        for file in files:
            # print(file)
            # im = cv2.imread('E:/胎号所有/训练_图片大于50resize_28/'+str(num)+'/'+file)
            im =  Image.open('E:/胎号所有/训练_图片大于50resize_28/'+str(num)+'/'+file)
            im_arr = np.array(im)
            output_3 = cv2.filter2D(im_arr, -1, kernel_sharpen_3)
            tmp = Image.fromarray(output_3)
            # cv2.imwrite("E:/胎号所有/训练_图片大于50resize_28_advance/" + str(num)+'/' +file, output_3)
            tmp.save('E:/胎号所有/训练_图片大于50resize_28_advance/'+str(num)+'/'+'_ad'+file)
for i in RANGE_DIR:
 advance_jpg(i)