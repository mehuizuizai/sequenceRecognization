from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from PIL import  Image
import  matplotlib as pyplot
import  numpy as np
import  scipy
import  os
import   cv2 as cv
from keras.preprocessing import  image
# RANGE_DIR = [0,1,2,3,4,5,6,7,8,9,"A","C","D","E","F","K","L","M","R","S","U","V","X","Y"]
RANGE_DIR = ["T"]
# RANGE_DIR = [0]
shift =0.2
dir = "E:/胎号所有数字_字母/shengxia_train/"
sdir ="E:/胎号所有数字_字母/shengxia_train_shear/"
if not os.path.exists(sdir):
    os.makedirs(sdir)
a =  os.listdir(dir)
# width_shift_range=shift,height_shift_range=shift
# rotation_range=20
# channel_shift_range=100
# zoom_range=0.5
# shear_range=0.5
datagen = image.ImageDataGenerator(width_shift_range=shift,height_shift_range=shift)
for i  in RANGE_DIR:
    dir_new = "E:/胎号所有数字_字母/总_new/%s/" %i
    next_dir= "E:/胎号所有数字_字母/总_new/%s/%s/" %(i,i)
    list_dir = os.listdir(next_dir)
    len1 = len(list_dir)
    sdir_new ="E:/胎号所有数字_字母/总_new_shift/%s/" %i
    if not os.path.exists(sdir_new):
       os.makedirs(sdir_new)
    gen_data = datagen.flow_from_directory(dir_new, batch_size=1, shuffle=False, save_to_dir=sdir_new,
                                           save_prefix='shitf',target_size=(28, 28))
    # datagen.fit(gen_data)
    for j in range(len1):
        gen_data.next()


