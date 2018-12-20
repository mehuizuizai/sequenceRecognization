import os
import tensorflow as tf
from PIL import Image
RANGE_DIR = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
# RANGE_DIR = ["0"]
cwd = os.getcwd()
writer = tf.python_io.TFRecordWriter("train.tfrecords_test_rotation_new")
label = -1
for i in RANGE_DIR:

    dir = 'E:/胎号所有/测试_图片大于50resize_28/%s/' % i
    # dir = './all_train_number_and_char/%s/' % i
    try:
        os.listdir(dir)
    except Exception:
        continue
    label += 1
    print(label)
    for rt, dirs, files in os.walk(dir):
        # for filename in files:
        for filename in files:
            imgpath =  dir+filename
            img = Image.open(imgpath)
            img_raw = img.tobytes()
    #
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    writer.write(example.SerializeToString())  # 序列化为字符串
writer.close()
