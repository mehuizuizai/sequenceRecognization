# -*-  coding: utf-8 -*-
import time
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import mnist_inference
import  model_train_continue
import mnist_train
import numpy as np
import  mnist_train_change1

#每十秒加载一次最新的模型，并在测试模型上测试最新模型的正确率
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
SIZE = 784*3
EVAL_INTERVAL_SECS = 10
BATCH_SIZE = 300
NUM_CLASSES = 32
LETTERS_DIGITS = ("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","P","R","S","T","U","V","X","Y","Z")
RANGE_DIR = ["0","1","2","3","4","5","6","7","8","9"]

# RANGE_DIR = ["A","B","C","D","E","F","G","H","I","J","K","L","M","P","R","S","T","U","V","X","Y","Z"]
def evaluate():
    with tf.Graph().as_default() as g:
    #定义输入输出格式
        x = tf.placeholder(tf.float32,[None,mnist_inference.IMAGE_WIDTH,mnist_inference.IMAGE_HEIGHT,mnist_inference.NUM_CHANNELS],name= "x-input")
        y_ = tf.placeholder(tf.float32,[None, mnist_inference.OUT_PUT], name = "y-input")
		#reshape_xs = mnist.validation.images
	#-----------------------------------
        input_count = 0
        # for i in range(0,NUM_CLASSES):
        for i in RANGE_DIR:
            dir = 'E:/胎号所有数字_字母/测试_图片大于50resize_28/%s/' % i
            try:
                os.listdir(dir)
            except Exception:
                continue
            for rt, dirs, files in os.walk(dir):
                for filename in files:
                    input_count += 1
        print(input_count)
        # 定义对应维数和各维长度的数组
        test_images = np.array([[0.0] * SIZE for i in range(input_count)])
        print(test_images.shape)
        test_labels = np.array([[0.0] * NUM_CLASSES for i in range(input_count)])
        # print(input_labels.shape)
        index = 0
        datas_test= []
        result = np.array([])
        # 下标
        change_to_num = -1
        flag = 0 #标记第一个文件夹
        for i in RANGE_DIR:

            # dir = './all_testa1/%s/' % i
            # dir = 'E:/胎号/338-2_new/分割后resize/%s/' % i
            dir = 'E:/胎号所有数字_字母/测试_图片大于50resize_28/%s/' % i
            # dir = 'E:/胎号/338-2_new/分割后RESIZE/%s/' % i
            try:
                os.listdir(dir)
            except Exception:
                continue
            print(i)
            #---判断eval 首字母是数字还是字母
            if flag ==0 and i.isalpha():
                change_to_num += 11
            else :
                change_to_num +=1
            flag =1
            #----
            for rt, dirs, files in os.walk(dir):
                # for filename in files:
                  for filename in files:

                    filename = dir + filename
                    img = Image.open(filename)
                    height = img.size[0]
                    width = img.size[1]
                    image_arr = np.array(img) / 255.0
                    image_arr_reshape = np.reshape(image_arr, (2352))
                    result = np.concatenate((result, image_arr_reshape))
                    test_labels[index][change_to_num] = 1
                    index += 1
        result = result.reshape(input_count, 784 * 3)

        reshaped_xs = np.reshape(np.array(result), [-1, 28, 28,3])

        validate_feed = {x:reshaped_xs,y_: test_labels}


            #前向传播，此处由于是测试不关心正则化，所以参数设置为None
        y = mnist_inference.inference(x, True, None)
     #   print("y shape is:",y.shape)
            #使用前向传播计算正确率，当要对位置的样例进行分类的时候，用argmax（y，1）就可以了
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #通过变量重命名的方式加载模型，在前向传播过程便不用调用滑动平均函数求获取平均值了
        variable_average = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
            #本来要用滑动平均的名子取值，现在用variable_average.variables_to_restore()能直接取到所有值
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

            #每隔EVAL_INTERVAL_SECS秒调用一次计算正确的过程，以检测训练过程中的正确率变化
        # while True:
        license_num = ""
        with tf.Session(config=config) as sess:
                    # tf.train.get_check_point_state 函数会通过checkpoint文件自动找到目录中最新模型的文件名
                    #     ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH_ALL_NUMBER_AND_CHAR_NotAug)
                    #     ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH_ALL_NUMBER_AND_CHAR_shear)
                        ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH_ALL_FINAL)
                        #ckpt.model_checkpoint_path是模型的存储位置，不需要提供模型的名字，它会去查看checkpoint文件，使用最新的值
                        if ckpt and ckpt.model_checkpoint_path:
                            #加载模型
                            saver.restore(sess, ckpt.model_checkpoint_path)
                            #通过文件名找到模型保存时迭代的轮数
                            #python中 【-1】是指最后一个
                            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                            accuracy_score = sess.run(accuracy, feed_dict= validate_feed)

                            print("after %s training step , test accuracy = %0.5f%%" %(global_step, accuracy_score*100))
                            result = sess.run(y, feed_dict=validate_feed)

                            for i in range(result.shape[0]):
                                max1 = 0
                                max2 = 0
                                max3 = 0
                                max1_index = 0
                                max2_index = 0
                                max3_index = 0
                                for j in range(NUM_CLASSES):
                                    if result[i][j] > max1:
                                        max1 = result[i][j]
                                        max1_index = j
                                        continue
                                    if (result[i][j] > max2) and (result[i][j] <= max1):
                                        max2 = result[0][j]
                                        max2_index = j
                                        continue
                                    if (result[i][j] > max3) and (result[i][j] <= max2):
                                        max3 = result[i][j]
                                        max3_index = j
                                        continue

                                license_num = license_num + LETTERS_DIGITS[max1_index]
                                print("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (
                                LETTERS_DIGITS[max1_index], max1 * 100, LETTERS_DIGITS[max2_index], max2 * 100,
                                LETTERS_DIGITS[max3_index], max3 * 100))

                            print("轮胎编号是: 【%s】" % license_num)

                        else:
                            print("no checkpoint file found")
                            return
                        time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
	evaluate()

if __name__ == '__main__':
	tf.app.run()