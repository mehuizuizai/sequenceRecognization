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
NUM_CLASSES = 31
LETTERS_DIGITS = ("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","P","R","S","U","V","X","Y","Z")
RANGE_DIR = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","P","R","S","U","V","X","Y","Z"]
def evaluate():
    with tf.Graph().as_default() as g:
    #定义输入输出格式
        x = tf.placeholder(tf.float32,[None,mnist_inference.IMAGE_WIDTH,mnist_inference.IMAGE_HEIGHT,mnist_inference.NUM_CHANNELS],name= "x-input")
        y_ = tf.placeholder(tf.float32,[None, mnist_inference.OUT_PUT], name = "y-input")
        sdir = 'E:/胎号图片_整体/test/AXL01027U.jpg'
        num = 9  # 序列号字符个数
        count = 0
        # eval_right_number =0
        result = np.array([])
        print(np.shape(result))
        img = np.array(Image.open(sdir))
        len1 = img.shape[1]
        mean_len = int(len1 / num)
        erro = int(mean_len * 0)
        for j in range(num):
            start = int(len1 * j / num)
            end = int(len1 * (j + 1) / num)
            if ((start + erro) <= len1):
                start = start
                end = end + erro
            else:
                start = start
                end = len1
            tmp = img[:, start:end]
            tmp = Image.fromarray(tmp)
            tmp_resize = tmp.resize((28, 28), Image.ANTIALIAS)
            r, g, b = tmp_resize.split()  # rgb通道分离
            # 注意：下面一定要reshpae(1024)使其变为一维数组，否则拼接的数据会出现错误，导致无法恢复图片
            r_arr = np.array(r).reshape(784) / 255
            g_arr = np.array(g).reshape(784) / 255
            b_arr = np.array(b).reshape(784) / 255
            image_arr = np.concatenate((r_arr, g_arr, b_arr))
            result = np.concatenate((result, image_arr))
        result = result.reshape(num, 784 * 3)
        reshaped_xs = np.reshape(np.array(result), [-1, 28, 28,3])
        # validate_feed = {x:reshaped_xs,y_: test_labels}
        validate_feed = {x: reshaped_xs}
            #前向传播，此处由于是测试不关心正则化，所以参数设置为None
        y = mnist_inference.inference(x, True, None)
                #使用前向传播计算正确率，当要对位置的样例进行分类的时候，用argmax（y，1）就可以了
                #通过变量重命名的方式加载模型，在前向传播过程便不用调用滑动平均函数求获取平均值了
        variable_average = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
            #本来要用滑动平均的名子取值，现在用variable_average.variables_to_restore()能直接取到所有值
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
            #每隔EVAL_INTERVAL_SECS秒调用一次计算正确的过程，以检测训练过程中的正确率变化
        # while True:
        license_num = ""
        with tf.Session(config=config) as sess:
                        tf.get_variable_scope().reuse_variables()
            # tf.train.get_check_point_state 函数会通过checkpoint文件自动找到目录中最新模型的文件名
                        ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH_ALL_NUMBER_AND_CHAR_shear)
                        #ckpt.model_checkpoint_path是模型的存储位置，不需要提供模型的名字，它会去查看checkpoint文件，使用最新的值
                        if ckpt and ckpt.model_checkpoint_path:
                            #加载模型
                            saver.restore(sess, ckpt.model_checkpoint_path)
                            #通过文件名找到模型保存时迭代的轮数
                            #python中 【-1】是指最后一个
                            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                            # accuracy_score = sess.run(accuracy, feed_dict= validate_feed)
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
                            print("轮胎编号是: 【%s】" % license_num)
                            count = 0
                        else:
                            print("no checkpoint file found")
                            return
                        time.sleep(EVAL_INTERVAL_SECS)
    return license_num

def main(argv=None):

	evaluate()

if __name__ == '__main__':
	tf.app.run()
