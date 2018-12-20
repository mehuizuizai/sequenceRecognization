
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
# LETTERS_DIGITS = ("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z")
LETTERS_DIGITS = ("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","P","R","S","T","U","V","X","Y","Z")
RANGE_DIR = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","P","R","S","T","U","V","X","Y","Z"]

def evaluate():

    with tf.Graph().as_default() as g:
    #定义输入输出格式
        x = tf.placeholder(tf.float32,[None,mnist_inference.IMAGE_WIDTH,mnist_inference.IMAGE_HEIGHT,mnist_inference.NUM_CHANNELS],name= "x-input")
        y_ = tf.placeholder(tf.float32,[None, mnist_inference.OUT_PUT], name = "y-input")
		#reshape_xs = mnist.validation.images
	#-----------------------------------
        sdir = 'E:/胎号图片_整体/301_3/'  # 来源
        ddir = 'E:/胎号图片_整体/301_3new_save/'  # 目的
        if not os.path.exists(ddir):
            os.makedirs(ddir)
        list = os.listdir(sdir)
        len_list =  len(list)
        print("len_list is" ,len_list)
        num = 9  # 序列号字符个数
        count = 0
        eval_right_number =0
        for i in list:

            #第几张图片的文件夹
            if not os.path.exists(ddir + i):
                os.makedirs(ddir + i)
            img = np.array(Image.open(sdir + i))
            len1 = img.shape[1]
            mean_len = int(len1 / num)
            erro = int(mean_len * 0)
            RangeOrderCarNum = []
            for j in range(num):
                if not os.path.exists(ddir +i+'/'+ i[j]):
                    os.makedirs(ddir +i+'/'+ i[j])
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
                tmp_resize.save(ddir +i+"/"+ i[j] + "/" + i + "_" + i[j] + '.jpg')
                RangeOrderCarNum.append(i[j])
            print("RangeOrderCarNum is ",RangeOrderCarNum)
            input_count = 0
            # for i in range(0,NUM_CLASSES):
            for m in RangeOrderCarNum:
                # dir = 'E:/胎号/338-2_new/分割后resize/%s/' % i
                # dir = 'E:/胎号/338-2_new/分割后RESIZE/%s/' % i
                if m not in RANGE_DIR:
                    continue
                dir = ddir+i+'/%s/' % m
                try:
                    os.listdir(dir)
                except Exception:
                    continue
                for rt, dirs, files in os.walk(dir):
                    for filename in files:
                        input_count += 1

            # 定义对应维数和各维长度的数组
            test_images = np.array([[0.0] * SIZE for i in range(input_count)])

            test_labels = np.array([[0.0] * NUM_CLASSES for i in range(input_count)])
            # print(input_labels.shape)
            index = 0
            datas_test= []
            result_val = np.array([])
            # 下标
            change_to_num = -1
            flag = 0 #标记第一个文件夹
            for m in RangeOrderCarNum:
                if m not in RANGE_DIR:
                    continue
                # dir = './all_testa1/%s/' % m
                # dir = 'E:/胎号/338-2_new/分割后resize/%s/' % i
                dir = ddir+i+'/%s/' % m
                # dir = 'E:/胎号/338-2_new/分割后RESIZE/%s/' % m
                try:
                    os.listdir(dir)
                except Exception:
                    continue
                #---判断eval 首字母是数字还是字母
                # if flag ==0 and i.isalpha():
                #     change_to_num += 11
                # else :
                #     change_to_num +=1
                # flag =1
                change_to_num = RANGE_DIR.index(m)
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
                        result_val = np.concatenate((result_val, image_arr_reshape))
                        test_labels[index][change_to_num] = 1
                        index += 1
            result = result_val.reshape(input_count, 784 * 3)

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
                            tf.get_variable_scope().reuse_variables()
                # tf.train.get_check_point_state 函数会通过checkpoint文件自动找到目录中最新模型的文件名
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
                                # print(result.shape[0])
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
                                    # print("max1_index is:",max1_index)
                                    license_num = license_num + LETTERS_DIGITS[max1_index]
                                    # print(license_num)
                                    # print("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (
                                    # LETTERS_DIGITS[max1_index], max1 * 100, LETTERS_DIGITS[max2_index], max2 * 100,
                                    # LETTERS_DIGITS[max3_index], max3 * 100)


                                print("轮胎编号是: 【%s】" % license_num)
                                count = 0
                                if (len(license_num) == len(RangeOrderCarNum)):
                                    for i in range(len(license_num)):
                                        if (license_num[i] == RangeOrderCarNum[i]):
                                            count +=1
                                        else:
                                            break
                                    if(count==len(license_num)):
                                        eval_right_number += 1
                                        print(eval_right_number)

                            else:
                                print("no checkpoint file found")
                                return
                            time.sleep(EVAL_INTERVAL_SECS)

        print("准确率 ：",float(eval_right_number/len(list)))
        return license_num

def main(argv=None):

	evaluate()

if __name__ == '__main__':
	tf.app.run()
