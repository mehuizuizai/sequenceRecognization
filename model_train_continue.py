# -*-  coding: utf-8 -*-
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
#加载自定义的前向传播（函数和常量）
import mnist_inference
import mnist_train
import numpy as np
from PIL import Image
#配置神经网络的参数
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
NUM_CLASSES = 32
INPUT_NODE =784*3
BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULRAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
#----pbpath
pb_file_path = os.getcwd()+"/pb_path/"
#模型保存的路径和文件名
#------7 是0-9数字的
# MODEL_SAVE_PATH = os.getcwd() + "/model10/"
# MODEL_NAME = "model10.ckpt"
MODEL_SAVE_PATH = os.getcwd() + "/model9/"
MODEL_NAME = "model9.ckpt"
MODEL_SAVE_PATH_only_number = os.getcwd() + "/model13/"
MODEL_NAME2 = "model13.ckpt"
MODEL_SAVE_PATHNumber_add_Char = os.getcwd() + "/model18/"
MODEL_NAME1 = "model18.ckpt"

#------------------------------------
MODEL_SAVE_PATH_AuData= os.getcwd()+"/model16/"
MODEL_NAME3 = "model16ckpt"
MODEL_SAVE_PATH_3FC= os.getcwd()+"/model21/"
MODEL_NAME5 = "model21ckpt"
MODEL_SAVE_PATH_3FC_Aug= os.getcwd()+"/model22/"
MODEL_NAME7 = "model22ckpt"
MODEL_SAVE_PATH_3FC_Aug_Rand= os.getcwd()+"/model23/"
MODEL_NAME8 = "model23ckpt"
MODEL_SAVE_PATH_3FC_Aug_Rand_clor= os.getcwd()+"/model24/"
MODEL_NAME9 = "model24ckpt"
MODEL_SAVE_PATH_3FC_Aug_Rand_clor_rh= os.getcwd()+"/model25/"
MODEL_NAME10= "model25ckpt"
MODEL_SAVE_PATH_3FC_Aug_Rand_clor_shift= os.getcwd()+"/model26/"
MODEL_NAME11= "model26ckpt"
MODEL_SAVE_PATH_3FC_Aug_Rand_clor_rh_add_layer= os.getcwd()+"/model27/"
MODEL_NAME12= "model27ckpt"
MODEL_SAVE_PATH_3FC_Aug_Rand_shift_color_rotaion= os.getcwd()+"/model28/"
MODEL_NAME13= "model28ckpt"
MODEL_SAVE_PATH_3FC_Aug_Rand_shift_color_rotaion_oher= os.getcwd()+"/model28/"
MODEL_NAME14= "model29ckpt"
MODEL_SAVE_PATH_ALL_NUMBER_AND_CHAR= os.getcwd()+"/model30/"
MODEL_NAME15= "model30ckpt"
MODEL_SAVE_PATH_ALL_NUMBER_AND_CHAR_rotation= os.getcwd()+"/model31/"
MODEL_NAME16= "model31ckpt"
MODEL_SAVE_PATH_ALL_NUMBER_AND_CHAR_NotAug = os.getcwd()+"/model32/"
MODEL_NAME17= "model32ckpt"
MODEL_SAVE_PATH_ALL_NUMBER_AND_CHAR_shear = os.getcwd()+"/model33/"
MODEL_NAME18= "model33ckpt"
MODEL_SAVE_PATH_ALL_NUMBER_AND_CHAR_LRN = os.getcwd()+"/model34/"
MODEL_NAME19= "model34ckpt"
MODEL_SAVE_PATH_ALL_NUMBER = os.getcwd()+"/model36/"
MODEL_NAME21 = "model36ckpt"
MODEL_SAVE_PATH_ALL_FINAL= os.getcwd()+"/model37/"
MODEL_NAME22 = "model37ckpt"
#-------------------------------
# RANGE_DIR = [0,1,2,3,4,5,6,7,8,9,"A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
# RANGE_DIR = [0,1,2,3,4,5,6,7,8,9,["A","B","C","D","E","F","J","P","V","Z"]]
# RANGE_DIR = [0,1,2,3,4,5,6,7,8,9,"A","B","C","D","E","F","G","H","I","J","K","L","M","P","R","S","U","V","X","Y","Z"]
RANGE_DIR = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","P","R","S","T","U","V","X","Y","Z"]
# RANGE_DIR = [0,1,2,3,4,5,6,7,8,9]
def train():
	#定义输入的placeholder
    x = tf.placeholder(
		tf.float32,[None,mnist_inference.IMAGE_WIDTH,mnist_inference.IMAGE_HEIGHT,mnist_inference.NUM_CHANNELS],name= "x-input")
	#正确答案
    y_ = tf.placeholder(
		tf.float32,[None,mnist_inference.OUT_PUT], name = "y-input")
    regularizer = tf.contrib.layers.l2_regularizer(REGULRAZTION_RATE)
	#使用定义好的前向传播
    y = mnist_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0,trainable = False)
	#定义损失函数、学习率、滑动平均操作并训练
    ema = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = ema.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
	#accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#-------------------------
    # learning_rate = tf.train.exponential_decay(
    input_count = 0
    # train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss,global_step = global_step)
    # train_step = tf.train.AdamOptimizer(0.00001).minimize(loss,global_step = global_step)
    train_step = tf.train.AdamOptimizer(0.000001).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step,variable_average_op]):
        train_op = tf.no_op(name= "train")
	#------------------------------------------------

    re_count = 0
    for i in RANGE_DIR:

        # dir = './all_train_number_and_char/%s/' % i
        dir = 'E:/胎号所有数字_字母/训练_图片大于50resize_28_all_add_shift_rotation/%s/' % i
        for rt, dirs, files in os.walk(dir):
            # for filename in files:
            for filename in files[:int(len(files) * 19//20)]:
                re_count += 1
    input_labels = np.array([[0.0] * NUM_CLASSES for i in range(re_count)])
    index = 0
    datas = []
    result=np.array([])
    # 0-9, A-Z (I,O不包括）转化为下标
    chan_to_num_train = -1
    for i in RANGE_DIR:
        print(i)
        # dir = './all_train_number_and_char/%s/' % i
        dir = 'E:/胎号所有数字_字母/训练_图片大于50resize_28_all_add_shift_rotation/%s/' % i
        try:
            os.listdir(dir)
        except Exception:
            continue
        chan_to_num_train+=1
        for rt, dirs, files in os.walk(dir):
            # for filename in files:
            for filename in files[:int(len(files)*19//20)]:
				# result = np.array([])
                filename = dir + filename
                img = Image.open(filename)

                height= img.size[1]
                width = img.size[0]
                image_arr = np.array(img) / 255.0
                image_arr_reshape = np.reshape(image_arr, (2352))
                result = np.concatenate((result, image_arr_reshape))
                input_labels[index][chan_to_num_train] = 1
                # print(input_labels)
                index += 1
    print("re_count is :", re_count)
    print(result.shape)
    result = result.reshape(re_count,784*3)
    print(result.shape)

		# -------------------验证集
    val_count =0
    val_images = np.array([[0] * INPUT_NODE for i in range(val_count)])
    re_count_val = 0
    for i in RANGE_DIR:
        # dir = './all_train_number_and_char/%s/' % i
        dir = 'E:/胎号所有数字_字母/训练_图片大于50resize_28_all_add_shift_rotation/%s/' % i
        try:
            os.listdir(dir)
        except Exception:
            continue
        for rt, dirs, files in os.walk(dir):
            for filename in files[int(len(files) * 19//20) + 1:]:
                re_count_val += 1
    val_labels = np.array([[0] * NUM_CLASSES for i in range(re_count_val)])
    val_index = 0
    datas = []
    result_val = np.array([])
    change_to_num_val = -1
    for i in RANGE_DIR:
        # dir = './all_train_number_and_char/%s/' % i
        dir = 'E:/胎号所有数字_字母/训练_图片大于50resize_28_all_add_shift_rotation/%s/' % i
        try:
            os.listdir(dir)
        except Exception:
            continue
        change_to_num_val +=1
        for rt, dirs, files in os.walk(dir):
            for filename in files[int(len(files)*19//20)+1:]:
                # re_count_val +=1
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
			#	datas.append(data)
                image_arr = np.array(img) / 255.0
                image_arr_reshape = np.reshape(image_arr, (2352))
                result_val = np.concatenate((result_val, image_arr_reshape))
                # val_images[val_index] = data.reshape(784)
                val_labels[val_index][change_to_num_val] = 1
                val_index += 1
    print("re_count_val is :",re_count_val)
	# print("datas shape is:",np.shape(datas))
    print("resule_val.shap",result_val.shape)
    result_val = result_val.reshape(re_count_val, 784 * 3)
    print("result_val.shape",result_val.shape)

    batches_count = int(re_count / BATCH_SIZE)
    remainder = re_count % BATCH_SIZE
    print("训练数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count + 1, BATCH_SIZE, remainder))
 #------------------------------------

    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH_ALL_FINAL)
        # ckpt.model_checkpoint_path是模型的存储位置，不需要提供模型的名字，它会去查看checkpoint文件，使用最新的值
        if ckpt and ckpt.model_checkpoint_path:

            # saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
            saver.restore(sess, ckpt.model_checkpoint_path)
		#在训练过程中不测试模型在验证集山的表看，验证和测试的过程由独立程序执行
        for i in range(TRAINING_STEPS):
			#xs, ys = mnist.train.next_batch(BATCH_SIZE)
			#----------------------------
            for n in range(batches_count):
                zzz, loss_value, step, iterate_accuracy = sess.run([train_op, loss, global_step, accuracy],
																   feed_dict={x: np.reshape(result[n * BATCH_SIZE:(n + 1) * BATCH_SIZE],
																							[BATCH_SIZE,mnist_inference.IMAGE_WIDTH,mnist_inference.IMAGE_HEIGHT,mnist_inference.NUM_CHANNELS]),
																				 y_: input_labels[n * BATCH_SIZE:(n + 1) * BATCH_SIZE]})
                if i % 5==0:
                    print("after %d train step, loss on trainable  batch is  %g." %(step,loss_value))
                    print('第 %d 次训练迭代: 准确率 %0.5f%%' % (i, iterate_accuracy * 100))
            if remainder > 0:
                start_index = batches_count * BATCH_SIZE;

                zzz, loss_value, step, iterate_accuracy = sess.run([train_op, loss, global_step, accuracy],
																   feed_dict={x: np.reshape(result[
																				 start_index:re_count],[remainder,mnist_inference.IMAGE_WIDTH,mnist_inference.IMAGE_HEIGHT,mnist_inference.NUM_CHANNELS]),
																			  y_: input_labels[
																				  start_index:re_count]})

			#----------------------------
			#每1000轮保存一次模型
            if i % 5 == 0:
                # print("after %d train step, loss on trainable  batch is  %g." %(step,loss_value))
                # print('第 %d 次训练迭代: 准确率 %0.5f%%' % (i, iterate_accuracy * 100))
                zz, val_loss_value, step, val_iterate_accuracy = sess.run([train_op, loss, global_step, accuracy],
																	  feed_dict={x: np.reshape(
																		  result_val,
																		  [re_count_val, mnist_inference.IMAGE_WIDTH,
																		   mnist_inference.IMAGE_HEIGHT, mnist_inference.NUM_CHANNELS]),
																		  y_: val_labels})
                print("valid lose is %g:",val_loss_value)
				# if iterate_accuracy >= 0.9999 and it >= iterations:
                print('第 %d 次训练迭代: 验证准确率 %0.5f%%' % (i, val_iterate_accuracy * 100))
                if val_iterate_accuracy >= 0.9995:
                    #存取模型格式为.pb
                    #------------
                    # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
                    # with tf.gfile.FastGFile(pb_file_path+str(i), mode='wb') as f:
                    #     f.write(constant_graph.SerializeToString())
                    #---------
                    saver.save(sess, MODEL_SAVE_PATH_ALL_FINAL+ MODEL_NAME22, global_step=global_step)
                    break;
                # 存取模型格式为.pb
                # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
                # with tf.gfile.FastGFile(pb_file_path+str(i), mode='wb') as f:
                #     f.write(constant_graph.SerializeToString())
                # ---------
                saver.save(sess, MODEL_SAVE_PATH_ALL_FINAL+ MODEL_NAME22, global_step = global_step)

def main(argv=None):
	# mnist = input_data.read_data_sets(os.getcwd()+ "/mnist/data",one_hot = True)
	train()

if __name__ == '__main__':
    tf.app.run()