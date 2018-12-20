import tensorflow as tf
import os

# 定义神经网络结构参数 784个输入特征 10种分类  中间层有500个节点
# INPUT_NODE = 784 * 3
INPUT_NODE = 50176* 3
OUT_PUT = 24
# 传入的图片数据 28*28*1 的三维矩阵   标签为10维矩阵
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
# NUM_CHANNELS = 1
NUM_CHANNELS = 3
NUM_LABELS = 24

# 第一层卷积神经网络的深度和尺寸
CONV1_DEEP = 96
CONV1_SIZE = 11

# 第二层卷积神经网络的深度和尺寸
CONV2_DEEP = 256
CONV2_SIZE = 5
# 第三层卷积神经网络的深度和尺寸
CONV3_DEEP = 384
CONV3_SIZE = 3
# 第四层卷积神经网络的深度和尺寸
CONV4_DEEP = 384
CONV4_SIZE = 3
# 第五层卷积神经网络的深度和尺寸
CONV5_DEEP = 256
CONV5_SIZE = 3

# 全连接层节点个数
FC_SIZE = 4096


# LAYER1_NODE = 500

# train参数训练过程和测试过程。
# dropout方法可以进一步提升模型的可靠性并防止过拟合（只在训练中使用）
# 定义神经网络的前向传播
def inference(input_tensor, train, regularizer):
    # 第一层卷积 过滤器为 5，5，1，32。输入为28*28*1。输出为
    with tf.variable_scope('layer1_conv1'):
        conv1_weights = tf.get_variable(
            'weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(
            'bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights, strides=[1, 4, 4, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    #LRN层
    lrn1 = tf.nn.lrn(relu1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
    # 第二层池化(最大池化) 过滤器为 边长为2 步长为2    in：28*28*32。  out :14*14*32
    with tf.name_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(
            lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第三层卷积 过滤器 5 5 32 64 ，  in：14*14*32  out : 14*14*64
    with tf.variable_scope('layer3_conv2'):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(
            "bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    lrn2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
    # 第四层 池化 过滤器 变长为2 步长为2 in :14*14*64  out :7*7*64
    with tf.name_scope("layer4_pool2"):
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 第三层卷积 过滤器 5 5 32 64 ，  in：14*14*32  out : 14*14*64
    with tf.variable_scope('layer5_conv3'):
        conv3_weights = tf.get_variable(
            "weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable(
            "bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
        # 第四层卷积 过滤器 5 5 32 64 ，  in：14*14*32  out : 14*14*64
    with tf.variable_scope('layer5_conv4'):
        conv4_weights = tf.get_variable(
            "weight", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable(
            "bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3, conv4_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
        # 第五层卷积 过滤器 5 5 32 64 ，  in：14*14*32  out : 14*14*64
    with tf.variable_scope('layer5_conv5'):
        conv5_weights = tf.get_variable(
            "weight", [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable(
            "bias", [CONV5_DEEP], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(relu4, conv5_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))
        # 第三层池化(最大池化) 过滤器为 边长为2 步长为2    in：28*28*32。  out :14*14*32
    with tf.name_scope('layer3_pool3'):
        pool3 = tf.nn.max_pool(
            relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 把第四层输出转化为第五层全连接的输入格式
    pool_shape = pool3.get_shape().as_list()
    # print(pool_shape)
    # #pool_shape[0]为一个batch中 数据的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # print(nodes)
    print("pool3", pool3)
    # 输入向量四维，第一维就用的是多个向量一起处理的思想
    # reshaped = tf.reshape(pool2, [pool_shape[0], nodes])#将第四层的输出变成一个batch的响亮
    reshaped = tf.reshape(pool3, [-1, pool3.get_shape().as_list()[1] * pool3.get_shape().as_list()[2] *
                                  pool3.get_shape().as_list()[3]])  # 将第四层的输出变成一个batch的响亮
    # 第五层 全连接 in ：3136    out :512
    with tf.variable_scope('layer5_fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接的权重要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))

        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        # train 为 传入决定是否用dropout的参数
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 第六层：全连接层，512->10  输出通过softmax 后就得到最后分类的结果
    with tf.variable_scope("layer6_fc2"):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        # logit = tf.matmul(fc1,fc2_weights) + fc2_biases
        logit = tf.nn.softmax(tf.matmul(fc1, fc2_weights) + fc2_biases)
        print(logit)
    return logit
