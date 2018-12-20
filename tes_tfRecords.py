import  tensorflow as tf
import get_img_from_TFRecords
img, label = get_img_from_TFRecords.read_and_decode("train.tfrecords5")

#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=30, capacity=200,
                                                min_after_dequeue=100,num_threads=3)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(3):
        print(i)
        val, l= sess.run([img_batch, label_batch])
        #我们也可以根据需要对val， l进行处理
        #l = to_categorical(l, 12)
        print(val.shape, l)
