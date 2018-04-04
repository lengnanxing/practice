# -*- coding:utf-8 -*-
import tensorflow as tf
import  numpy as np

x = tf.placeholder(tf.float32, [None, 3])
W = tf.Variable(tf.zeros([3,1]))
b = tf.Variable(tf.zeros([1]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,1])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)






filename_queue = tf.train.string_input_producer(["train1.csv"])
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1],[1], [1], [1],[1], [1], [1],[1], [1]]
col1, col2, col3,co21,  col5,col6, col7, col8,col9, co22, co23 = tf.decode_csv(value, record_defaults = record_defaults)
features = tf.stack([col1, col2,co21])
init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()  # local variables like epoch_num, batch_size 可以不初始化local
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)



    for i in range(40):
        # Retrieve a single instan
        example, label = sess.run([features, co23])
        print(type(example))
        print(example)
        example.shape=1,3
        print(example.shape)
        aaa=np.array([label])
        aaa.shape=1,1
        temp=example
        example=example.vstack((example))
        train_step.run(feed_dict={x: p, y_: aaa})



    coord.request_stop()
    coord.join(threads)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


