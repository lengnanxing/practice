import  csv
import numpy as np
f0=csv.reader(open("train0.csv","r"))
train0=[]
for tr in f0:
    train0.append(tr)
train1=[]
y_train=[]
for i in range(len(train0)):
    y_train.append(train0[i][1])
    train1.append([train0[i][3],train0[i][4],train0[i][5]])
"""""""""""""""
print((np.array(train1)).shape)
for i in range(89):
    train2=[]
    y_test2=[]
    for j in range(10):
        train2.append(train1[i*10+j])
        y_test2.append(y_train[i*10+j])
    train_feature=np.array(train2)
    train_label=np.array(y_test2).reshape((10,1))

##test data
test_feature0=[]
test_label0=[]
for i in range(100):
    test_feature0.append(train1[i])
    test_label0.append(y_train[i])
test_feature=np.array(test_feature0)
test_label=np.array(test_label0).reshape(100,1)
print(test_label)





##m=np.array(train0)
##print(np.hsplit(m,(3,6)))
"""""""""""""""""""""""""""
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
##for i in range(79):
##for train data
train2=[]
y_test2=[]
for j in range(10):
    train2.append(train1[10+j])
    y_test2.append(y_train[10+j])
train_feature=np.array(train2)
train_label=np.array(y_test2).reshape((10,1))
    ##section running
    ##print(train_label)
print(train_label)
sess.run(train_step, feed_dict={x:train_feature , y_: train_label})
print(sess.run(b))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
test_feature0=[]
test_label0=[]
##for test data
for i in range(790,890):
    test_feature0.append(train1[i])
    test_label0.append(y_train[i])
test_feature=np.array(test_feature0)
test_label=np.array(test_label0).reshape(100,1)
##for accuracy
##print(tf.argmax(y,1))
print (sess.run(W, feed_dict={x: test_feature, y_:test_label}))