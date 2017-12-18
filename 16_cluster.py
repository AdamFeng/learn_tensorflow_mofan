'''
只显示 encoder 之后的数据， 并画在一个二维直角坐标系内。
将原有 784 Features 的数据压缩成仅剩 2 Features 的数据
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

# Visualize decoder setting
# Parameters
learning_rate = 0.001
training_epochs = 20 # 10组训练
batch_size = 256
display_step = 1

# Network Parameters
n_inputs = 784  # img shape: 28*28

# tf Graph input(only pictures)
X = tf.placeholder('float', [None, n_inputs])

# 通过四层hidden layers实现784 --> 2
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_inputs, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),

    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3])),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2])),
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1])),
    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_inputs])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),

    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([n_inputs])),
}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4'])
    return layer_4

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']), biases['decoder_b4']))
    return layer_4


# construct model
encoder_op = encoder(X) # 将原始784个feature encode到128个 features
decoder_op = decoder(encoder_op)    # 将得到的128个feature decode为784features

# prediction
y_pred = decoder_op # 经过encode and decode 后得到的feature
# Targets(Labels) are the input data
y_true = X  # 原始输入的784个feature

# define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true-y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)
    # training
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X:batch_xs})

        if epoch % display_step == 0:
            print('Epoch:', '%04d' % (epoch+1), 'cost=','{:.9f}'.format(c))

    print('Optimization Finished!')

    # 看解压之前的结果
    encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    plt.scatter(encoder_result[:,0], encoder_result[:,1], c=mnist.test.labels)
    plt.savefig('aaa.png')
    plt.show()
