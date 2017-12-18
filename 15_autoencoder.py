'''
自编码的前半部分称为encoder 编码器。
编码器能得到原数据的精髓, 然后我们只需要再创建一个小的神经网络学习这个精髓的数据,
不仅减少了神经网络的负担, 而且同样能达到很好的效果。
他能从原数据中总结出每种类型数据的特征, 如果把这些特征类型都放在一张二维的图片上,
每种类型都已经被很好的用原数据的精髓区分开来。
如果你了解 PCA 主成分分析, 再提取主要特征时, 自编码和它一样,甚至超越了 PCA。
换句话说, 自编码 可以像 PCA 一样 给特征属性降维。
'''

'''
在压缩环节：我们要把这个Features不断压缩，经过第一个隐藏层压缩至256个 Features，再经过第二个隐藏层压缩至128个。
在解压环节：我们将128个Features还原至256个，再经过一步还原至784个。
在对比环节：比较原始数据与还原后的拥有 784 Features 的数据进行 cost 的对比，根据 cost 来提升我的 Autoencoder 的准确率。
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

# Visualize decoder setting
# Parameters
learning_rate = 0.01
training_epochs = 5 # 五组训练
batch_size = 256
display_step = 1
example_to_show = 10

# Network Parameters
n_inputs = 784  # img shape: 28*28

# tf Graph input(only pictures)
X = tf.placeholder('float', [None, n_inputs])

# hidden layer settings
# 经过第一层将784个feature压缩到256个feature，
# 然后经过第二层压缩到128个feature
n_hidden_1 = 256    # 1st layer num features
n_hidden_2 = 128    # 2nd layer num features

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_inputs, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_inputs])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_inputs])),
}


# 使用的 Activation function 是 sigmoid， 压缩之后的值应该在 [0,1] 这个范围内。
def encoder(x):
    # 784 --> 256
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # 256 --> 128
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

def decoder(x):
    # 128 --> 256
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # 256 --> 784
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2

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

    encode_decode = sess.run(
        y_pred, feed_dict={X:mnist.test.images[:example_to_show]}
    )
    f, a = plt.subplots(2, 10, figsize=(10,2))
    for i in range(example_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28,28)))
    plt.show()
