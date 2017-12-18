'''
使用RNN进行分类训练。
使用手写数字MNIST数据集，让RNN从每张图片第一行像素读到最后一行，然后进行判断。
'''

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 
tf.set_random_seed(1)

# import data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 设置参数
lr = 0.001 # learning rare
training_iters = 100000 # train step 上限
batch_size = 128

n_inputs = 28 # MNIST data input(img shape: 28*28) / rnn input size / image width
n_steps = 28  # rnn time steps / image height
n_hidden_units = 128 # hidden layer 神经元数
n_classes = 10  # MNIST classes(0-9 digits)

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    # shape(28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape(128, 10)
    'out':tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape(128,)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
    # shape(10,)
    'out':tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}

# 定义RNN主体结构
def RNN(X, weights, biases):
    '''input'''
    # 原始X为三维数据，需要转换为二维数据才能使用weights和矩阵乘法
    # X --> (128 batches * 28 steps, 28 inputs)
    # -1代表128*28
    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']

    # X_in --> (128 batches, 28 steps, 128 hidden)换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    '''cell'''
    # 使用basicLSTM Cell, 初始不希望foget, 设置forget_bias=1, 设置初始state为tuple
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    
    # lstm cell is divided into two paorts : c_state主线state，m_state分线state
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # 初始化全为零

    # 计算outputs和final_state, dynamic_rnn的rnn网络更好，time_step不在第一个维度，设为false
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    '''outputs'''
    # 方式一: 直接调用final_state 中的 h_state (final_state[1]) 来进行运算
    #results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # 方式二: 调用最后一个 outputs
    outputs = tf.stack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


# 计算cost和train_op
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# 训练RNN,不断输出accuracy
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x:batch_xs,
            y:batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x:batch_xs,
                y:batch_ys,
            }))
        step += 1