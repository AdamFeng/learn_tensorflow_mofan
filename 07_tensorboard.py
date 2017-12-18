'''
关键语句：with tf.name_scope()
'''
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

# tensorboard可视化神经网络结构  Graph

# 定义隐藏层，传入输入值，输入单位量，输出单位量，激励函数
def add_layer(inputs, in_size, out_size, activation_func = None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

        # activation_function 的话，可以暂时忽略。
        # 因为当你自己选择用 tensorflow 中的激励函数的时候，
        # tensorflow会默认添加名称。
        if activation_func is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_func(Wx_plus_b)
        return outputs

# 构建神经网络，要学习的是 y = x^2 - 0.5
x_data = np.linspace(-1, 1, 300)[:, np.newaxis] # x数据集，介于(-1,1)，有300个，[]增加维度
noise = np.random.normal(0, 0.05, x_data.shape) # 增加噪声
y_data = np.square(x_data) - 0.5 + noise # y数据集，需要学习的公式

# 使用with tf.name_scope('input')可以将xs和ys包含进来，形成一个大的图层，
# 图层的名字就是with tf.name_scope()方法里的参数。
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 加隐藏层
l1 = add_layer(xs, 1, 10, activation_func = tf.nn.relu) 
# 加输出层
prediction = add_layer(l1, 10, 1, activation_func = None)

# 预测值和真实值之间的误差
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
# SummaryWriter改为tf.summary.FileWriter
writer = tf.summary.FileWriter("log", sess.graph)
sess.run(init)
