'''
tf.histogram_summary()方法,用来绘制图片
'''
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

# tensorboard可视化  Graph(神经网络结构), histograms(训练过程), events(显示更多东西)

# 定义隐藏层，传入输入值，输入单位量，输出单位量，激励函数
# n_layer用来标识层数, 并且用变量 layer_name 代表其每层的名称
def add_layer(inputs, in_size, out_size, n_layer, activation_func = None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            # tf.histogram_summary()方法,用来绘制图片, 
            # 第一个参数是图表的名称, 第二个参数是图表要记录的变量
            tf.summary.histogram(layer_name + '/weights', Weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

        if activation_func is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_func(Wx_plus_b)

        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

# 造一些数据， 构建神经网络，要学习的是 y = x^2 - 0.5
x_data = np.linspace(-1, 1, 300)[:, np.newaxis] # x数据集，介于(-1,1)，有300个，[]增加维度
noise = np.random.normal(0, 0.05, x_data.shape) # 增加噪声
y_data = np.square(x_data) - 0.5 + noise # y数据集，需要学习的公式

# 定义placeholder
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 加隐藏层
l1 = add_layer(xs, 1, 10, n_layer=1, activation_func = tf.nn.relu) 
# 加输出层
prediction = add_layer(l1, 10, 1, n_layer=2, activation_func = None)

# 预测值和真实值之间的误差
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# 给所有训练图合并
sess = tf.Session()
merged = tf.summary.merge_all()
# SummaryWriter改为tf.summary.FileWriter
writer = tf.summary.FileWriter("log", sess.graph)
sess.run(tf.global_variables_initializer())


# 训练数据
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i%50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
