'''
不断增加图片的厚度，最终变成分类器
CNN包括输入层、隐藏层和输出层，隐藏层又包含卷积层和pooling层
(1)图像输入到卷积神经网络后通过卷积来不断的提取特征，每提取一个特征就会增加一个feature map，立方体增加厚度
(2)pooling层也就是下采样，通常采用的是最大值pooling和平均值pooling，所以通过pooling来稀疏参数，使我们的网络不至于太复杂。
   池化是一个筛选过滤的过程, 能将 layer 中有用的信息筛选出来, 给下一个层分析. 同时也减轻了神经网络的计算负担
流行的CNN结构：

    分类器classifier
    全连接神经层fully connected
    全连接神经层fully connected
    池化层pooling
    卷积层convolution    
    池化层pooling
    卷积层convolution
    图片image

'''

import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    return result

# 定义weight变量
def weight_variable(shape):
    # 用tf.truncted_normal产生随机变量来进行初始化
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 定义bias变量
def bias_variable(shape):
    # 用tf.constant常量函数来进行初始化
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义卷积神经网络层，x是图片的所有参数，W是此卷积层的权重，
# 定义步长strides=[1,1,1,1]，strides[0]和strides[3]是两个1的默认值，
# 中间两个1代表padding时在x方向运动一步，y方向运动一步，padding采用的方式是SAME
def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')


# 定义池化层pooling，为了得到更多图片信息，padding时我们选的是一次一步，也就是strides[1]=strides[2]=1，
# 这样的到的图片尺寸没有变化，我们希望压缩一下图片也就是参数少一点可以减小系统复杂度，
# 因此使用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层。
# pooling有两种方式，一种是最大值池化，一种是平均值池化
# 本例采用的是最大值池化tf.max_pool()。
# 池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]:
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

# 解决过拟合
keep_prob = tf.placeholder(tf.float32)

# 把xs的形状变成[-1,28,28,1]，-1代表先不考虑输入的图片例子多少这个维度，
# 后面的1是channel的数量，黑白图片channel是1，RGB图片channel就是3。
x_image = tf.reshape(xs, [-1,28,28,1])




''' 定义第一个卷积层+pooling'''

# 定义本层的weight，本层的卷积核patch大小是5x5，输出是32个featuremap
W_conv1 = weight_variable([5,5,1,32])

# 定义本层bias，大小为32个长度，因此我们出入它的shape为[32]
b_conv1 = bias_variable([32])

# 定义卷积层,对h_conv1使用tf.nn.relu进行非线性处理
# 因为采用了SAME的padding方式，输出图片的长宽没有变化依然是28x28，
# 只是厚度变厚了，因此现在的输出大小就变成了28x28x32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)

# 进行pooling，输出大小变为了14x14x32
h_pool1 = max_pool_2x2(h_conv1)


'''定义第二层卷积+pooling'''
# 本层输入为上一层输出，本层卷积核patch为5x5，
# 有32个featuremap，输入为32，定输出为64
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
# 输出的大小就是14x14x64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
# 输出大小为7x7x64
h_pool2 = max_pool_2x2(h_conv2)


'''建立全连接层1'''

# 此时weight_variable的shape输入就是第二个卷积层展平了的输出大小: 7x7x64，
# 后面的输出size我们继续扩大，定为1024
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据, 
# -1表示先不考虑输入图片例子维度, 将上一个输出结果展平.
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])

# 然后将展平后的h_pool2_flat与本层的W_fc1相乘（注意这个时候不是卷积了）
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

# dropout解决过拟合问题
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


'''建立全连接层2'''


# 输入是1024，最后的输出是10
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

# 用softmax分类器（多分类，输出是各个类的概率）,对我们的输出进行分类
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 利用交叉熵损失函数来定义我们的cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))

# 用tf.train.AdamOptimizer()作为我们的优化器进行优化，使我们的cross_entropy最小
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.5})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))