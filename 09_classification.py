import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

# classification 分类器

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    # 预测y的值
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    # 预测的y值和真是值之间的差别
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    # 计算这组数据中多少是对的，多少时错的
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 返回百分比
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# 每张图片的分辨率是28×28，所以我们的训练网络输入应该是28×28=784个像素数据。
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28

# 每张图片都表示一个数字，所以我们的输出是数字0到9，共10类。
ys = tf.placeholder(tf.float32, [None, 10])

# 添加输出层 输入为xs, 输入数据784个特征， 输出数据10个特征，激励函数为softmax 
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# 预测值和真实值间的误差
# loss函数（即最优化目标函数）选用交叉熵函数。
# 交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss

# train方法（最优化算法）采用梯度下降法。
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# 进行训练
for i in range(1000):
    # 每次选取100张图片进行训练
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})

    # 每50次输出训练精度
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
