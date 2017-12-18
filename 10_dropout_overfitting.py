'''
https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-02-A-overfitting/

过拟合的解决方法：
    方法一: 增加数据量, 大部分过拟合产生的原因是因为数据量太少了.
    方法二: 运用正规化. L1, l2 regularization等等,
    方法三: 专门用在神经网络的正规化的方法, 叫作 dropout.
'''

# 使用dropout避免过拟合
import tensorflow as tf 
from sklearn.datasets import load_digits 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    # 添加层并返回此层的输出
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

# load data
digits = load_digits()
X = digits.data # 加载从0-9的数字的图片
y = digits.target
y = LabelBinarizer().fit_transform(y) # 10个长度表示图片为数字几
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


# define placeholder for inputs to network
# dropout必须设置保留概率keep_prob,即我们要保留的结果所占比例。
# keep_prob也是一个placeholder，在run时传入。
# keep_prob=1相当于100%保留
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64]) #8x8
ys = tf.placeholder(tf.float32, [None, 10])

# add hidden layer 输入64，输出100，显示overfitting
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
# add output layer 输入100，输出10
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

# loss函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，交叉熵就等于零。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])) # loss
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

# summary writer goes in here
# train data summary
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
# test data summary
test_writer = tf.summary.FileWriter('logs/test', sess.graph)

sess.run(tf.global_variables_initializer())


# 训练
for i in range(500):
    sess.run(train_step, feed_dict={xs:X_train, ys:y_train, keep_prob:0.5}) # 保持50%不被drop
    if i%50 == 0:
        train_result = sess.run(merged, feed_dict={xs:X_train, ys:y_train, keep_prob:1})
        test_result = sess.run(merged, feed_dict={xs:X_test, ys:y_test, keep_prob:1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)

