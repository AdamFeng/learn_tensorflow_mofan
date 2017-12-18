'''
定义添加层，构建神经网络并使用matplotlib进行可视化
'''
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

# 神经层里常见的参数通常有weights、biases和激励函数。
# 定义添加层，传入输入值，输入单位量，输出单位量，激励函数
def add_layer(inputs, in_size, out_size, activation_func = None):
    # weights为一个in_size行, out_size列的随机变量矩阵。
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))

    # 在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。
    biases = tf.Variable(tf.zeros([1, out_size])) + 0.1

    # 定义Wx_plus_b, 即神经网络未激活的值。
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # 当激励函数为none时，直接输出当前预测值Wx_plus_b；不为none时，将Wx_plus_b传入AF中得到输出
    if activation_func is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_func(Wx_plus_b)
    return outputs

# 构建神经网络，要学习的是 y = x^2 - 0.5

# x数据集，介于(-1,1)，有300个，[]增加维度
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  
# 增加噪声
noise = np.random.normal(0, 0.05, x_data.shape)
# y数据集，需要学习的公式
y_data = np.square(x_data) - 0.5 + noise 

# tf.placeholder()就是代表占位符，这里的None代表无论输入有多少都可以，
# 因为输入只有一个特征，所以这里是1。
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 定义隐藏层
l1 = add_layer(xs, 1, 10, activation_func = tf.nn.relu) 
# 定义输出层
prediction = add_layer(l1, 10, 1, activation_func = None)

# 预测值和真实值之间的误差，对二者差的平方求和再取平均。
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 
                        reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion() # plot后不暂停，连续显示
plt.show()

# 机器学习的内容是train_step, 
# 用 Session 来 run 每一次 training 的数据，
# 逐步提升神经网络的预测准确性。
for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i%50 == 0:
        # print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))

        # 结果可视化
        # 首次无lines[0]，先try一下
        try:
            ax.lines.remove(lines[0]) # 去除lines[0]
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs:x_data})

        # 用红线画出预测数据与输入之间的关系
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
