import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LR = 0.1
REAL_PARAMS = [1.2, 2.5]    # 我们假设的需要被学习的真实参数
INIT_PARAMS = [  # 在调参中不同初始化的参数点[0][1][2]
    [5, 4],
    [5, 1],
    [2, 4.5]
][1]
# x 数据
x = np.linspace(-1, 1, 200, dtype=np.float32)

'''
# test 1
# 拟合公式中的a,b参数
y_fun = lambda a, b: a * x + b
tf_y_fun = lambda a, b: a * x + b
'''

'''
# test 2
# 作为优化工具给经验公式调参
y_fun = lambda a, b: a * x**3 + b * x**2
tf_y_fun = lambda a, b: a * x**3 + b * x**2
'''


# test 3 初始参数 造成 局部最优，全局最优
# y_fun是由两层神经层组成的网络，cos and sin
y_fun = lambda a,b: np.sin(b*np.cos(a*x))
tf_y_fun = lambda a,b: tf.sin(b*tf.cos(a*x))


noise = np.random.randn(200)/10
y = y_fun(*REAL_PARAMS) + noise

# 定义 Tensorflow 的计算优化图纸.
# 定义两个参数 a, b 并给他们初始化成我们最开始假设的初始化值 INIT_PARAMS.
# 然后预测, 然后算误差, 最后优化.
a, b = [tf.Variable(initial_value=p, dtype=tf.float32) for p in INIT_PARAMS]
pred = tf_y_fun(a,b)
mse = tf.reduce_mean(tf.square(y-pred))
train_op = tf.train.GradientDescentOptimizer(LR).minimize(mse)

a_list, b_list, cost_list = [], [], []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(400):
        a_, b_, mes_ = sess.run([a, b, mse])
        a_list.append(a_)
        b_list.append(b_)
        cost_list.append(mes_)
        result, _ = sess.run([pred, train_op])

# 可视化
print('a=', a_, 'b=', b_)
plt.figure(1)
plt.scatter(x, y, c='b')
plt.plot(x, result, 'r-', lw=2)

# 3D cost figure
fig = plt.figure(2)
ax = Axes3D(fig)
a3D, b3D = np.meshgrid(np.linspace(-2, 7, 30), np.linspace(-2, 7, 30))
cost3D = np.array([np.mean(np.square(y_fun(a_, b_) - y)) for a_, b_ in zip(a3D.flatten(), b3D.flatten())]).reshape(a3D.shape)
ax.plot_surface(a3D, b3D, cost3D, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha=0.5)
ax.scatter(a_list[0], b_list[0], zs=cost_list[0], s=300, c='r')
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.plot(a_list, b_list, zs=cost_list, zdir='z', c='r', lw=3)
plt.show()