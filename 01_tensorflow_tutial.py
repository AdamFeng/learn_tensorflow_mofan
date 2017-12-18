'''
安装tensorflow：
    pip install tensorflow
    pip install tensorflow-gpu
Tensorflow 首先要定义神经网络的结构, 然后再把数据放入结构当中去运算和 training。
TensorFlow是采用数据流图（data　flow　graphs）来计算。
将数据（数据以张量(tensor)的形式存在）放在数据流图中计算。
节点（Nodes）在图中表示数学操作,图中的线（edges）则表示在节点间相互联系的多维数据数组, 即张量（tensor).
训练模型时tensor会不断的从数据流图中的一个节点flow到另一节点, 这就是TensorFlow名字的由来.

张量（Tensor):
    张量有多种. 零阶张量为 纯量或标量 (scalar) 也就是一个数值. 比如 [1]
    一阶张量为 向量 (vector), 比如 一维的 [1, 2, 3]
    二阶张量为 矩阵 (matrix), 比如 二维的 [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
    以此类推, 还有 三阶 三维的 
'''

import tensorflow as tf 
import numpy as np 

# 创建数据
x_data = np.random.rand(100).astype(np.float32) # tensorflow中数据位float32格式的
y_data = x_data*0.1 + 0.3 # 需要学习的公式 Weights=0.1  biases=0.3

# 搭建模型
# 用 tf.Variable 来创建描述 y 的参数
'''
# 把 y_data = x_data*0.1 + 0.3 想象成 y=Weights * x + biases, 
# 然后神经网络也就是学着把 Weights 变成 0.1, biases 变成 0.3.
'''
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # Weights为一维结构，初始值随机，在(-1,1)之间
biases = tf.Variable(tf.zeros([1])) # biases初始值为0
# tensorflow通过学习，通过修改上述两个参数，逼近需要学习的公式

y = Weights * x_data + biases # 预测的y值

# 计算误差
loss = tf.reduce_mean(tf.square(y-y_data)) # 计算预测的y与实际y之间的差值，即误差

# 传播误差，反向传递误差工作交给optimizer

# 建立优化器，使用GradientDescentOptimizer优化器，设置学习效率为0.5
# 误差传递方法是梯度下降法：Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(0.5) 

# 使用 optimizer 来进行参数的更新。
train = optimizer.minimize(loss) 


# 通过训练，减少误差
#init = tf.initialize_all_variables() # v1.2初始化
init = tf.global_variables_initializer() # v1.4初始化

##### create tensorflow structure end ###

sess = tf.Session() # 设置tensorflow的session
sess.run(init)  # 激活神经网络

for step in range(201):
    sess.run(train) # 进行训练
    if step % 20 == 0:
        # 每隔20步，打印Weights和biases
        print(step, sess.run(Weights), sess.run(biases))