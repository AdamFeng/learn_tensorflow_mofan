'''
Batch normalization 对每层都进行一次 normalization
Batch normalization 是一种解决深度神经网络层数太多, 而没办法有效前向传递(forward propagate)的问题.
因为每一层的输出值都会有不同的 均值(mean) 和 方差(deviation), 所以输出数据的分布也不一样
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ACTIVATION = tf.nn.tanh # activation function
N_LAYERS = 7    # 7 hidden layers
N_HIDDEN_UNITS = 30 # every hidden layer has 30 cells

# 复写
def fix_seed(seed=1):
    np.random.seed(seed)
    tf.set_random_seed(seed)

# 对每层的inputs画柱状图
def plot_his(inputs, inputs_norm):
    for j, all_inputs in enumerate([inputs, inputs_norm]):
        for i, input in enumerate(all_inputs):
            plt.subplot(2, len(all_inputs), j*len(all_inputs)+(i+1))
            plt.cla()
            if i==0:
                the_range = (-7,10)
            else:
                the_range = (-1,1)
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            plt.yticks(())
            if j==1:
                plt.xticks(the_range)
            else:
                plt.xticks(())
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title('%s normalizing' % ('without' if j == 0 else 'With'))
    plt.draw()
    plt.pause(0.01)


# 搭建网络
def built_net(xs, ys, norm):
    def add_layer(inputs, in_size, out_size, activation_function=None,norm=False ):
        Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0, stddev=1))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        # 全连接层结果
        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        # normalize 所有层
        if norm:
            # Batch normalize,均值，方差
            fc_mean, fc_var = tf.nn.moments(
                Wx_plus_b,
                axes=[0],   #normalize的方向，这里[0]代表batch
                                # 如果是图像，[0,1,2]代表[batch, height, width]
            )
            scale = tf.Variable(tf.ones([out_size]))    # 扩大参数
            shift = tf.Variable(tf.zeros([out_size]))   # 平移参数
            epsilon = 0.001

            # apply moving average for mean and var when train on batch
            '''
            如果你是使用 batch 进行每次的更新, 那每个 batch 的 mean/var 都会不同, 
            所以我们可以使用 moving average 的方法记录并慢慢改进 mean/var 的值.
            然后将修改提升后的 mean/var 放入 tf.nn.batch_normalization(). 
            而且在 test 阶段, 我们就可以直接调用最后一次修改的 mean/var 值进行测试, 
            而不是采用 test 时的 fc_mean/fc_var.
            '''
            ema = tf.train.ExponentialMovingAverage(decay=0.5)   # exponential moving average 的 decay 度
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([fc_mean, fc_var]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()  # 根据新的 batch 数据, 记录并稍微修改之前的 mean/var
            # mean, var = tf.cond(
            #     on_train,    # on_train 的值是 True/False
            #     mean_var_with_update,   # 如果是 True, 更新 mean/var
            #     lambda:(    # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
            #         ema.average(fc_mean),
            #         ema.average(fc_var)
            #     )
            # )

            # 将修改后的 mean / var 放入下面的公式
            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
            # similar with this two steps:
            # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
            # Wx_plus_b = Wx_plus_b * scale + shift

        # activation
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    fix_seed(1)

    if norm:
        # BN for the first input
        fc_mean, fc_var = tf.nn.moments(
            xs,
            axes=[0],
        )
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001
        # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)
        mean, var = mean_var_with_update()
        xs = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)

    layers_inputs = [xs]    # 记录每层的input

    # 循环建立所有层
    for l_n in range(N_LAYERS):
        layer_input = layers_inputs[l_n]
        in_size = layers_inputs[l_n].get_shape()[1].value

        output = add_layer(
            layer_input,
            in_size,
            N_HIDDEN_UNITS,
            ACTIVATION,
            norm,
        )
        layers_inputs.append(output) # 把output放入记录

    # 建立output layer
    prediction = add_layer(layers_inputs[-1], 30, 1, activation_function=None)

    cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    return [train_op, cost, layers_inputs]

# 造数据
fix_seed(1)
x_data = np.linspace(-7,10,2500)[:, np.newaxis]
np.random.shuffle(x_data)
noise = np.random.normal(0,8,x_data.shape)
y_data = np.square(x_data)-5 + noise

# plot input data
plt.scatter(x_data, y_data)
plt.show()

xs = tf.placeholder(tf.float32, [None,1])
ys = tf.placeholder(tf.float32, [None,1])

# 搭建两个神经网络，一个有BN，一个没有BN
train_op, cost, layers_inputs = built_net(xs, ys, norm=False)
train_op_norm, cost_norm, layers_inputs_norm = built_net(xs, ys, norm=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 记录两种网络cost的变化
cost_his = []
cost_his_norm = []
record_step = 5

plt.ion()
plt.figure(figsize=(7,3))
for i in range(250):
    if i % 50 == 0:
        # plot histogram
        all_inputs, all_inputs_norm = sess.run([layers_inputs, layers_inputs_norm], feed_dict={xs: x_data, ys: y_data})
        plot_his(all_inputs, all_inputs_norm)

    # train on batch
    sess.run([train_op, train_op_norm], feed_dict={xs: x_data[i*10:i*10+10], ys: y_data[i*10:i*10+10]})

    if i % record_step == 0:
        # record cost
        cost_his.append(sess.run(cost, feed_dict={xs: x_data, ys: y_data}))
        cost_his_norm.append(sess.run(cost_norm, feed_dict={xs: x_data, ys: y_data}))

plt.ioff()
plt.figure()
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his), label='no BN')     # no norm
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his_norm), label='BN')   # norm
plt.legend()
plt.show()



