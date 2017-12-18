'''
我们搭建好了一个神经网络, 训练好了, 想保存起来, 用于再次加载.
只能保存提取神经网络中的变量，无法保存神经网络的结构。
需要提取变量后再定义结构进行训练。
'''

import tensorflow as tf 
import numpy as np 

'''
# 定义相同的dtype和shape才能正确的导出导入
W = tf.Variable([[1,2,3], [3,4,5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()

# 用于保存提取变量
saver = tf.train.Saver()

# 保存网络到文件
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, 'my_net/save_net.ckpt')
    print('Save to path:', save_path)
'''


# 定义相同的dtype和shape才能正确的导出导入
W = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name='biases')

# 用于保存提取变量
saver = tf.train.Saver()

# 提取变量
with tf.Session() as sess:
    saver.restore(sess, 'my_net/save_net.ckpt')
    print('weights:', sess.run(W))
    print('biases:', sess.run(b))

'''
weights: [[ 1.  2.  3.]
 [ 3.  4.  5.]]
biases: [[ 1.  2.  3.]]
'''