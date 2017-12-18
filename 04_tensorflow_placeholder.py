'''
placeholder 是 Tensorflow 中的占位符，暂时储存变量。
'''

import tensorflow as tf 

# Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder()
# placeholder()每次run从外界传入值
# 在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output 
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # 每次run需要传入值，传入的是dict
    # 需要传入的值放在feed_dict={}
    # placeholder 与 feed_dict={} 是绑定在一起出现的。
    print(sess.run(output, feed_dict={input1:[7.], input2:[3.]}))