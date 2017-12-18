'''
Session 是 Tensorflow 为了控制,和输出文件的执行的语句. 
运行 session.run() 可以获得你要得知的运算结果, 或者是你所要运算的部分.
'''

# Session()的两种打开方式

import tensorflow as tf 

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])

product = tf.matmul(matrix1, matrix2)  # 矩阵乘法  np.dot(m1,m2)

# method 1
# sess = tf.Session()
# results = sess.run(product)
# print(results)
# sess.close()

# 结果：[[12]]

# method 2 自动关闭session
with tf.Session() as sess:
    results2 = sess.run(product)
    print(results2)

# 结果：[[12]]
