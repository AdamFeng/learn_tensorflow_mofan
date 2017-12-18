import tensorflow as tf 

# tensorflow activation func
# http://wiki.jikexueyuan.com/project/tensorflow-zh/api_docs/python/nn.html#sigmoid

'''
激励函数运行时激活神经网络中某一部分神经元，将激活信息向后传入下一层的神经系统。
激励函数也就是为了解决不能用线性方程所概括的问题。
把整个网络简化成这样一个式子：Y = Wx, W 就是我们要求的参数, y 是预测值, x 是输入值。

Y = AF (Wx)
AF 就是指的激励函数， 它其实就是另外一个非线性函数。

在卷积神经网络 Convolutional neural networks 的卷积层中, 推荐的激励函数是 relu. 
在循环神经网络中 recurrent neural networks, 推荐的是 tanh 或者是 relu
'''
