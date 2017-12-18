'''
在 Tensorflow 中，定义了某字符串是变量，它才是变量。
'''
import tensorflow as tf 

# 定义变量state
state = tf.Variable(0, name = 'counter')
#print(state.name)

# 定义常量one
one = tf.constant(1) # 加常量1

# 定义加法步骤(注：此步没有直接计算)
new_value = tf.add(state , one) # new_value = state + 1

# 将state更新成new_value
update = tf.assign(state, new_value) # state = new_value

#init = tf.initialize_all_variables() # v1.2初始化

# 定义完所有变量后记得init
init = tf.global_variables_initializer() # v1.4初始化

with tf.Session() as sess:
    # 激活init
    sess.run(init) # 设置Session后首先run一下init
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
        # 直接 print(state) 不起作用！！