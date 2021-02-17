from tf_fast_api import *
tf.random.set_seed(2021)
x = tf.random.normal((3, 5))
print(x.sum())
print(x.reduce_sum())
print(tf.reduce_sum(x))
