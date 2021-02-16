import tensorflow as tf
from utils import *

x = tf.random.normal((3, 5, 8, 9))
y = tf.random.normal((10, 5))

print(x.argmin(axis=2, keepdims=True))
