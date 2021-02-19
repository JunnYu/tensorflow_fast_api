from tf_fast_api import *
tf.random.set_seed(2021)
x = tf.random.normal((3, 5))
print(x.sum())
print(x.reduce_sum())
print(tf.reduce_sum(x))

import timeit
conv_layer = tf.keras.layers.Conv2D(100, 3)


@tf.function
def conv_fn(image):
    return conv_layer(image)


image = tf.zeros([1, 200, 200, 100])
# warm up
conv_layer(image)
conv_fn(image)
print("Eager conv:", timeit.timeit(lambda: conv_layer(image), number=10))
print("Function conv:", timeit.timeit(lambda: conv_fn(image), number=10))
print("Note how there's not much difference in performance for convolutions")

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)