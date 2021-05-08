import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import numpy as np
from fastcore.utils import patch_to

ttf = tf.constant(0.0)


@patch_to(cls=[tf.Tensor, tf.Variable])
def embedding_lookup(self, ids, max_norm=None, name=None):
    # 从第一个维度self.shape[0]查找
    return tf.nn.embedding_lookup(self, ids=ids, max_norm=max_norm, name=name)


def transform_size(size):

    return size


@patch_to(cls=[tf.Tensor, tf.Variable])
def bool(self):
    return self.astype("bool")


@patch_to(cls=[tf.Tensor, tf.Variable])
def float(self):
    return self.astype("float32")


@patch_to(cls=[tf.Tensor, tf.Variable])
def float32(self):
    return self.astype("float32")


@patch_to(cls=[tf.Tensor, tf.Variable])
def half(self):
    return self.astype("float16")


@patch_to(cls=[tf.Tensor, tf.Variable])
def float16(self):
    return self.astype("float16")


@patch_to(cls=[tf.Tensor, tf.Variable])
def double(self):
    return self.astype("float64")


@patch_to(cls=[tf.Tensor, tf.Variable])
def float64(self):
    return self.astype("float64")


@patch_to(cls=[tf.Tensor, tf.Variable])
def int(self):
    return self.astype("int32")


@patch_to(cls=[tf.Tensor, tf.Variable])
def int32(self):
    return self.astype("int32")


@patch_to(cls=[tf.Tensor, tf.Variable])
def int8(self):
    return self.astype("int8")


@patch_to(cls=[tf.Tensor, tf.Variable])
def int16(self):
    return self.astype("int16")


@patch_to(cls=[tf.Tensor, tf.Variable])
def short(self):
    return self.astype("int16")


@patch_to(cls=[tf.Tensor, tf.Variable])
def int64(self):
    return self.astype("int64")


@patch_to(cls=[tf.Tensor, tf.Variable])
def long(self):
    return self.astype("int64")


@patch_to(cls=[tf.Tensor, tf.Variable])
def add(self, y, name=None):
    return tf.add(self, y, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def subtract(self, y, name=None):
    return tf.subtract(self, y, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def sub(self, y, name=None):
    return self.subtract(y, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def multiply(self, y, name=None):
    return tf.multiply(self, y, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def mul(self, y, name=None):
    return self.multiply(y, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def divide(self, y, name=None):
    return tf.divide(self, y, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def div(self, y, name=None):
    return self.divide(y, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def realdiv(self, y, name=None):
    return tf.realdiv(self, y, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def abs(self, name=None):
    return tf.abs(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def pow(self, y=None, name=None):
    if y is None:
        y = 2
    return tf.pow(self, y, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def argsort(self, axis=-1, direction='ASCENDING', stable=False, name=None):
    return tf.argsort(self,
                      axis=axis,
                      direction=direction,
                      stable=stable,
                      name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def batch_to_space(self, block_shape, crops, name=None):
    return tf.batch_to_space(self, block_shape, crops, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def boolean_mask(self, mask, axis=None, name="boolean_mask"):
    return tf.boolean_mask(self, mask, axis=axis, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def broadcast_to(self, *size, shape=None, name=None):
    if shape is None and len(size) != 0:
        shape = size
    return tf.broadcast_to(self, shape, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def clip_by_norm(self, clip_norm, axes=None, name=None):
    return tf.clip_by_norm(self, clip_norm, axes=axes, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def clip_by_value(self, clip_value_min, clip_value_max, name=None):
    return tf.clip_by_value(self, clip_value_min, clip_value_max, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def clip(self, clip_value_min, clip_value_max, name=None):
    return self.clip_by_value(clip_value_min, clip_value_max, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def cond(self, true_fn=None, false_fn=None, name=None):
    return tf.cond(self, true_fn=true_fn, false_fn=false_fn, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def expand_dims(self, axis, name=None):
    return tf.expand_dims(self, axis, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def unsqueeze(self, axis, name=None):
    return self.expand_dims(axis, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def sin(self, name=None):
    return tf.math.sin(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def cos(self, name=None):
    return tf.math.cos(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def flatten(self, start=0, end=-1, name=None):
    size = self.size()
    mmax = len(size)
    if start < 0:
        start += mmax
    if end < 0:
        end += mmax
    assert start != end, "start must less end"
    left = []
    mid = []
    right = []
    if 0 <= start < mmax:
        left.extend(size[:start])
    if 0 < end <= mmax:
        right.extend(size[end + 1:])
    if start != end:
        if end == mmax:
            mid.extend(np.cumprod(size[start:])[-1:].tolist())
        else:
            mid.extend(np.cumprod(size[start:end + 1])[-1:].tolist())

    new_size = left + mid + right

    return tf.reshape(self, new_size, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def size(self, axis=None):
    static = self.shape.as_list()
    dynamic = tf.shape(self)
    shape_list = [dynamic[i] if s is None else s for i, s in enumerate(static)]
    if axis is None:
        return shape_list
    else:
        return shape_list[axis]


@patch_to(cls=[tf.Tensor, tf.Variable])
def identity(self, name=None):
    return tf.identity(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def clone(self, name=None):
    return self.identity(name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def norm(self, ord='euclidean', axis=None, keepdims=None, name=None):
    return tf.norm(self, ord=ord, axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def ones_like(self, dtype=None, name=None):
    if dtype is None:
        dtype = self.dtype
    return tf.ones_like(self, dtype=dtype, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def zeros_like(self, dtype=None, name=None):
    if dtype is None:
        dtype = self.dtype
    return tf.zeros_like(self, dtype=dtype, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def full_like(self, value, name=None):
    value = tf.cast(value, self.dtype)
    return tf.fill(self.shape, value, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def ones(self, *size, shape=None, dtype=None, name=None):
    if dtype is None:
        dtype = self.dtype
    if shape is None and len(size) != 0:
        shape = size
    return tf.ones(shape, dtype=dtype, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def zeros(self, *size, shape=None, dtype=None, name=None):
    if dtype is None:
        dtype = self.dtype
    if shape is None and len(size) != 0:
        shape = size
    return tf.zeros(shape, dtype=dtype, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def rank(self, name=None):
    return tf.rank(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def cast(self, dtype, name=None):
    return tf.cast(self, dtype, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def astype(self, dtype, name=None):
    return self.cast(dtype, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def reshape(self, *size, shape=None, name=None):
    if shape is None and len(size) != 0:
        shape = size
    return tf.reshape(self, shape, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def view(self, *size, shape=None, name=None):
    return self.reshape(size, shape=shape, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def reverse(self, axis, name=None):
    return tf.reverse(self, axis, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def flip(self, axis, name=None):
    return self.reverse(axis, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def numel(self, out_type=tf.int32, name=None):
    return tf.size(self, out_type, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def slice(self, begin, size, name=None):
    return tf.slice(self, begin, size, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def sort(self, axis=-1, direction='ASCENDING', name=None):
    return tf.sort(self, axis=axis, direction=direction, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def squeeze(self, *size, axis=None, name=None):
    if axis is None and len(size) != 0:
        axis = transform_size(size)
    return tf.squeeze(self, axis=axis, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def strided_slice(self,
                  begin,
                  end,
                  strides=None,
                  begin_mask=0,
                  end_mask=0,
                  ellipsis_mask=0,
                  new_axis_mask=0,
                  shrink_axis_mask=0,
                  var=None,
                  name=None):
    return tf.strided_slice(self,
                            begin,
                            end,
                            strides=strides,
                            begin_mask=begin_mask,
                            end_mask=end_mask,
                            ellipsis_mask=ellipsis_mask,
                            new_axis_mask=new_axis_mask,
                            shrink_axis_mask=shrink_axis_mask,
                            var=var,
                            name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def reduce_all(self, axis=None, keepdims=False, name=None):
    if self.dtype != tf.bool:
        self = self.astype("bool")
    return tf.reduce_all(self, axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def all(self, axis=None, keepdims=False, name=None):
    return self.reduce_all(axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def reduce_any(self, axis=None, keepdims=False, name=None):
    if self.dtype != tf.bool:
        self = self.astype("bool")
    return tf.reduce_any(self, axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def any(self, axis=None, keepdims=False, name=None):
    return self.reduce_any(axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def equal(self, b, name=None):
    return tf.equal(self, b, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def tensordot(self, b, axes, name=None):
    return tf.tensordot(self, b, axes, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def tensor_scatter_nd_add(self, indices, updates, name=None):
    return tf.tensor_scatter_nd_add(self, indices, updates, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def tensor_scatter_nd_sub(self, indices, updates, name=None):
    return tf.tensor_scatter_nd_sub(self, indices, updates, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def tensor_scatter_nd_update(self, indices, updates, name=None):
    return tf.tensor_scatter_nd_update(self, indices, updates, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def tile(self, *size, multiples=None, name=None):
    if multiples is None and len(size) != 0:
        multiples = transform_size(size)
    return tf.tile(self, multiples, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def repeat(self, *size, multiples=None, name=None):
    return self.tile(*size, multiples=multiples, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def transpose(self, *size, perm=None, conjugate=False, name='transpose'):
    if perm is None and len(size) != 0:
        perm = transform_size(size)
    return tf.transpose(self, perm=perm, conjugate=conjugate, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def t(self):
    assert self.ndim <= 2, f"expects a tensor with <= 2 dimensions,but get ndim {self.ndim}!"
    return self.transpose()


@patch_to(cls=[tf.Tensor, tf.Variable], as_prop=True)
def T(self):
    return self.transpose()


@patch_to(cls=[tf.Tensor, tf.Variable])
def permute(self, *size, perm=None, conjugate=False, name='transpose'):
    return self.transpose(*size, perm=perm, conjugate=conjugate, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def truncatediv(self, y, name=None):
    return tf.truncatediv(self, y, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def truncatemod(self, y, name=None):
    return tf.truncatemod(self, y, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def unique(self, out_idx=tf.int32, name=None):
    return tf.unique(self, out_idx, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def unique_with_counts(self, out_idx=tf.int32, name=None):
    return tf.unique_with_counts(self, out_idx, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def where(self, x=None, y=None, name=None):
    return tf.where(self, x=x, y=y, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def relu(self, name=None):
    return tf.nn.relu(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def selu(self, name=None):
    return tf.nn.selu(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def relu6(self, name=None):
    return tf.nn.relu6(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def tanh(self, name=None):
    return tf.nn.tanh(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def sigmoid(self, name=None):
    return tf.nn.sigmoid(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def leaky_relu(self, alpha=0.2, name=None):
    return tf.nn.leaky_relu(self, alpha=alpha, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def elu(self, name=None):
    return tf.nn.elu(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def softsign(self, name=None):
    return tf.nn.softsign(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def sign(self, name=None):
    return tf.sign(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def sqrt(self, name=None):
    return tf.sqrt(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def l2_loss(self, name=None):
    return tf.nn.l2_loss(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def log_softmax(self, axis=None, name=None):
    return tf.nn.log_softmax(self, axis=axis, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def softmax(self, axis=None, name=None):
    return tf.nn.softmax(self, axis=axis, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def square(self, name=None):
    return tf.square(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def arctanh(self, name=None):
    return tf.arctanh(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def reduce_sum(self, axis=None, keepdims=False, name=None):
    if self.dtype == tf.bool:
        self = self.astype(tf.int64)
    return tf.reduce_sum(self, axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def reduce_logsumexp(self, axis=None, keepdims=False, name=None):
    if self.dtype == tf.bool:
        self = self.astype(tf.int64)
    return tf.reduce_logsumexp(self, axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def logsumexp(self, axis=None, keepdims=False, name=None):
    return self.reduce_logsumexp(axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def sum(self, axis=None, keepdims=False, name=None):
    return self.reduce_sum(axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def reduce_prod(self, axis=None, keepdims=False, name=None):
    if self.dtype == tf.bool:
        self = self.astype(tf.int64)
    return tf.reduce_prod(self, axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def prod(self, axis=None, keepdims=False, name=None):
    return self.reduce_prod(axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def reduce_mean(self, axis=None, keepdims=False, name=None):
    if self.dtype == tf.bool:
        self = self.astype(tf.int64)
    return tf.reduce_mean(self, axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def mean(self, axis=None, keepdims=False, name=None):
    return self.reduce_mean(axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def reduce_min(self, axis=None, keepdims=False, name=None):
    return tf.reduce_min(self, axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def min(self, axis=None, keepdims=False, name=None):
    return self.reduce_min(axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def reduce_max(self, axis=None, keepdims=False, name=None):
    return tf.reduce_max(self, axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def max(self, axis=None, keepdims=False, name=None):
    return self.reduce_max(axis=axis, keepdims=keepdims, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def argmin(self, axis=None, output_type=tf.int64, name=None, keepdims=False):
    out = tf.argmin(self, axis=axis, output_type=output_type, name=name)
    if keepdims:
        out = out.unsqueeze(axis=axis)
    return out


@patch_to(cls=[tf.Tensor, tf.Variable])
def argmax(self, axis=None, output_type=tf.int64, name=None, keepdims=False):
    out = tf.argmax(self, axis=axis, output_type=output_type, name=name)
    if keepdims:
        out = out.unsqueeze(axis=axis)
    return out


@patch_to(cls=[tf.Tensor, tf.Variable])
def minimum(self, y, name=None):
    return tf.minimum(self, y, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def maximum(self, y, name=None):
    return tf.maximum(self, y, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def sort(self, axis=-1, direction='ASCENDING', name=None):
    return tf.sort(self, axis=axis, direction=direction, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def top_k(self, k=1, sorted=True, name=None):
    return tf.math.top_k(self, k=k, sorted=sorted, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def topk(self, k=1, sorted=True, name=None):
    return self.top_k(k=k, sorted=sorted, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def rand(self,
         *size,
         shape=None,
         minval=0,
         maxval=None,
         dtype=tf.float32,
         seed=None,
         name=None):
    if shape is None and len(size) != 0:
        shape = size
    return tf.random.uniform(shape=shape,
                             minval=minval,
                             maxval=maxval,
                             dtype=dtype,
                             seed=seed,
                             name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def uniform(self,
            *size,
            shape=None,
            minval=0,
            maxval=None,
            dtype=tf.float32,
            seed=None,
            name=None):
    return self.rand(*size,
                     shape=shape,
                     minval=minval,
                     maxval=maxval,
                     dtype=dtype,
                     seed=seed,
                     name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def randn(self,
          *size,
          shape=None,
          mean=0.0,
          stddev=1.0,
          dtype=tf.float32,
          seed=None,
          name=None):
    if shape is None and len(size) != 0:
        shape = size
    return tf.random.normal(shape=shape,
                            mean=mean,
                            stddev=stddev,
                            dtype=dtype,
                            seed=seed,
                            name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def normal(self,
           *size,
           shape=None,
           mean=0.0,
           stddev=1.0,
           dtype=tf.float32,
           seed=None,
           name=None):
    return self.randn(*size,
                      shape=shape,
                      mean=mean,
                      stddev=stddev,
                      dtype=dtype,
                      seed=seed,
                      name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def item(self):
    return self.numpy().item()


@patch_to(cls=[tf.Tensor, tf.Variable])
def exp(self, name=None):
    return tf.exp(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def log(self, name=None):
    return tf.math.log(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def log2(self, name=None):
    return self.log(name=name) / tf.math.log(2.0)


@patch_to(cls=[tf.Tensor, tf.Variable])
def log10(self, name=None):
    return self.log(name=name) / tf.math.log(10.0)


@patch_to(cls=[tf.Tensor, tf.Variable])
def log1p(self, name=None):
    return tf.math.log1p(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def isnan(self, name=None):
    return tf.math.is_nan(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def isinf(self, name=None):
    return tf.math.is_inf(self, name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def cumsum(self, axis=None, exclusive=False, reverse=False, name=None):
    if axis is None:
        self = self.reshape(-1)
        axis = 0
    return tf.cumsum(self,
                     axis=axis,
                     exclusive=exclusive,
                     reverse=reverse,
                     name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def gather(self,
           indices,
           validate_indices=None,
           axis=None,
           batch_dims=0,
           name=None):
    return tf.gather(self,
                     indices=indices,
                     validate_indices=validate_indices,
                     axis=axis,
                     batch_dims=batch_dims,
                     name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def matmul(self,
           b,
           transpose_a=False,
           transpose_b=False,
           adjoint_a=False,
           adjoint_b=False,
           a_is_sparse=False,
           b_is_sparse=False,
           name=None):
    return tf.matmul(self,
                     b,
                     transpose_a=transpose_a,
                     transpose_b=transpose_b,
                     adjoint_a=adjoint_a,
                     adjoint_b=adjoint_b,
                     a_is_sparse=a_is_sparse,
                     b_is_sparse=b_is_sparse,
                     name=name)


@patch_to(cls=[tf.Tensor, tf.Variable])
def getitem(self, index, axis=None):
    if isinstance(index, tuple):
        index = tuple(x if isinstance(x, tf.Tensor) else x for x in index)
        basic = all(x is None or x is Ellipsis or isinstance(x, int)
                    or isinstance(x, slice) for x in index)
        if not basic:
            # workaround for missing support for this in TensorFlow
            index = [tf.convert_to_tensor(x) for x in index]
            shapes = [tuple(x.shape) for x in index]
            shape = tuple(max(x) for x in zip(*shapes))
            int64 = any(x.dtype == tf.int64 for x in index)
            for i in range(len(index)):
                t = index[i]
                if int64:
                    t = tf.cast(t, tf.int64)  # pragma: no cover
                assert t.ndim == len(shape)
                tiling = []
                for b, k in zip(shape, t.shape):
                    if k == 1:
                        tiling.append(b)
                    elif k == b:
                        tiling.append(1)
                    else:
                        raise ValueError(  # pragma: no cover
                            f"{tuple(t.shape)} cannot be broadcasted to {shape}"
                        )
                index[i] = tf.tile(t, tiling)
            index = tf.stack(index, axis=-1)
            return tf.gather_nd(self, index, axis=axis)
    elif (isinstance(index, range) or isinstance(index, list)
          or isinstance(index, np.ndarray)):
        return tf.gather(self, index, axis=axis)
    elif isinstance(index, tf.Tensor):
        if index.dtype == tf.bool:
            return self.getitem(index)
        else:
            return tf.gather(self, index, axis=axis)
    return self.getitem(index)
