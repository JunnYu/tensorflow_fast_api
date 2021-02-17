import tensorflow as tf
from fastcore.utils import patch


@patch
def bool(self: tf.Tensor):
    return self.astype("bool")


@patch
def float(self: tf.Tensor):
    return self.astype("float32")


@patch
def float32(self: tf.Tensor):
    return self.astype("float32")


@patch
def half(self: tf.Tensor):
    return self.astype("float16")


@patch
def float16(self: tf.Tensor):
    return self.astype("float16")


@patch
def double(self: tf.Tensor):
    return self.astype("float64")


@patch
def float64(self: tf.Tensor):
    return self.astype("float64")


@patch
def int(self: tf.Tensor):
    return self.astype("int32")


@patch
def int32(self: tf.Tensor):
    return self.astype("int32")


@patch
def int8(self: tf.Tensor):
    return self.astype("int8")


@patch
def int16(self: tf.Tensor):
    return self.astype("int16")


@patch
def short(self: tf.Tensor):
    return self.astype("int16")


@patch
def int64(self: tf.Tensor):
    return self.astype("int64")


@patch
def long(self: tf.Tensor):
    return self.astype("int64")


@patch
def add(self: tf.Tensor, y: tf.Tensor, name=None):
    return tf.add(self, y, name=name)


@patch
def subtract(self: tf.Tensor, y: tf.Tensor, name=None):
    return tf.subtract(self, y, name=name)


@patch
def sub(self: tf.Tensor, y: tf.Tensor, name=None):
    return self.subtract(y, name=name)


@patch
def multiply(self: tf.Tensor, y: tf.Tensor, name=None):
    return tf.multiply(self, y, name=name)


@patch
def mul(self: tf.Tensor, y: tf.Tensor, name=None):
    return self.multiply(y, name=name)


@patch
def divide(self: tf.Tensor, y: tf.Tensor, name=None):
    return tf.divide(self, y, name=name)


@patch
def div(self: tf.Tensor, y: tf.Tensor, name=None):
    return self.divide(y, name=name)


@patch
def realdiv(self: tf.Tensor, y: tf.Tensor, name=None):
    return tf.realdiv(self, y, name=name)


@patch
def argsort(self: tf.Tensor,
            axis=-1,
            direction='ASCENDING',
            stable=False,
            name=None):
    return tf.argsort(self,
                      axis=axis,
                      direction=direction,
                      stable=stable,
                      name=name)


@patch
def batch_to_space(self: tf.Tensor, block_shape, crops, name=None):
    return tf.batch_to_space(self, block_shape, crops, name=name)


@patch
def boolean_mask(self: tf.Tensor, mask, axis=None, name="boolean_mask"):
    return tf.boolean_mask(self, mask, axis=axis, name=name)


@patch
def broadcast_to(self: tf.Tensor, *shape, name=None):
    return tf.broadcast_to(self, shape, name=name)


@patch
def clip_by_norm(self: tf.Tensor, clip_norm, axes=None, name=None):
    return tf.clip_by_norm(self, clip_norm, axes=axes, name=name)


@patch
def clip_by_value(self: tf.Tensor, clip_value_min, clip_value_max, name=None):
    return tf.clip_by_value(self, clip_value_min, clip_value_max, name=name)


@patch
def clip(self: tf.Tensor, clip_value_min, clip_value_max, name=None):
    return self.clip_by_value(clip_value_min, clip_value_max, name=name)


@patch
def cond(self: tf.Tensor, true_fn=None, false_fn=None, name=None):
    return tf.cond(self, true_fn=true_fn, false_fn=false_fn, name=name)


@patch
def expand_dims(self: tf.Tensor, axis, name=None):
    return tf.expand_dims(self, axis, name=name)


@patch
def unsqueeze(self: tf.Tensor, axis, name=None):
    return self.expand_dims(axis, name=name)


@patch
def size(self: tf.Tensor, axis=None):
    if axis is not None:
        return self.shape[axis]
    return self.shape


@patch
def identity(self: tf.Tensor, name=None):
    return tf.identity(self, name=name)


@patch
def clone(self: tf.Tensor, name=None):
    return self.identity(name=name)


@patch
def norm(self: tf.Tensor,
         ord='euclidean',
         axis=None,
         keepdims=None,
         name=None):
    return tf.norm(self, ord=ord, axis=axis, keepdims=keepdims, name=name)


@patch
def ones_like(self: tf.Tensor, dtype=None, name=None):
    if dtype is None:
        dtype = self.dtype
    return tf.ones_like(self, dtype=dtype, name=name)


@patch
def zeros_like(self: tf.Tensor, dtype=None, name=None):
    if dtype is None:
        dtype = self.dtype
    return tf.zeros_like(self, dtype=dtype, name=name)


@patch
def full_like(self: tf.Tensor, value, name=None):
    value = tf.cast(value, self.dtype)
    return tf.fill(self.shape, value, name=name)


@patch
def ones(self: tf.Tensor, shape, dtype=None, name=None):
    if dtype is None:
        dtype = self.dtype
    return tf.ones(shape, dtype=dtype, name=name)


@patch
def zeros(self: tf.Tensor, shape, dtype=None, name=None):
    if dtype is None:
        dtype = self.dtype
    return tf.zeros(shape, dtype=dtype, name=name)


@patch
def rank(self: tf.Tensor, name=None):
    return tf.rank(self, name=name)


@patch
def cast(self: tf.Tensor, dtype, name=None):
    return tf.cast(self, dtype, name=name)


@patch
def astype(self: tf.Tensor, dtype, name=None):
    return self.cast(dtype, name=name)


@patch
def reshape(self: tf.Tensor, shape, name=None):
    return tf.reshape(self, shape, name=name)


@patch
def view(self: tf.Tensor, shape, name=None):
    return tf.reshape(self, shape, name=name)


@patch
def reverse(self: tf.Tensor, axis, name=None):
    return tf.reverse(self, axis, name=name)


@patch
def flip(self: tf.Tensor, axis, name=None):
    return self.reverse(axis, name=name)


@patch
def numel(self: tf.Tensor, out_type=tf.int32, name=None):
    return tf.size(self, out_type, name=name)


@patch
def slice(self: tf.Tensor, begin, size, name=None):
    return tf.slice(self, begin, size, name=name)


@patch
def sort(self: tf.Tensor, axis=-1, direction='ASCENDING', name=None):
    return tf.sort(self, axis=axis, direction=direction, name=name)


@patch
def squeeze(self: tf.Tensor, axis, name=None):
    return tf.squeeze(self, axis=axis, name=name)


@patch
def strided_slice(self: tf.Tensor,
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


@patch
def reduce_all(self: tf.Tensor, axis=None, keepdims=False, name=None):
    if self.dtype != tf.bool:
        self = self.astype("bool")
    return tf.reduce_all(self, axis=axis, keepdims=keepdims, name=name)


@patch
def all(self: tf.Tensor, axis=None, keepdims=False, name=None):
    return self.reduce_all(axis=axis, keepdims=keepdims, name=name)


@patch
def reduce_any(self: tf.Tensor, axis=None, keepdims=False, name=None):
    if self.dtype != tf.bool:
        self = self.astype("bool")
    return tf.reduce_any(self, axis=axis, keepdims=keepdims, name=name)


@patch
def any(self: tf.Tensor, axis=None, keepdims=False, name=None):
    return self.reduce_any(axis=axis, keepdims=keepdims, name=name)


@patch
def equal(self: tf.Tensor, b: tf.Tensor):
    return tf.equal(self, b).all()


@patch
def tensordot(self: tf.Tensor, b: tf.Tensor, axes, name=None):
    return tf.tensordot(self, b, axes, name=name)


@patch
def tensor_scatter_nd_add(self: tf.Tensor, indices, updates, name=None):
    return tf.tensor_scatter_nd_add(self, indices, updates, name=name)


@patch
def tensor_scatter_nd_sub(self: tf.Tensor, indices, updates, name=None):
    return tf.tensor_scatter_nd_sub(self, indices, updates, name=name)


@patch
def tensor_scatter_nd_update(self: tf.Tensor, indices, updates, name=None):
    return tf.tensor_scatter_nd_update(self, indices, updates, name=name)


@patch
def tile(self: tf.Tensor, multiples, name=None):
    return tf.tile(self, multiples, name=name)


@patch
def repeat(self: tf.Tensor, multiples, name=None):
    return self.tile(multiples, name=name)


@patch
def transpose(self: tf.Tensor, perm=None, conjugate=False, name='transpose'):
    return tf.transpose(self, perm=perm, conjugate=conjugate, name=name)


@patch
def permute(self: tf.Tensor, perm=None, conjugate=False, name='transpose'):
    return self.transpose(perm=perm, conjugate=conjugate, name=name)


@patch
def truncatediv(self: tf.Tensor, y: tf.Tensor, name=None):
    return tf.truncatediv(self, y, name=name)


@patch
def truncatemod(self: tf.Tensor, y: tf.Tensor, name=None):
    return tf.truncatemod(self, y, name=name)


@patch
def unique(self: tf.Tensor, out_idx=tf.int32, name=None):
    return tf.unique(self, out_idx, name=name)


@patch
def unique_with_counts(self: tf.Tensor, out_idx=tf.int32, name=None):
    return tf.unique_with_counts(self, out_idx, name=name)


@patch
def where(self: tf.Tensor, x=None, y=None, name=None):
    return tf.where(self, x=x, y=y, name=name)


@patch
def relu(self: tf.Tensor, name=None):
    return tf.nn.relu(self, name=name)


@patch
def selu(self: tf.Tensor, name=None):
    return tf.nn.selu(self, name=name)


@patch
def relu6(self: tf.Tensor, name=None):
    return tf.nn.relu6(self, name=name)


@patch
def tanh(self: tf.Tensor, name=None):
    return tf.nn.tanh(self, name=name)


@patch
def sigmoid(self: tf.Tensor, name=None):
    return tf.nn.sigmoid(self, name=name)


@patch
def leaky_relu(self: tf.Tensor, alpha=0.2, name=None):
    return tf.nn.leaky_relu(self, alpha=alpha, name=name)


@patch
def elu(self: tf.Tensor, name=None):
    return tf.nn.elu(self, name=name)


@patch
def softsign(self: tf.Tensor, name=None):
    return tf.nn.softsign(self, name=name)


@patch
def sign(self: tf.Tensor, name=None):
    return tf.sign(self, name=name)


@patch
def sqrt(self: tf.Tensor, name=None):
    return tf.sqrt(self, name=name)


@patch
def l2_loss(self: tf.Tensor, name=None):
    return tf.nn.l2_loss(self, name=name)


@patch
def log_softmax(self: tf.Tensor, axis=None, name=None):
    return tf.nn.log_softmax(self, axis=axis, name=name)


@patch
def softmax(self: tf.Tensor, axis=None, name=None):
    return tf.nn.softmax(self, axis=axis, name=name)


@patch
def flatten(self: tf.Tensor, name=None):
    return tf.nn.l2_loss(self, name=name)


@patch
def square(self: tf.Tensor, name=None):
    return tf.square(self, name=name)


@patch
def arctanh(self: tf.Tensor, name=None):
    return tf.arctanh(self, name=name)


@patch
def reduce_sum(self: tf.Tensor, axis=None, keepdims=False, name=None):
    if self.dtype == tf.bool:
        self = self.astype(tf.int64)
    return tf.reduce_sum(self, axis=axis, keepdims=keepdims, name=name)


@patch
def sum(self: tf.Tensor, axis=None, keepdims=False, name=None):
    return self.reduce_sum(axis=axis, keepdims=keepdims, name=name)


@patch
def reduce_prod(self: tf.Tensor, axis=None, keepdims=False, name=None):
    if self.dtype == tf.bool:
        self = self.astype(tf.int64)
    return tf.reduce_prod(self, axis=axis, keepdims=keepdims, name=name)


@patch
def prod(self: tf.Tensor, axis=None, keepdims=False, name=None):
    return self.reduce_prod(axis=axis, keepdims=keepdims, name=name)


@patch
def reduce_mean(self: tf.Tensor, axis=None, keepdims=False, name=None):
    if self.dtype == tf.bool:
        self = self.astype(tf.int64)
    return tf.reduce_mean(self, axis=axis, keepdims=keepdims, name=name)


@patch
def mean(self: tf.Tensor, axis=None, keepdims=False, name=None):
    return self.reduce_mean(axis=axis, keepdims=keepdims, name=name)


@patch
def reduce_min(self: tf.Tensor, axis=None, keepdims=False, name=None):
    return tf.reduce_min(self, axis=axis, keepdims=keepdims, name=name)


@patch
def min(self: tf.Tensor, axis=None, keepdims=False, name=None):
    return self.reduce_min(axis=axis, keepdims=keepdims, name=name)


@patch
def reduce_max(self: tf.Tensor, axis=None, keepdims=False, name=None):
    return tf.reduce_max(self, axis=axis, keepdims=keepdims, name=name)


@patch
def max(self: tf.Tensor, axis=None, keepdims=False, name=None):
    return self.reduce_max(axis=axis, keepdims=keepdims, name=name)


@patch
def argmin(self: tf.Tensor,
           axis=None,
           output_type=tf.int64,
           name=None,
           keepdims=False):
    out = tf.argmin(self, axis=axis, output_type=output_type, name=name)
    if keepdims:
        out = out.unsqueeze(axis=axis)
    return out


@patch
def argmax(self: tf.Tensor,
           axis=None,
           output_type=tf.int64,
           name=None,
           keepdims=False):
    out = tf.argmax(self, axis=axis, output_type=output_type, name=name)
    if keepdims:
        out = out.unsqueeze(axis=axis)
    return out


@patch
def minimum(self: tf.Tensor, y: tf.Tensor, name=None):
    return tf.minimum(self, y, name=name)


@patch
def maximum(self: tf.Tensor, y: tf.Tensor, name=None):
    return tf.maximum(self, y, name=name)


@patch
def sort(self: tf.Tensor, axis=-1, direction='ASCENDING', name=None):
    return tf.sort(self, axis=axis, direction=direction, name=name)


@patch
def top_k(self: tf.Tensor, k=1, sorted=True, name=None):
    return tf.math.top_k(self, k=k, sorted=sorted, name=name)


@patch
def topk(self: tf.Tensor, k=1, sorted=True, name=None):
    return self.top_k(k=k, sorted=sorted, name=name)


@patch
def rand(self: tf.Tensor,
         shape,
         minval=0,
         maxval=None,
         dtype=None,
         seed=None,
         name=None):
    if dtype is None:
        dtype = self.dtype
    return tf.random.uniform(shape,
                             minval=minval,
                             maxval=maxval,
                             dtype=dtype,
                             seed=seed,
                             name=name)


@patch
def uniform(self: tf.Tensor,
            shape,
            minval=0,
            maxval=None,
            dtype=None,
            seed=None,
            name=None):
    return self.rand(shape,
                     minval=minval,
                     maxval=maxval,
                     dtype=dtype,
                     seed=seed,
                     name=name)


@patch
def randn(self: tf.Tensor,
          shape,
          mean=0.0,
          stddev=1.0,
          dtype=None,
          seed=None,
          name=None):
    if dtype is None:
        dtype = self.dtype
    return tf.random.normal(shape,
                            mean=mean,
                            stddev=stddev,
                            dtype=dtype,
                            seed=seed,
                            name=name)


@patch
def normal(self: tf.Tensor,
           shape,
           mean=0.0,
           stddev=1.0,
           dtype=None,
           seed=None,
           name=None):
    return self.randn(shape,
                      mean=mean,
                      stddev=stddev,
                      dtype=dtype,
                      seed=seed,
                      name=name)


@patch
def item(self: tf.Tensor):
    return self.numpy().item()


@patch
def exp(self: tf.Tensor, name=None):
    return tf.exp(self, name=name)


@patch
def log(self: tf.Tensor, name=None):
    return tf.math.log(self, name=name)


@patch
def log2(self: tf.Tensor, name=None):
    return self.log(name=name) / tf.math.log(2.0)


@patch
def log10(self: tf.Tensor, name=None):
    return self.log(name=name) / tf.math.log(10.0)


@patch
def log1p(self: tf.Tensor, name=None):
    return tf.math.log1p(self, name=name)


@patch
def isnan(self: tf.Tensor, name=None):
    return tf.math.is_nan(self, name=name)


@patch
def isinf(self: tf.Tensor, name=None):
    return tf.math.is_inf(self, name=name)


@patch
def cumsum(self: tf.Tensor,
           axis=None,
           exclusive=False,
           reverse=False,
           name=None):
    if axis is None:
        self = self.reshape(-1)
        axis = 0
    return tf.cumsum(self,
                     axis=axis,
                     exclusive=exclusive,
                     reverse=reverse,
                     name=name)


@patch
def gather(self: tf.Tensor,
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


@patch
def matmul(self: tf.Tensor,
           b: tf.Tensor,
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


@patch
def getitem(self: tf.Tensor, index, axis=None):
    if isinstance(index, tuple):
        index = tuple(x if isinstance(x, Tensor) else x for x in index)
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