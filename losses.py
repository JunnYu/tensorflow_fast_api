import tensorflow as tf


class PytorchCrossEntropyLoss:
    def __init__(self,
                 from_logits: bool = True,
                 reduction: str = "mean",
                 weight: tf.Tensor = None,
                 ignore_index: int = -100,
                 name: str = 'pytorch_cross_entropy_loss'):
        super().__init__()
        self.name = name
        self.from_logits = from_logits
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def __call__(self, y_true, y_pred):
        return self.call(y_true, y_pred)

    @tf.function
    def call(self, y_true, y_pred):
        active_loss = tf.not_equal(tf.reshape(y_true, (-1, )),
                                   self.ignore_index)
        y_pred = tf.boolean_mask(tf.reshape(y_pred, (-1, y_pred.shape[-1])),
                                 active_loss)
        y_true = tf.boolean_mask(tf.reshape(y_true, (-1, )), active_loss)
        loss = tf.losses.sparse_categorical_crossentropy(
            y_true=y_true, y_pred=y_pred, from_logits=self.from_logits)

        if self.weight is None:
            if self.reduction == "mean":
                loss = tf.reduce_mean(loss)
            elif self.reduction == "sum":
                loss = tf.reduce_sum(loss)
            elif self.reduction == "none":
                pass
            else:
                raise ValueError(
                    "reduction must choose from [mean, sum, none]")
        else:
            values = tf.gather(self.weight, y_true)
            if self.reduction == "mean":
                loss = tf.reduce_sum(values * loss) / tf.reduce_sum(values)
            elif self.reduction == "sum":
                loss = tf.reduce_sum(values * loss)
            elif self.reduction == "none":
                loss = values * loss
            else:
                raise ValueError(
                    "reduction must choose from [mean, sum, none]")

        return loss