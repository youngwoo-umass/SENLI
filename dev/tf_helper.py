import tensorflow as tf
from tensorflow.python.data.experimental import AutoShardPolicy


def apply_gradient_warning_less(optimizer, gradients, trainable_variables):
    l = [(grad, var) for (grad, var) in zip(gradients, trainable_variables) if grad is not None]
    optimizer.apply_gradients(l)


def distribute_dataset(strategy, dataset):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset