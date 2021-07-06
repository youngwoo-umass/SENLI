import numpy as np
import tensorflow as tf

from trainer.bert_optimizer import AdamWeightDecayOptimizer


def init_session(allow_soft_placement=True, log_device_placement=False):
    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                            log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = True

    return tf.Session(config=config)


def get_train_op(loss, lr, name='Adam'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-8,
                                       name=name)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, global_step



def get_accumulated_optimizer_from_config(loss, train_config, tvars, gradient_accmulation_multiplier):
    train_op = get_accumulated_optimizer(
        loss,
        train_config.learning_rate,
        train_config.num_train_steps,
        train_config.num_warmup_steps,
        train_config.use_tpu,
        tvars,
        gradient_accmulation_multiplier
    )
    return train_op


def get_accumulated_optimizer(loss, init_lr, num_train_steps,
                              tvars, gradient_accmulation_multiplier):
    global_step = tf.compat.v1.train.get_or_create_global_step()

    learning_rate = get_learning_rate(global_step, init_lr, num_train_steps)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    # compute batch gradient
    grads = tf.gradients(loss, tvars)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    # this is a list of sum(dy/dx) for each variable that must be paired with a tvars list.
    # element may be an IndexedSlices object that does not support assignning, e.g. [g.assign(value) for g in grads]
    # some of the elements are None, meaning y and x does not depend on each other.
    # Nonetypes must be handled using Python, tensorflow cannot convert Nonetypes to 0.

    # declare a temp variable for summation
    sum_gradient = list([ tf.Variable(name="sum_grads" + str(i),
                                 initial_value=np.zeros(tv.shape),
                                 trainable=False,
                                 dtype=tf.float32,
                                 ) for i, tv in enumerate(tvars)])
    sum_ops = []
    unused_variable_in_batch = []

    # gradient accumulation
    for i, gv in enumerate(grads):
        if gv is not None:
            sum_ops.append(sum_gradient[i].assign_add(gv, name="accumulate_gradient"))
        else:
            unused_variable_in_batch.append(sum_gradient[i])
            sum_gradient[i] = None

    # NOTE : calling .assign_add does NOTHING in estimator, must wrap them all and handle them via train_ops

    def apply_accumulated_gradients(sums):
        # normalize gradient
        normalize_ops = []
        for i, g in enumerate(sums):
            if g is not None:
                normalize_ops.append(sums[i].assign(tf.multiply(g, 1 / gradient_accmulation_multiplier)))
                # assign to make sure it still is a variable, or else it will become a Tensor
        with tf.control_dependencies(normalize_ops):
            minimize_op = optimizer.apply_gradients(zip(sums, tvars), global_step=global_step)
        return tf.group(minimize_op, *normalize_ops, name="apply_accumulated_gradients")

    train_op = tf.cond(tf.math.equal(global_step % gradient_accmulation_multiplier, 0),
                       lambda: apply_accumulated_gradients(sum_gradient),
                       lambda: optimizer.apply_gradients(zip([None for _ in grads], tvars), global_step=global_step))

    # reset accumulation when necessary
    def reset():
        counter = 0
        for i, s in enumerate(sum_gradient):
            if s is None:
                # restore reference from None to the original variable
                sum_gradient[i] = unused_variable_in_batch[counter]
                counter += 1
        return tf.group([s.assign(tf.zeros_like(s)) for s in sum_gradient])

    with tf.control_dependencies([train_op]):
        reset_ops = tf.cond(tf.math.equal(global_step % gradient_accmulation_multiplier, 0),
                            reset,
                            tf.no_op)
    # the 2 branches must have identical structure, [op1, op2, ...] || no_op cannot be valid cond branch.
    # tf.group to convert all resets into 1 op and match with no_op: tf.group() || np_op

    # Increment global step
    new_global_step = global_step + 1
    train_op = tf.group(*sum_ops, [train_op, global_step.assign(new_global_step), reset_ops])
    return train_op


def get_gradient_acc_train_op(loss, lr, num_train_steps=0):
    tvars = tf.trainable_variables()
    return get_accumulated_optimizer(loss, lr, num_train_steps, tvars, 2)


def get_train_op2(loss, lr, name='Adam', num_train_steps=0):
    tvars = tf.trainable_variables()
    grads = tf.gradients(ys=loss, xs=tvars)
    return get_train_op_from_grads_and_tvars(grads, tvars, lr, name, num_train_steps)


def get_train_op_from_grads_and_tvars(grads, tvars, lr, name='Adam', num_train_steps=0):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = get_learning_rate_w_warmup(global_step, lr, num_train_steps, 10000)

    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.02,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    return train_op


def get_train_op_from_cliped_grads_and_tvars(cliped_grads, tvars, lr, name='Adam', num_train_steps=0):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = get_learning_rate_w_warmup(global_step, lr, num_train_steps, 10000)

    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.02,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    # This is how the model was pre-trained.
    grads, _ = cliped_grads

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    return train_op



def get_train_op_wo_gstep_update2(loss, lr, name, num_train_steps):
    tvars = tf.trainable_variables()
    grads = tf.gradients(ys=loss, xs=tvars)
    return get_train_op_wo_gstep_update(grads, tvars, lr, name, num_train_steps)


# This function does not update global_step
def get_train_op_wo_gstep_update(grads, tvars, lr, name, num_train_steps):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = get_learning_rate_w_warmup(global_step, lr, num_train_steps, 10000)

    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.02,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        name=name
    )
    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    return global_step, train_op


def get_learning_rate(global_step, lr, num_train_steps):
    if num_train_steps:
        learning_rate = tf.constant(value=lr, shape=[], dtype=tf.float32)

        # Implements linear decay of the learning rate.
        learning_rate = tf.compat.v1.train.polynomial_decay(
            learning_rate,
            global_step,
            num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)
    else:
        learning_rate = lr
    return learning_rate

def get_learning_rate_w_warmup(global_step, init_lr, num_train_steps, num_warmup_steps):
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
    return learning_rate
