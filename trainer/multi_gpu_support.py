import tensorflow as tf

from trainer.tf_train_module import get_train_op_from_grads_and_tvars, get_train_op_wo_gstep_update


def get_multiple_models(model_init_fn, n_gpu):
    models = []
    root_scope = tf.get_variable_scope()

    for gpu_idx in range(n_gpu):
        with tf.device("/gpu:{}".format(gpu_idx)):
            with tf.variable_scope(root_scope, reuse= gpu_idx >0):
                models.append(model_init_fn())

    return models


def get_averaged_gradients(losses):
    tvars = tf.trainable_variables()
    n_gpu = len(losses)

    tower_grads = []
    for gpu_idx, loss in enumerate(losses):
        with tf.device("/gpu:{}".format(gpu_idx)):
            grads = tf.gradients(loss, tvars)
            tower_grads.append(grads)

    avg_grads = []
    for t_idx, _ in enumerate(tvars):
        first_item = tower_grads[0][t_idx]
        if first_item is not None:
            try:
                g_list = [tower_grads[gpu_idx][t_idx] for gpu_idx in range(n_gpu)]
                g_avg = tf.reduce_mean(g_list, axis=0)
            except TypeError:
                g_list = [tf.convert_to_tensor(t) for t in g_list]
                g_avg = tf.reduce_mean(g_list, axis=0)
        else:
            g_avg = None

        avg_grads.append(g_avg)
    return avg_grads


def get_loss(model):
    return model.loss


def get_avg_loss(models):
    return get_avg_tensors_from_models(models, get_loss)


def get_avg_tensors_from_models(models, get_tensor_fn):
    sum = 0
    for model in models:
        sum += get_tensor_fn(model)
    return sum / len(models)


def get_concat_tensors_from_models(models, get_tensor_fn):
    return tf.concat([get_tensor_fn(model) for model in models], axis=0)


def get_concat_tensors_list_from_models(models, get_tensor_list_fn):
    r = None
    for model in models:
        tensor_list = get_tensor_list_fn(model)
        if r is None:
            r = list([[tensor] for tensor in tensor_list])
        else:
            for idx, tensor in enumerate(tensor_list):
                r[idx].append(tensor)

    return [tf.concat(tensors, axis=0) for tensors in r]


def get_train_op(losses, lr, num_steps):
    gradients = get_averaged_gradients(losses)

    with tf.device("/device:CPU:0"):
        train_cls = get_train_op_from_grads_and_tvars(gradients,
                                                      tf.trainable_variables(), lr, "adam", num_steps)
    return train_cls


def get_other_train_op_multi_gpu(losses, lr, num_steps):
    gradients = get_averaged_gradients(losses)

    with tf.device("/device:CPU:0"):
        train_cls = get_train_op_wo_gstep_update(gradients, tf.trainable_variables(), lr, "adam", num_steps)
    return train_cls



def get_batch2feed_dict_for_multi_gpu(models):
    n_gpu = len(models)

    def batch2feed_dict(batch):
        n_elem = len(batch)
        inst_per_gpu = int(len(batch[0]) / n_gpu)
        feed_dict = {}
        for idx, model in enumerate(models):
            st = inst_per_gpu * idx
            ed = inst_per_gpu * (idx + 1)

            sub_batch = []
            for elem_idx in range(n_elem):
                sub_batch.append(batch[elem_idx][st:ed])

            feed_dict.update(model.batch2feed_dict(sub_batch))
        return feed_dict
    return batch2feed_dict