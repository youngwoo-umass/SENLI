from functools import partial

import tensorflow as tf

from cache import load_cache, save_to_pickle
from data_generator.NLI import nli
from data_generator.shared_setting import BertNLI
from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.nli_common import train_classification_factory, valid_fn_factory, save_fn_factory
from transformer import hyperparams
from transformer.nli_base import transformer_nli_pooled
from tf_logging import set_level_debug
from trainer.get_param_num import get_param_num
from trainer.model_saver import save_model_to_dir_path, load_bert_v2, tf_logger
from trainer.multi_gpu_support import get_multiple_models, get_averaged_gradients, get_avg_loss, \
    get_avg_tensors_from_models
from trainer.tf_module import get_nli_batches_from_data_loader, step_runner
from trainer.tf_train_module import get_train_op2, init_session, get_train_op_from_grads_and_tvars


def train_nli(hparam, nli_setting, save_dir, max_steps, data, model_path, load_fn):
    print("Train nil :", save_dir)

    task = transformer_nli_pooled(hparam, nli_setting.vocab_size)
    with tf.variable_scope("optimizer"):
        train_cls = get_train_op2(task.loss, hparam.lr, "adam", max_steps)

    train_batches, dev_batches = data

    sess = init_session()
    sess.run(tf.global_variables_initializer())
    if model_path is not None:
        load_fn(sess, model_path)

    def batch2feed_dict(batch):
        x0, x1, x2, y  = batch
        feed_dict = {
            task.x_list[0]: x0,
            task.x_list[1]: x1,
            task.x_list[2]: x2,
            task.y: y,
        }
        return feed_dict

    train_classification = partial(train_classification_factory, sess, task.loss, task.acc, train_cls, batch2feed_dict)

    global_step = tf.train.get_or_create_global_step()
    valid_fn = partial(valid_fn_factory, sess, dev_batches[:100], task.loss, task.acc, global_step, batch2feed_dict)

    save_fn = partial(save_fn_factory, sess, save_dir, global_step)
    init_step,  = sess.run([global_step])
    print("Initialize step to {}".format(init_step))
    print("{} train batches".format(len(train_batches)))
    valid_freq = 5000
    save_interval = 10000
    loss, _ = step_runner(train_batches, train_classification, init_step,
                          valid_fn, valid_freq,
                          save_fn, save_interval, max_steps)

    return save_fn()


def train_nli_multi_gpu(hparam, nli_setting, save_dir, num_steps, data, model_path, load_fn, n_gpu):
    print("Train nil :", save_dir)
    model_init_fn = lambda: transformer_nli_pooled(hparam, nli_setting.vocab_size)
    models = get_multiple_models(model_init_fn, n_gpu)
    losses = [model.loss for model in models]
    gradients = get_averaged_gradients(losses)
    avg_loss = get_avg_loss(models)
    avg_acc = get_avg_tensors_from_models(models, lambda model:model.acc)

    with tf.variable_scope("optimizer"):
        with tf.device("/device:CPU:0"):
            train_cls = get_train_op_from_grads_and_tvars(gradients,
                                                          tf.trainable_variables(), hparam.lr, "adam", num_steps)

    print("Number of parameter : ", get_param_num())
    train_batches, dev_batches = data

    sess = init_session()
    sess.run(tf.global_variables_initializer())
    if model_path is not None:
        load_fn(sess, model_path)

    def batch2feed_dict(batch):
        x0, x1, x2, y = batch
        batch_size = len(x0)
        batch_size_per_gpu = int(batch_size / n_gpu)

        feed_dict = {}
        for gpu_idx in range(n_gpu):
            st = batch_size_per_gpu * gpu_idx
            ed = batch_size_per_gpu * (gpu_idx + 1)
            feed_dict[models[gpu_idx].x_list[0]] = x0[st:ed]
            feed_dict[models[gpu_idx].x_list[1]] = x1[st:ed]
            feed_dict[models[gpu_idx].x_list[2]] = x2[st:ed]
            feed_dict[models[gpu_idx].y] = y[st:ed]
        return feed_dict
    global_step = tf.train.get_or_create_global_step()
    train_classification = partial(train_classification_factory, sess, avg_loss, avg_acc, train_cls, batch2feed_dict)
    valid_fn = partial(valid_fn_factory, sess, dev_batches[:100], avg_loss, avg_acc, global_step, batch2feed_dict)

    def save_fn():
        return save_model_to_dir_path(sess, save_dir, global_step)

    init_step,  = sess.run([global_step])
    print("Initialize step to {}".format(init_step))
    print("{} train batches".format(len(train_batches)))
    valid_freq = 5000
    save_interval = 7200
    loss, _ = step_runner(train_batches, train_classification, init_step,
                              valid_fn, valid_freq,
                              save_fn, save_interval, num_steps)

    return save_fn()


def eval_nli(hparam, nli_setting, save_dir, data, model_path, load_fn):
    print("Train nil :", save_dir)

    task = transformer_nli_pooled(hparam, nli_setting.vocab_size)

    train_batches, dev_batches = data


    sess = init_session()
    sess.run(tf.global_variables_initializer())
    load_fn(sess, model_path)

    def batch2feed_dict(batch):
        x0, x1, x2, y  = batch
        feed_dict = {
            task.x_list[0]: x0,
            task.x_list[1]: x1,
            task.x_list[2]: x2,
            task.y: y,
        }
        return feed_dict

    global_step = tf.train.get_or_create_global_step()
    valid_fn = partial(valid_fn_factory, sess, dev_batches, task.loss, task.acc, global_step, batch2feed_dict)


    return valid_fn()


def train_nil_from_v2_checkpoint(run_name, model_path):
    steps = 12271
    return train_nil_from(run_name, model_path, load_bert_v2, steps)


def train_nil_from(save_dir, model_path, load_fn, max_steps):
    hp = hyperparams.HPTrain()
    set_level_debug()
    nli_setting = BertNLI()
    data = get_nli_data(hp, nli_setting)
    train_nli(hp, nli_setting, save_dir, max_steps, data, model_path, load_fn)


def get_nli_data(hp, nli_setting):
    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    tokenizer = get_tokenizer()
    CLS_ID = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    SEP_ID = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
    data_loader.CLS_ID = CLS_ID
    data_loader.SEP_ID = SEP_ID
    cache_name = "nli_batch{}_seq{}".format(hp.batch_size, hp.seq_max)
    data = load_cache(cache_name)
    if data is None:
        tf_logger.info("Encoding data from csv")
        data = get_nli_batches_from_data_loader(data_loader, hp.batch_size)
        save_to_pickle(data, cache_name)
    return data