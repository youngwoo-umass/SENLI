import logging
import re

import tensorflow as tf

from cpath import model_path, output_path
from misc_lib import *
from tf_v2_support import tf1

tf_logger = logging.getLogger('tensorflow')



def get_canonical_model_path(name):
    run_dir = os.path.join(model_path, 'runs')
    save_dir = os.path.join(run_dir, name)
    exist_or_mkdir(model_path)
    exist_or_mkdir(run_dir)
    exist_or_mkdir(save_dir)
    return save_dir


def setup_summary_writer(exp_name):
    summary_path = os.path.join(output_path, "summary")
    exist_or_mkdir(summary_path)
    summary_run_path = os.path.join(summary_path, exp_name)
    exist_or_mkdir(summary_run_path)

    train_log_path = os.path.join(summary_run_path, "train")
    test_log_path = os.path.join(summary_run_path, "test")
    train_writer = tf.summary.FileWriter(train_log_path)
    test_writer = tf.summary.FileWriter(test_log_path)
    return train_writer, test_writer


def save_model(sess, name, global_step):
    run_dir = os.path.join(model_path, 'runs')
    save_dir = os.path.join(run_dir, name)

    exist_or_mkdir(model_path)
    exist_or_mkdir(run_dir)
    exist_or_mkdir(save_dir)
    return save_model_to_dir_path(sess, save_dir, global_step)


def save_model_to_dir_path(sess, save_dir, global_step):
    path = os.path.join(save_dir, "model")
    saver = tf1.train.Saver(tf1.global_variables(), max_to_keep=1)
    ret = saver.save(sess, path, global_step=global_step)
    tf_logger.info("Model saved at {} - {}".format(path, ret))
    return ret


def load_model(sess, model_path):
    loader = tf1.train.Saver(tf1.global_variables(), max_to_keep=1)
    loader.restore(sess, model_path)


def load_bert_v2(sess, model_path):
    tvars = tf.contrib.slim.get_variables_to_restore()
    name_to_variable = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
        print("Current vars : ", name)

    init_vars = tf.train.list_variables(model_path)

    initialized = set()
    load_mapping = dict()
    for v in init_vars:
        name_tokens = v[0].split('/')
        checkpoint_name = '/'.join(name_tokens).split(":")[0]
        tvar_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", checkpoint_name)
        tvar_name = re.sub("dense[_]?\d*", "dense", tvar_name)
        print(tvar_name)
        if tvar_name in name_to_variable:
            print("{} -> {}".format(checkpoint_name, tvar_name))
            tf_logger.debug("{} -> {}".format(checkpoint_name, tvar_name))
            load_mapping[checkpoint_name] = name_to_variable[tvar_name]
            initialized.add(tvar_name)
        else:
            print("NOT used : ", tvar_name)
            raise Exception()

    for name in name_to_variable:
        if name not in initialized :
            print(name, "not initialized")
    print("Restoring: {}".format(model_path))
    loader = tf.train.Saver(load_mapping, max_to_keep=1)
    loader.restore(sess, model_path)


def load_v2_to_v2(sess, model_path, exception_on_not_used = True):
    tvars = tf1.global_variables()

    def get_simple_name(name):
        name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
        name = re.sub("dense[_]?\d*", "dense", name)
        return name

    name_to_variable = {}
    real_name = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        simple_name = get_simple_name(name)
        name_to_variable[simple_name] = var
        real_name[simple_name] = name
        print("Current vars : ", name)

    init_vars = tf.train.list_variables(model_path)

    initialized = set()
    load_mapping = dict()
    for v in init_vars:
        name_tokens = v[0].split('/')
        checkpoint_name = '/'.join(name_tokens).split(":")[0]
        simple_name = get_simple_name(checkpoint_name)
        print(checkpoint_name)
        if simple_name in name_to_variable:
            print("{} -> {}".format(checkpoint_name, real_name[simple_name]))
            tf_logger.debug("{} -> {}".format(checkpoint_name, real_name[simple_name]))
            load_mapping[checkpoint_name] = name_to_variable[simple_name]
            initialized.add(real_name[simple_name])
        else:

            print("NOT used : ", checkpoint_name)
            if exception_on_not_used:
                raise Exception()

    for simple_name in name_to_variable:
        name = real_name[simple_name]
        if name not in initialized:
            print(name, "not initialized")
    print("Restoring: {}".format(model_path))
    loader = tf1.train.Saver(load_mapping, max_to_keep=1)
    loader.restore(sess, model_path)



def load_model_w_scope(sess, path, include_namespace, verbose=True):
    def condition(v):
        if v.name.split('/')[0] in include_namespace:
            return True
        return False

    variables = tf.contrib.slim.get_variables_to_restore()
    variables_to_restore = [v for v in variables if condition(v)]
    if verbose:
        for v in variables_to_restore:
            print(v)
    loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
    loader.restore(sess, path)


def load_model_with_blacklist(sess, path, exclude_namespace, verbose=True):
    def exclude(v):
        if v.name.split('/')[0] in exclude_namespace:
            return True
        return False

    print("Model path : ", path)

    variables = tf.contrib.slim.get_variables_to_restore()
    variables_to_restore = [v for v in variables if not exclude(v)]
    if verbose:
        for v in variables_to_restore:
            print(v)
        for v in variables:
            if exclude(v):
                print("Skip: ", v)
    loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
    loader.restore(sess, path)


def get_model_path(run_name, step_name):
    return os.path.join(model_path, 'runs', run_name, step_name)