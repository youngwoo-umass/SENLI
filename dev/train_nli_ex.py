import argparse
import os
import re
import sys

from tensorflow.python.data.experimental import AutoShardPolicy
from tensorflow.python.distribute.distribute_lib import Strategy

from tf_logging import senli_logging

from functools import partial
from typing import Tuple, Callable, List
import numpy as np
from tensorflow.python.types.core import Tensor

from dev.explain_model import CrossEntropyModeling, CorrelationModeling, tag_informative_eq, ExplainModeling
from dev.explain_trainer import ExplainTrainerM
from dev.nli_common import get_nli_data, tags
from dev.bert_common import BERT_CLS, eval_fn, ModelConfig, BERT_CLS_EX, load_pooler, is_interesting_step

import bert
from tensorflow import keras
import tensorflow as tf

from dev.optimize import get_learning_rate_w_warmup, AdamWeightDecayOptimizer
from path_manager import bert_model_folder


class RunConfigEx:
    batch_size = 16
    num_classes = 3
    train_step = 0
    eval_every_n_step = 100
    save_every_n_step = 5000
    learning_rate = 1e-5
    model_save_path = "saved_model_ex"
    init_checkpoint = ""
    checkpoint_type = "bert"


def get_run_config() -> RunConfigEx:
    parser = argparse.ArgumentParser(description='File should be stored in ')
    parser.add_argument("--model_save_path")
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--checkpoint_type")
    args = parser.parse_args(sys.argv[1:])
    run_config = RunConfigEx()
    nli_train_data_size = 392702
    step_per_epoch = int(nli_train_data_size / run_config.batch_size)

    if args.model_save_path:
        run_config.model_save_path = args.model_save_path

    run_config.checkpoint_type = args.checkpoint_type

    if args.init_checkpoint:
        run_config.init_checkpoint = args.init_checkpoint

    if run_config.checkpoint_type in ["none", "bert"]:
        num_epoch = 2
    elif run_config.checkpoint_type == "nli_saved_model":
        num_epoch = 1
    else:
        assert False

    run_config.train_step = num_epoch * step_per_epoch

    return run_config


@tf.function
def train_cls(model: keras.Model, item, loss_fn, optimizer):
    x1, x2, y = item
    with tf.GradientTape() as tape:
        prediction, _ = model([x1, x2], training=True)
        loss = loss_fn(y, prediction)

    senli_logging.debug("train_ex_tf_fn called")
    gradients = tape.gradient(loss, model.trainable_variables)
    apply_gradient_warning_less(optimizer, gradients, model.trainable_variables)
    return loss


@tf.function
def distributed_train_step(mirrored_strategy, train_step_fn, dist_inputs: Tuple):
    per_replica_losses = mirrored_strategy.run(train_step_fn, args=dist_inputs)
    loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    new_global_step = global_step + 1
    global_step.assign(new_global_step)
    return loss


class ExTrainConfig:
    num_deletion = 20
    g_val = 0.5
    save_train_payload = False
    # drop_thres = 0.3
    drop_thres = 0.0


def apply_gradient_warning_less(optimizer, gradients, trainable_variables):
    l = [(grad, var) for (grad, var) in zip(gradients, trainable_variables) if grad is not None]
    optimizer.apply_gradients(l)


def distribute_dataset(strategy, dataset):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset


# This defines graph
def init_ex_trainer(bert_cls_ex: BERT_CLS_EX,
                    run_config,
                    forward_run: Callable,
                    dist_strategy,
                    ):
    modeling_option = "ce"
    ex_modeling_class = {
        'ce': CrossEntropyModeling,
        'co': CorrelationModeling
    }[modeling_option]
    lr_factor = 0.3
    lr = 1e-5 * lr_factor
    ex_config = ExTrainConfig()
    ex_config.num_deletion = 3
    information_fn_list = list([partial(tag_informative_eq, t) for t in tags])
    def cls_batch_to_input(cls_batch):
        x0, x1 = cls_batch
        x0 = dist_strategy.gather(x0, axis=0)
        x1 = dist_strategy.gather(x1, axis=0)
        return x0, x1

    explain_trainer = ExplainTrainerM(information_fn_list,
                                      len(tags),
                                      action_to_label=ex_modeling_class.action_to_label,
                                      get_null_label=ex_modeling_class.get_null_label,
                                      forward_run=forward_run,
                                      batch_size=run_config.batch_size,
                                      num_deletion=ex_config.num_deletion,
                                      g_val=ex_config.g_val,
                                      drop_thres=ex_config.drop_thres,
                                      cls_batch_to_input=cls_batch_to_input,
                                      )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE,
        from_logits=True
    )
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_ex_tf_fn(*ex_train_batch):
        senli_logging.debug("train_ex_tf_fn ENTRY")
        senli_logging.debug(ex_train_batch[0].shape)
        loss_list = []
        with tf.GradientTape() as tape:
            x1, x2 = ex_train_batch[:2]
            _, ex_logits_list = bert_cls_ex.model([x1, x2], training=True)
            loss = 0
            for i in range(len(tags)):
                st = i * 2 + 2
                ed = i * 2 + 4
                ex_labels, valid_mask = ex_train_batch[st:ed]
                ex_logits = ex_logits_list[i]
                losses = loss_fn(ex_labels, ex_logits)
                losses = tf.cast(valid_mask, tf.float32) * losses
                loss_per_tag = tf.reduce_mean(losses)
                loss_list.append(loss_per_tag)
                loss += loss_per_tag
        senli_logging.debug("train_ex_tf_fn 1")
        trainable_variables = bert_cls_ex.model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        apply_gradient_warning_less(optimizer, gradients, trainable_variables)
        senli_logging.debug("train_ex_tf_fn EXIT")
        return loss_list

    def train_ex_tf_fn_distributed(ex_batches) -> np.ndarray:
        senli_logging.debug("train_ex_tf_fn_distributed ENTRY")
        ex_batches = distribute_dataset(dist_strategy, ex_batches)
        losses_list = []
        for batch in ex_batches:
            ret = dist_strategy.run(train_ex_tf_fn, batch)
            ret = dist_strategy.gather(ret, axis=0)
            losses_list.append(ret)

        losses = np.sum(losses_list, axis=0)
        senli_logging.debug("train_ex_tf_fn_distributed EXIT")
        return losses

    def train_ex_fn(cls_batch) -> str:
        ex_losses = explain_trainer.weak_supervision(cls_batch, train_ex_tf_fn_distributed)
        out_str = "ex_train_loss : "
        for tag_idx, loss_val in enumerate(ex_losses):
            out_str += " {0:.2f}".format(loss_val)
        return out_str

    return train_ex_fn


@tf.function
def get_cls_logits(model, batch):
    cls_logits, _ = model(batch, training=False)
    return cls_logits


def load_from_nli_saved_model(bert_cls_ex: BERT_CLS_EX, ckpt_path):
    senli_logging.debug("load_from_nli_only ENTRY")
    senli_logging.debug("checkpoint_path: {}".format(ckpt_path))
    imported = tf.saved_model.load(ckpt_path)
    import_var_d = {}
    for v in imported.variables:
        import_var_d[v.name] = v.numpy()
    senli_logging.debug("state: {}".format(imported))
    variables = bert_cls_ex.model.trainable_variables
    todo = []
    for cur_v in variables:
        if cur_v.name in import_var_d:
            senli_logging.debug("initialize {} ".format(cur_v.name))
            todo.append((cur_v, import_var_d[cur_v.name]))
        else:
            senli_logging.debug("skip {} ".format(cur_v.name))
    keras.backend.batch_set_value(todo)
    senli_logging.debug("load_from_nli_only DONE")
    return


class EvalObject:
    def __init__(self, model, eval_batches, dist_strategy: Strategy, compute_loss):
        self.loss = tf.keras.metrics.Mean(name='dev_loss')
        self.acc = tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
        self.eval_batches = eval_batches
        self.model = model
        self.compute_loss = compute_loss
        self.dist_strategy: Strategy = dist_strategy

    def do_eval(self):
        self.acc.reset_state()
        for e_batch in self.eval_batches:
            args = (self.model, e_batch, self.compute_loss, self.loss, self.acc)
            per_replica = self.dist_strategy.run(eval_fn, args=args)
        eval_loss = self.loss.result().numpy()
        eval_acc = self.acc.result().numpy()
        return eval_loss, eval_acc


def load_checkpoint(bert_cls_ex: BERT_CLS_EX, run_config: RunConfigEx):
    checkpoint_path = run_config.init_checkpoint
    if run_config.checkpoint_type == 'bert':
        bert.load_stock_weights(bert_cls_ex.l_bert, checkpoint_path)
        load_pooler(bert_cls_ex.pooler, checkpoint_path)
    elif run_config.checkpoint_type == "nli_saved_model":
        load_from_nli_saved_model(bert_cls_ex, run_config.init_checkpoint)


def main():
    # Train the model here
    debug_run = False
    bert_params = bert.params_from_pretrained_ckpt(bert_model_folder)
    model_config = ModelConfig()
    run_config: RunConfigEx = get_run_config()
    step_per_epoch = int(400 * 1000 / 16)

    if run_config.checkpoint_type == "bert":
        step_to_start_ex_train = step_per_epoch * 2
    elif run_config.checkpoint_type == "nli_saved_model":
        step_to_start_ex_train = 0
    else:
        assert False

    model_save_path: str = run_config.model_save_path
    dist_strategy = tf.distribute.MirroredStrategy()
    batch_size: int = run_config.batch_size
    with dist_strategy.scope():
        bert_cls_ex = BERT_CLS_EX(bert_params, model_config, len(tags))
        model = bert_cls_ex.model

        loss_fn_inner = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            per_example_loss = loss_fn_inner(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

        def forward_run_fn(dist_batches) -> np.ndarray:
            senli_logging.debug("forward_run_fn ENTRY")
            out_t = []
            for batch in dist_batches:
                t = dist_strategy.run(get_cls_logits, (model, batch))
                t = dist_strategy.gather(t, axis=0)
                out_t.append(t.numpy())
            senli_logging.debug("forward_run_fn EXIT")
            return np.concatenate(out_t)

        train_ex_tf_fn = init_ex_trainer(bert_cls_ex, run_config, forward_run_fn, dist_strategy)

    optimizer = tf.optimizers.Adam(learning_rate=1e-5)
    eval_batches, train_dataset = load_nli_dataset(batch_size, debug_run, model_config)
    eval_batches = distribute_dataset(dist_strategy, eval_batches)
    dist_train_dataset = distribute_dataset(dist_strategy, train_dataset)

    eval_object = EvalObject(model, eval_batches, dist_strategy, compute_loss)
    train_itr = iter(dist_train_dataset)

    load_checkpoint(bert_cls_ex, run_config)
    for step_idx in range(run_config.train_step):
        f_do_cls_train = True
        f_do_ex_train = step_idx >= step_to_start_ex_train
        f_do_eval = step_idx % run_config.eval_every_n_step == 0
        f_do_save = step_idx % run_config.save_every_n_step == 0

        batch_item = next(train_itr)
        x1, x2, y = batch_item
        per_step_msg = "step {0}".format(step_idx)
        if f_do_cls_train:
            args = model, batch_item, compute_loss, optimizer
            train_loss = distributed_train_step(dist_strategy, train_cls, args)
            train_loss = np.array(train_loss)
            per_step_msg += " train_cls_loss={0:.2f}".format(train_loss)

        if f_do_ex_train:
            batch_x = [x1, x2]
            msg = train_ex_tf_fn(batch_x)
            per_step_msg += " " + msg

        if f_do_eval:
            eval_loss, eval_acc = eval_object.do_eval()
            per_step_msg += " dev_cls_loss={0:.2f} dev_cls_acc={1:.2f}".format(eval_loss, eval_acc)

        if f_do_save:
            model.save(model_save_path)
            senli_logging.info("Model saved at {}".format(model_save_path))

        if f_do_eval or is_interesting_step(step_idx):
            senli_logging.info(per_step_msg)
    senli_logging.info("Training completed")
    model.save(model_save_path)
    senli_logging.info("Model saved at {}".format(model_save_path))


def load_nli_dataset(batch_size, debug_run, model_config):
    dev_dataset = get_nli_data(model_config.max_seq_length, "dev_matched.tsv")
    eval_batches = dev_dataset.batch(batch_size).take(10)
    if debug_run:
        train_dataset = get_nli_data(model_config.max_seq_length, "dev_matched.tsv")
    else:
        train_dataset = get_nli_data(model_config.max_seq_length, "train.tsv")
    train_dataset = train_dataset.repeat(4).batch(batch_size)
    return eval_batches, train_dataset


if __name__ == "__main__":
    main()


