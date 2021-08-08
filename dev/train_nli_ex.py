import argparse
import datetime
import sys
from functools import partial
from typing import Tuple, Callable, List, Dict

import bert
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.distribute.distribute_lib import Strategy

from dev.bert_common import BERT_CLS_EX, eval_fn, load_pooler
from dev.ex_eval import EvalSet, DataID, scores_to_ap
from dev.explain_model import CrossEntropyModeling, CorrelationModeling, tag_informative_eq
from dev.explain_trainer import ExplainTrainerM
from dev.nli_common import tags
from dev.optimize import get_optimizer
from dev.tf_helper import apply_gradient_warning_less, distribute_dataset
from misc_lib import average
from senli_log import senli_logging


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

    senli_logging.debug("train_cls called")
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
    drop_thres = 0.3


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
    optimizer = get_optimizer(2e-5, run_config.train_step)

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
            out_str += " {0:.4f}".format(loss_val)
        return out_str

    return train_ex_fn


@tf.function
def get_cls_logits(model, batch):
    cls_logits, _ = model(batch, training=False)
    return cls_logits


@tf.function
def get_ex_scores(model: keras.Model, batch):
    _, ex_logits = model(batch, training=False)
    ex_logits = tf.stack(ex_logits, axis=1)
    probs = tf.nn.softmax(ex_logits, axis=3)
    return probs[:, :, :, 1]


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


@tf.function
def eval_fn(model, item, loss_fn, dev_loss, dev_acc):
    x1, x2, y = item
    prediction, _ = model([x1, x2], training=False)
    loss = loss_fn(y, prediction)
    dev_loss.update_state(loss)
    pred = tf.argmax(prediction, axis=1)
    dev_acc.update_state(y, pred)


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


score_valid_policy = {
    'match': [0],
    'conflict': [0, 1],
    'mismatch': [1]
}


def two_digit_float(f):
    return "{0:.2f}".format(f)


def dict_to_tuple_list(d):
    out_l = []
    for k, v in d.items():
        out_l.append((k, v))

    return out_l


def merge_and_sort_scores(token_scores: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    d = {}
    for idx, score in token_scores:
        if idx not in d:
            d[idx] = score
        else:
            d[idx] = max(d[idx], score)

    scores = dict_to_tuple_list(d)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


class ExEvaluator:
    def __init__(self, model: keras.Model,
                 tags: List[str],
                 ex_eval_data: List[Tuple[str, EvalSet]],
                 batch_size: int,
                 dist_strategy: Strategy,
                 summary_writer
                 ):
        self.loss = tf.keras.metrics.Mean(name='dev_loss')
        self.acc = tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
        self.ex_eval_data: List[Tuple[str, EvalSet]] = ex_eval_data
        self.model: keras.Model = model
        self.dist_strategy: Strategy = dist_strategy
        self.batch_size: int = batch_size
        self.log_idx = 0
        self.tags = tags
        self.summary_writer = summary_writer

    def get_tag_idx_from_prefix(self, tag_prefix):
        for tag_idx, tag in enumerate(tags):
            if tag[0] == tag_prefix:
                return tag_idx
        raise Exception("Unexpected tag_prefix : {}".format(tag_prefix))


    def do_eval(self, step_idx):
        senli_logging.debug("ExEvaluator:do_eval ENTRY")
        log_out = open("eval_log{}.txt".format(self.log_idx), "w")
        self.log_idx += 1
        output = []
        for tag, eval_set in self.ex_eval_data:
            data_id_to_ex_scores: Dict[DataID, np.array] = self.get_ex_scores(eval_set.dataset)
            ap_list = []
            log_out.write("tag: {}".format(tag))
            tag_idx = tags.index(tag)
            for data_id, ex_scores in data_id_to_ex_scores.items():
                ex_scores_for_tag = ex_scores[tag_idx, :]
                tokenize_mapping = eval_set.tokenize_mapping_d[data_id]
                senli_logging.debug("tokenize_mapping {}".format(tokenize_mapping))
                sent1_scores: List[Tuple[int, float]] = []
                sent2_scores: List[Tuple[int, float]] = []
                per_space_token_scores: List[List[Tuple[int, float]]] = [sent1_scores, sent2_scores]
                senli_logging.debug("ex_scores.shape (batch, 3, 300) {}".format(ex_scores_for_tag.shape))
                n_lookup_fail = 0
                for idx, score in enumerate(ex_scores_for_tag):
                    try:
                        sent_idx, space_idx = tokenize_mapping[idx]
                        # It will raise exception for [CLS], [SEP], [PAD], because they are not tokens from texts
                        scores_per_sent: List[Tuple[int, float]] = per_space_token_scores[sent_idx]
                        scores_per_sent.append((space_idx, score))
                        n_lookup_fail = 0
                    except KeyError:
                        senli_logging.debug("KeyError at {}".format(idx))
                        n_lookup_fail += 1
                        # More than 5 consecutive KeyError should be PAD tokens
                        if n_lookup_fail > 5:
                            break

                ap_for_instance = []
                for sent_idx in score_valid_policy[tag]:
                    s = "label: {}\n".format(eval_set.label_d[data_id][sent_idx])
                    cur_sentence_scores = per_space_token_scores[sent_idx]
                    score_len = len(cur_sentence_scores)
                    assert score_len > 0
                    s += "score_len: {}\n".format(score_len)
                    scores = [s for idx, s in cur_sentence_scores]
                    score_str = " ".join(map(two_digit_float, scores))
                    s += "scores: {}\n".format(score_str)
                    cur_sentence_scores_merged = merge_and_sort_scores(cur_sentence_scores)
                    log_out.write(s)
                    ap = scores_to_ap(eval_set.label_d[data_id][sent_idx],
                                      cur_sentence_scores_merged)
                    ap_for_instance.append(ap)
                instance_ap = average(ap_for_instance)
                ap_list.append(instance_ap)
            mean_ap = average((ap_list))
            output.append(mean_ap)
            if self.summary_writer is not None:
                with self.summary_writer.as_default():
                    tf.summary.scalar('MAP:ex_{}'.format(tag), mean_ap, step=step_idx)

        senli_logging.debug("ExEvaluator:do_eval EXIT")
        return output

    def get_ex_scores(self, dataset) -> Dict[DataID, np.array]:
        senli_logging.debug("ExEvaluator:get_ex_scores ENTRY")
        dataset = dataset.batch(self.batch_size)
        data_id_to_ex_scores = {}
        for batch in dataset:
            x0, x1, data_id_batch = batch
            args = self.model, (x0, x1)
            per_replica = self.dist_strategy.run(get_ex_scores, args=args)
            ex_scores_batch = self.dist_strategy.gather(per_replica, axis=0)
            ex_scores_batch = ex_scores_batch.numpy()
            assert type(ex_scores_batch) == np.ndarray
            data_id_batch_np = data_id_batch.numpy()
            for data_id, ex_logits in zip(data_id_batch_np, ex_scores_batch):
                data_id_to_ex_scores[DataID(data_id)] = ex_logits
        senli_logging.debug("ExEvaluator:get_ex_scores EXIT")
        return data_id_to_ex_scores


def load_checkpoint(bert_cls_ex: BERT_CLS_EX, run_config: RunConfigEx):
    checkpoint_path = run_config.init_checkpoint
    if run_config.checkpoint_type == 'bert':
        bert.load_stock_weights(bert_cls_ex.l_bert, checkpoint_path)
        load_pooler(bert_cls_ex.pooler, checkpoint_path)
    elif run_config.checkpoint_type == "nli_saved_model":
        load_from_nli_saved_model(bert_cls_ex, run_config.init_checkpoint)


def init_log():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    return test_summary_writer