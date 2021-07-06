import argparse
import csv
import os
import re
import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set

from tf_logging import senli_logging

import bert
from bert.tokenization.bert_tokenization import FullTokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import tensorflow as tf

from dev.optimize import get_learning_rate_w_warmup, AdamWeightDecayOptimizer
from path_manager import bert_voca_path, mnli_dir, bert_model_folder

# model_folder = "/home/youngwookim/code/Chair/output/model/runs/uncased_L-12_H-768_A-12"
# model_folder = "c:\\work\\code\\Chair\\output\\model\\runs\\uncased_L-12_H-768_A-12"
label_list = ["entailment", "neutral", "contradiction", ]


def read_nli_data(file_name) -> List[Tuple[str, str, int]]:
    file_path = os.path.join(mnli_dir, file_name)
    data = []
    with open(file_path, encoding='utf-8', ) as csvFile:
        csv_reader = csv.reader(csvFile, delimiter="\t", quoting=csv.QUOTE_NONE)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            if line_count == 1:
                continue

            prem = row[8]
            hypo = row[9]
            label_str = row[-1]
            label = label_list.index(label_str)
            out_row = prem, hypo, label
            data.append(out_row)
    csvFile.close()
    senli_logging.info("{} datapoints from {}".format(len(data), file_name))
    return data


def encode_pair_inst_w_label(tokenizer: FullTokenizer, max_seq_length, paired: Tuple[str, str, int])\
        -> Tuple[List, List, int]:
    t1, t2, label = paired
    tokens1 = tokenizer.tokenize(t1)
    tokens2 = tokenizer.tokenize(t2)
    combined_tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
    seg1_len = 2 + len(tokens1)
    seg2_len = 1 + len(tokens2)
    token_ids = tokenizer.convert_tokens_to_ids(combined_tokens)
    segment_ids = [0] * seg1_len + [1] * seg2_len
    token_ids = pad_sequences([token_ids], max_seq_length)[0]
    segment_ids = pad_sequences([segment_ids], max_seq_length)[0]
    return token_ids, segment_ids, label


def get_nli_data(max_seq_length, name):
    text_data = read_nli_data(name)
    tokenizer = createTokenizer()

    def encode(inst: Tuple[str, str, int]):
        return encode_pair_inst_w_label(tokenizer, max_seq_length, inst)

    def gen():
        for t in text_data:
            yield encode(t)

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32),
            tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )
    return dataset


def createTokenizer():
    tokenizer = bert.bert_tokenization.FullTokenizer(bert_voca_path, do_lower_case=True)
    return tokenizer


class ModelConfig:
    max_seq_length = 300
    num_classes = 3


class RunConfig:
    batch_size = 16
    num_classes = 3
    train_step = 49875
    eval_every_n_step = 100
    save_every_n_step = 5000
    learning_rate = 1e-5
    model_save_path = "saved_model"


def get_run_config():
    parser = argparse.ArgumentParser(description='File should be stored in ')
    parser.add_argument("--model_save_path")
    args = parser.parse_args(sys.argv[1:])
    run_config = RunConfig()

    if args.model_save_path:
        run_config.model_save_path = args.model_save_path

    return run_config



class BERT_CLS:
    def __init__(self, bert_params, config: ModelConfig):
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        max_seq_len = config.max_seq_length
        num_classes = config.num_classes

        l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="segment_ids")
        seq_out = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
        first_token = seq_out[:, 0, :]
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
        pooled = pooler(first_token)
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(pooled)
        model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output, name="bert_model")
        self.model = model
        self.l_bert = l_bert
        self.pooler = pooler


# def get_bert_cls_model(bert_params, config: ModelConfig):
#     l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
#     max_seq_len = config.max_seq_length
#     num_classes = config.num_classes
#
#     l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
#     l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="segment_ids")
#     seq_out = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
#     first_token = seq_out[:, 0, :]
#     pooled = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")(first_token)
#     output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(pooled)
#     model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output, name="bert_model")
#     return model, l_bert


@tf.function
def train(model, item, loss_fn, optimizer):
    x1, x2, y = item
    with tf.GradientTape() as tape:
        prediction = model([x1, x2], training=True)
        loss = loss_fn(y, prediction)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


@tf.function
def eval_fn(model, item, loss_fn, dev_loss, dev_acc):
    x1, x2, y = item
    prediction = model([x1, x2], training=True)
    loss = loss_fn(y, prediction)
    dev_loss.update_state(loss)
    pred = tf.argmax(prediction, axis=1)
    dev_acc.update_state(y, pred)


@tf.function
def distributed_train_step(mirrored_strategy, train_step_fn, dist_inputs: Tuple, batch_size):
    per_replica_losses = mirrored_strategy.run(train_step_fn, args=dist_inputs)
    loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    new_global_step = global_step + 1
    global_step.assign(new_global_step)
    return loss


def get_optimizer(lr: float, num_train_steps: int):
    global_step = tf.compat.v1.train.get_or_create_global_step()
    learning_rate = get_learning_rate_w_warmup(global_step, lr, num_train_steps, 0)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.02,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    return optimizer


def load_pooler(pooler: tf.keras.layers.Dense, ckpt_path):
    """
    Use this method to load the weights from a pre-trained BERT checkpoint into a bert layer.

    :param pooler: a dense layer instance within a built keras model.
    :param ckpt_path: checkpoint path, i.e. `uncased_L-12_H-768_A-12/bert_model.ckpt` or `albert_base_zh/albert_model.ckpt`
    :return: list of weights with mismatched shapes. This can be used to extend
    the segment/token_type embeddings.
    """
    ckpt_reader = tf.train.load_checkpoint(ckpt_path)

    stock_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())

    loaded_weights = set()
    skip_count = 0
    weight_value_tuples = []
    bert_params = pooler.weights
    for ndx, (param) in enumerate(bert_params):
        name = param.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            stock_name = m.group(1)
        else:
            stock_name = name

        if ckpt_reader.has_tensor(stock_name):
            ckpt_value = ckpt_reader.get_tensor(stock_name)
            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            print("{} not found".format(stock_name))
            skip_count += 1
    keras.backend.batch_set_value(weight_value_tuples)


def log_worthy(step_idx):
    if step_idx < 100:
        return True
    elif step_idx % 10 == 0:
        return True
    return False


def main():
    debug_run = False
    bert_params = bert.params_from_pretrained_ckpt(bert_model_folder)
    config = ModelConfig()
    run_config = get_run_config()
    model_save_path: str = run_config.model_save_path
    mirrored_strategy = tf.distribute.MirroredStrategy()
    batch_size: int = run_config.batch_size

    with mirrored_strategy.scope():
        bert_cls = BERT_CLS(bert_params, config)
        model = bert_cls.model

        loss_fn_inner = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            per_example_loss = loss_fn_inner(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

    # optimizer = get_optimizer(run_config.learning_rate, run_config.train_step)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    optimizer = tf.optimizers.Adam(learning_rate=1e-5)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    dev_loss = tf.keras.metrics.Mean(name='dev_loss')
    dev_acc = tf.keras.metrics.Accuracy(name='accuracy', dtype=None)

    dev_dataset = get_nli_data(config.max_seq_length, "dev_matched.tsv")
    eval_batches = dev_dataset.batch(batch_size).take(10)

    if debug_run:
        train_dataset = get_nli_data(config.max_seq_length, "dev_matched.tsv")
    else:
        train_dataset = get_nli_data(config.max_seq_length, "train.tsv")

    train_dataset = train_dataset.repeat(4).batch(batch_size)
    train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
    train_itr = iter(train_dataset)
    bert_ckpt_file = os.path.join(bert_model_folder, "bert_model.ckpt")
    bert.load_stock_weights(bert_cls.l_bert, bert_ckpt_file)
    load_pooler(bert_cls.pooler, bert_ckpt_file)

    for step_idx in range(run_config.train_step):
        batch_item = next(train_itr)
        train_loss.reset_state()
        dev_loss.reset_state()
        args = model, batch_item, compute_loss, optimizer
        loss = distributed_train_step(mirrored_strategy, train, args, batch_size)
        train_loss.update_state(loss)
        if step_idx % run_config.eval_every_n_step == 0:
            dev_acc.reset_state()
            for e_batch in eval_batches:
                eval_fn(model, e_batch, compute_loss, dev_loss, dev_acc)
            senli_logging.info("step {0} train_loss={1:.2f} dev_loss={2:.2f} dev_acc={3:.2f}"
                               .format(step_idx, train_loss.result().numpy(),
                                       dev_loss.result().numpy(), dev_acc.result().numpy()))
        elif log_worthy(step_idx):
            senli_logging.info("step {0} train_loss={1:.2f}".format(step_idx, train_loss.result().numpy()))

        if step_idx % run_config.save_every_n_step == 0:
            model.save(model_save_path)
            senli_logging.info("Model saved at {}".format(model_save_path))

    senli_logging.info("Training completed")
    model.save(model_save_path)
    senli_logging.info("Model saved at {}".format(model_save_path))


if __name__ == "__main__":
    main()