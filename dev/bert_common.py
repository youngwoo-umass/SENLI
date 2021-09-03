import argparse
import re
import sys
from typing import Tuple, List

import bert
import tensorflow as tf

from bert.tokenization.bert_tokenization import FullTokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from path_manager import bert_voca_path


class RunConfig:
    batch_size = 16
    num_classes = 3
    train_step = 49875
    eval_every_n_step = 100
    save_every_n_step = 5000
    learning_rate = 1e-5
    model_save_path = "saved_model"
    init_checkpoint = ""


class ModelConfig:
    max_seq_length = 300
    num_classes = 3


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
    token_ids = pad_sequences([token_ids], max_seq_length, padding='post')[0]
    segment_ids = pad_sequences([segment_ids], max_seq_length, padding='post')[0]
    return token_ids, segment_ids, label


class BERT_CLS:
    def __init__(self, bert_params, config: ModelConfig):
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        max_seq_len = config.max_seq_length
        num_classes = config.num_classes

        l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="segment_ids")
        seq_out = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
        self.seq_out = seq_out
        first_token = seq_out[:, 0, :]
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
        pooled = pooler(first_token)
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(pooled)
        model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert = l_bert
        self.pooler = pooler


class BERT_CLS_EX:
    def __init__(self, bert_params, config: ModelConfig, num_tags, training):
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        max_seq_len = config.max_seq_length
        num_classes = config.num_classes

        l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="segment_ids")
        seq_out = l_bert([l_input_ids, l_token_type_ids], training=training)  # [batch_size, max_seq_len, hidden_size]
        self.seq_out = seq_out
        first_token = seq_out[:, 0, :]
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
        pooled = pooler(first_token)
        cls_logits = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(pooled)

        self.l_bert = l_bert
        self.pooler = pooler
        ex_logits_list = []
        for tag_idx in range(num_tags):
            ex_logits = tf.keras.layers.Dense(2, name="ex_{}".format(tag_idx))(seq_out)
            ex_logits_list.append(ex_logits)

        output = [cls_logits, ex_logits_list]
        self.cls_logits = cls_logits
        self.ex_logits_list = ex_logits_list
        model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output,
                            name="bert_model",
                            )

        self.model: keras.Model = model


@tf.function
def eval_fn(model, item, loss_fn, dev_loss, dev_acc):
    x1, x2, y = item
    prediction, _ = model([x1, x2], training=False)
    loss = loss_fn(y, prediction)
    dev_loss.update_state(loss)
    pred = tf.argmax(prediction, axis=1)
    dev_acc.update_state(y, pred)


def createTokenizer():
    tokenizer = bert.bert_tokenization.FullTokenizer(bert_voca_path, do_lower_case=True)
    return tokenizer


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


def is_interesting_step(step_idx):
    if step_idx < 100:
        return True
    elif step_idx % 10 == 0:
        return True
    return False