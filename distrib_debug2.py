import csv
import os
import time
from typing import List, Iterable, Callable, Dict, Tuple, Set
from tf_logging import senli_logging

import bert
from bert.tokenization.bert_tokenization import FullTokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import tensorflow as tf

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


def encode_pair_inst_w_label(tokenizer: FullTokenizer, max_seq_length, paired: Tuple[str, str, int]) \
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


def get_bert_cls_model(bert_params, config: ModelConfig):
    l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

    max_seq_len = config.max_seq_length
    num_classes = config.num_classes

    l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    # l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="segment_ids")
    l_token_type_ids = tf.zeros_like(l_input_ids)
    seq_out = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
    first_token = seq_out[:, 0, :]
    pooled = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh)(first_token)
    output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(pooled)
    model = keras.Model(inputs=[l_input_ids], outputs=output, name="bert_model")
    return model


mode = "bert"


@tf.function
def train(model, item, loss_fn, optimizer):
    if mode == "sample":
        x1, y = item
    else:
        x1, x2, y = item

    with tf.GradientTape() as tape:
        predictions = model([x1], training=True)
        # predictions = model(x1, training=True)
        loss = loss_fn(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


@tf.function
def distributed_train_step(mirrored_strategy, train_step_fn, dist_inputs: Tuple, batch_size):
    per_replica_losses = mirrored_strategy.run(train_step_fn, args=dist_inputs)
    loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    return loss


def main():
    bert_params = bert.params_from_pretrained_ckpt(bert_model_folder)
    config = ModelConfig()
    run_config = RunConfig()
    mirrored_strategy = tf.distribute.MirroredStrategy()
    batch_size = run_config.batch_size

    with mirrored_strategy.scope():
        if mode == "sample":
            model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
        else:
            model = get_bert_cls_model(bert_params, config)
        loss_fn_inner = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            per_example_loss = loss_fn_inner(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

    optimizer = tf.optimizers.Adam(learning_rate=1e-5)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    dev_loss = tf.keras.metrics.Mean(name='dev_loss')
    if mode == "sample":
        dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(
            4)
    else:
        dataset = get_nli_data(config.max_seq_length, "dev_matched.tsv")
        dataset = dataset.batch(16)

    dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
    train_itr = iter(dist_dataset)

    st = None
    for step_idx in range(100):
        batch_item = next(train_itr)
        train_loss.reset_state()
        dev_loss.reset_state()
        args = model, batch_item, compute_loss, optimizer
        loss = distributed_train_step(mirrored_strategy, train, args, batch_size)
        print(loss)
        if st is None:
            st = time.time()
        train_loss.update_state(loss)
    ed = time.time()
    print("time elapsed", ed-st)


if __name__ == "__main__":
    main()
