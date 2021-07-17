import os
import re
from typing import Tuple

from tqdm import tqdm

from dev.nli_common import get_nli_data
from dev.bert_common import BERT_CLS, eval_fn, get_run_config, ModelConfig
from tf_logging import senli_logging

import bert
from tensorflow import keras
import tensorflow as tf

from path_manager import bert_model_folder


def main():
    debug_run = False
    bert_params = bert.params_from_pretrained_ckpt(bert_model_folder)
    config = ModelConfig()
    run_config = get_run_config()
    model_save_path: str = run_config.model_save_path
    dist_strategy = tf.distribute.MirroredStrategy()
    batch_size: int = run_config.batch_size

    with dist_strategy.scope():
        bert_cls = BERT_CLS(bert_params, config)
        model = bert_cls.model

        loss_fn_inner = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            per_example_loss = loss_fn_inner(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

    dev_loss = tf.keras.metrics.Mean(name='dev_loss')
    dev_acc = tf.keras.metrics.Accuracy(name='accuracy', dtype=None)

    dev_dataset = get_nli_data(config.max_seq_length, "dev_matched.tsv")
    eval_batches = dev_dataset.batch(batch_size)
    if debug_run:
        eval_batches = eval_batches.take(100)
    model.load_weights(model_save_path)
    dev_loss.reset_state()

    for batch in tqdm(eval_batches):
        eval_fn(model, batch, compute_loss, dev_loss, dev_acc)

    print("Dev loss: {}".format(dev_loss.result().numpy()))
    print("Dev acc : {}".format(dev_acc.result().numpy()))


if __name__ == "__main__":
    main()
