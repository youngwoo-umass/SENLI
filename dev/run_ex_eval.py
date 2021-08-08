import sys
from typing import List

from dev.ex_eval import load_nli_ex_eval_encoded, EvalSet
from dev.train_nli_ex import RunConfigEx, get_run_config, train_cls, distributed_train_step, init_ex_trainer, \
    get_cls_logits, EvalObject, ExEvaluator, load_checkpoint, init_log
from senli_log import senli_logging

from typing import Tuple
import numpy as np

from dev.nli_common import tags, load_nli_dataset
from dev.bert_common import ModelConfig, BERT_CLS_EX, is_interesting_step
import bert
from tensorflow import keras
import tensorflow as tf
from dev.tf_helper import distribute_dataset

from path_manager import bert_model_folder

# This defines graph


def main():
    if len(sys.argv) < 2:
        print("model load path should be the first argument")
    # Train the model here
    bert_params = bert.params_from_pretrained_ckpt(bert_model_folder)
    model_config = ModelConfig()
    run_config = RunConfigEx()
    ex_eval_data: List[Tuple[str, EvalSet]] = load_nli_ex_eval_encoded("test", model_config.max_seq_length)
    dist_strategy = tf.distribute.MirroredStrategy()
    with dist_strategy.scope():
        bert_cls_ex = BERT_CLS_EX(bert_params, model_config, len(tags), training=False)
        bert_cls_ex.model = tf.keras.models.load_model(sys.argv[1])

        batch_size: int = run_config.batch_size

    ex_evaluator = ExEvaluator(bert_cls_ex.model, tags, ex_eval_data, batch_size, dist_strategy, None)
    ex_aps = ex_evaluator.do_eval(0)

    for tag, score in zip(tags, ex_aps):
        print("{}\t{}".format(tag, score))


if __name__ == "__main__":
    main()