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
    # Train the model here
    debug_run = False
    if debug_run:
        senli_logging.warning("DEBUGGING in use")
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

    ex_eval_data: List[Tuple[str, EvalSet]] = load_nli_ex_eval_encoded("dev", model_config.max_seq_length)
    summary_writer = init_log()
    model_save_path: str = run_config.model_save_path
    dist_strategy = tf.distribute.MirroredStrategy()
    batch_size: int = run_config.batch_size
    with dist_strategy.scope():
        bert_cls_ex = BERT_CLS_EX(bert_params, model_config, len(tags), training=True)
        model: keras.Model = bert_cls_ex.model

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

    optimizer = tf.optimizers.Adam(learning_rate=2e-5)
    eval_batches, train_dataset = load_nli_dataset(batch_size, debug_run, model_config.max_seq_length)
    eval_batches = distribute_dataset(dist_strategy, eval_batches)
    dist_train_dataset = distribute_dataset(dist_strategy, train_dataset)

    eval_object = EvalObject(model, eval_batches, dist_strategy, compute_loss)
    ex_evaluator = ExEvaluator(model, tags, ex_eval_data, batch_size, dist_strategy, summary_writer)
    train_itr = iter(dist_train_dataset)

    load_checkpoint(bert_cls_ex, run_config)
    senli_logging.info("START Training")
    for step_idx in range(run_config.train_step):
        f_do_cls_train = True
        f_do_ex_train = step_idx >= step_to_start_ex_train
        f_do_eval = step_idx % run_config.eval_every_n_step == 0
        f_do_eval_ex = f_do_eval and f_do_ex_train
        f_do_save = step_idx % run_config.save_every_n_step == 0 and not debug_run

        batch_item = next(train_itr)
        x1, x2, y = batch_item
        per_step_msg = "step {0}".format(step_idx)

        if f_do_eval_ex:
            ex_aps = ex_evaluator.do_eval(step_idx)
            score_str = " ".join(map("{0:.4f}".format, ex_aps))
            per_step_msg += " ex_MAP=" + score_str

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


if __name__ == "__main__":
    main()


