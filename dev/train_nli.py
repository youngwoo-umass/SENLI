import os
from typing import Tuple

from dev.nli_common import get_nli_data
from dev.bert_common import BERT_CLS, eval_fn, get_run_config, ModelConfig, load_pooler, is_interesting_step
from tf_logging import senli_logging

import bert
import tensorflow as tf

from dev.optimize import get_learning_rate_w_warmup, AdamWeightDecayOptimizer
from path_manager import bert_model_folder

# model_folder = "/home/youngwookim/code/Chair/output/model/runs/uncased_L-12_H-768_A-12"
# model_folder = "c:\\work\\code\\Chair\\output\\model\\runs\\uncased_L-12_H-768_A-12"


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
def distributed_train_step(mirrored_strategy, train_step_fn, dist_inputs: Tuple):
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
        loss = distributed_train_step(mirrored_strategy, train, args)
        train_loss.update_state(loss)
        if step_idx % run_config.eval_every_n_step == 0:
            dev_acc.reset_state()
            for e_batch in eval_batches:
                eval_fn(model, e_batch, compute_loss, dev_loss, dev_acc)
            senli_logging.info("step {0} train_loss={1:.2f} dev_loss={2:.2f} dev_acc={3:.2f}"
                               .format(step_idx, train_loss.result().numpy(),
                                       dev_loss.result().numpy(), dev_acc.result().numpy()))
        elif is_interesting_step(step_idx):
            senli_logging.info("step {0} train_loss={1:.2f}".format(step_idx, train_loss.result().numpy()))

        if step_idx % run_config.save_every_n_step == 0:
            model.save(model_save_path)
            senli_logging.info("Model saved at {}".format(model_save_path))

    senli_logging.info("Training completed")
    model.save(model_save_path)
    senli_logging.info("Model saved at {}".format(model_save_path))


if __name__ == "__main__":
    main()
