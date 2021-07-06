import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from cpath import output_path
from explain.runner.eval_accuracy import get_eval_params
from transformer.nli_base import transformer_nli_pooled
from trainer.tf_train_module import init_session


def predict(hparam, nli_setting, run_name, dev_batches, model_path, load_fn):
    print("predict :", run_name)
    task = transformer_nli_pooled(hparam, nli_setting.vocab_size, False)
    sess = init_session()
    sess.run(tf.global_variables_initializer())
    if model_path is not None:
        load_fn(sess, model_path)

    def batch2feed_dict(batch):
        x0, x1, x2, y  = batch
        feed_dict = {
            task.x_list[0]: x0,
            task.x_list[1]: x1,
            task.x_list[2]: x2,
            task.y: y,
        }
        return feed_dict

    label_list = []
    pred_list = []
    for batch in dev_batches:
        logits, = sess.run([task.logits, ],
                           feed_dict=batch2feed_dict(batch)
                           )

        _, _, _, y = batch
        label_list.append(y)
        pred_list.append(np.argmax(logits, axis=-1))

    pred_list = np.concatenate(pred_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)
    if len(pred_list) != len(label_list):
        print("WARNING , data size is different : ", len(pred_list), len(label_list))

    output = pred_list, label_list

    out_path = os.path.join(output_path, "nli_pred_" + run_name.replace("/", "_"))
    pickle.dump(output, open(out_path, "wb"))


def predict_nli(model_path, load_type):
    dev_batches, hp, load_fn, nli_setting, run_name = get_eval_params(load_type, model_path)
    predict(hp, nli_setting, run_name, dev_batches, model_path, load_fn)


if __name__  == "__main__":
    predict_nli(sys.argv[1], sys.argv[2])