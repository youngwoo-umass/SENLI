import argparse
import sys

import numpy as np
import tensorflow as tf

from attribution.baselines import explain_by_seq_deletion, explain_by_random, IdfScorer, explain_by_deletion
from attribution.eval import predict_translate
from attribution.lime import explain_by_lime
from cache import save_to_pickle
from data_generator.NLI.nli import get_modified_data_loader2
from data_generator.shared_setting import BertNLI
from explain.nli_gradient_baselines import nli_attribution_predict
from explain.train_nli import get_nli_data
from transformer import hyperparams
from transformer.nli_base import transformer_nli_pooled
from trainer.model_saver import load_model
from trainer.np_modules import get_batches_ex
from trainer.tf_train_module import init_session

nli_ex_prediction_parser = argparse.ArgumentParser(description='')
nli_ex_prediction_parser.add_argument("--tag", help="Your input file.")
nli_ex_prediction_parser.add_argument("--model_path", help="Your model path.")
nli_ex_prediction_parser.add_argument("--method_name", )
nli_ex_prediction_parser.add_argument("--data_id")
nli_ex_prediction_parser.add_argument("--sub_range")

load_nli_data = NotImplemented

def nli_baseline_predict(hparam, nli_setting, data_loader,
                         explain_tag, method_name, data_id, sub_range, model_path):
    enc_payload, plain_payload = data_loader.load_plain_text(data_id)
    assert enc_payload is not None
    assert plain_payload    is not None

    name_format = "pred_{}_" + data_id
    if sub_range is not None:
        st, ed = [int(t) for t in sub_range.split(",")]
        enc_payload = enc_payload[st:ed]
        plain_payload = plain_payload[st:ed]
        name_format = "pred_{}_" + data_id + "__{}_{}".format(st, ed)
        print(name_format)
    task = transformer_nli_pooled(hparam, nli_setting.vocab_size)
    sout = tf.nn.softmax(task.logits, axis=-1)
    sess = init_session()
    sess.run(tf.global_variables_initializer())
    load_model(sess, model_path)

    def forward_run(inputs):
        batches = get_batches_ex(inputs, hparam.batch_size, 3)
        logit_list = []
        for batch in batches:
            x0, x1, x2 = batch
            soft_out, = sess.run([sout, ],
                                  feed_dict={
                                      task.x_list[0]: x0,
                                      task.x_list[1]: x1,
                                      task.x_list[2]: x2,
                                  })
            logit_list.append(soft_out)
        return np.concatenate(logit_list)

    # train_batches, dev_batches = self.load_nli_data(data_loader)
    def idf_explain(enc_payload, explain_tag, forward_run):
        train_batches, dev_batches = get_nli_data(hparam, nli_setting)
        idf_scorer = IdfScorer(train_batches)
        return idf_scorer.explain(enc_payload, explain_tag, forward_run)

    todo_list = [
        ('deletion_seq', explain_by_seq_deletion),
        ('random', explain_by_random),
        ('idf', idf_explain),
        ('deletion', explain_by_deletion),
        ('LIME', explain_by_lime),
    ]
    method_dict = dict(todo_list)
    method = method_dict[method_name]
    explains = method(enc_payload, explain_tag, forward_run)
    pred_list = predict_translate(explains, data_loader, enc_payload, plain_payload)
    save_to_pickle(pred_list, name_format.format(method_name))


def run(args):
    hp = hyperparams.HPTrain()
    nli_setting = BertNLI()
    data_loader = get_modified_data_loader2(hp)

    if args.method_name in ['deletion_seq', "random", 'idf', 'deletion', 'LIME']:
        predictor = nli_baseline_predict
    elif args.method_name in  [ "elrp", "deeplift", "saliency","grad*input", "intgrad", ]:
        predictor = nli_attribution_predict
    else:
        raise Exception("method_name={} is not in the known method list.".format(args.method_name))

    predictor(hp, nli_setting, data_loader,
                         args.tag,
                         args.method_name,
                         args.data_id,
                         args.sub_range,
                         args.model_path
                         )

if __name__ == "__main__":
    args = nli_ex_prediction_parser.parse_args(sys.argv[1:])
    run(args)