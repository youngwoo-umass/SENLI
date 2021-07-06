import sys

import tensorflow as tf

from cache import save_to_pickle
from data_generator.NLI.nli import get_modified_data_loader2
from data_generator.NLI.nli_info import tags
from data_generator.shared_setting import BertNLI
from explain.explain_model import ExplainPredictor
from explain.runner.predict_params import parser
from transformer import hyperparams
from transformer.nli_base import transformer_nli_pooled
from trainer.model_saver import load_model
from trainer.np_modules import get_batches_ex
from trainer.tf_train_module import init_session


def tuple_list_select(list_of_list, i):
    r = []
    for list_thing in list_of_list:
        r.append(list_thing[i])
    return r


def predict_for_view(hparam, nli_setting, data_loader, data_id, model_path, run_name, modeling_option, tags):
    print("predict_nli_ex")
    print("Modeling option: ", modeling_option)
    enc_payload, plain_payload = data_loader.load_plain_text(data_id)
    batches = get_batches_ex(enc_payload, hparam.batch_size, 3)

    task = transformer_nli_pooled(hparam, nli_setting.vocab_size)

    explain_predictor = ExplainPredictor(len(tags), task.model.get_sequence_output(), modeling_option)
    sess = init_session()
    sess.run(tf.global_variables_initializer())
    load_model(sess, model_path)


    out_entries = []
    for batch in batches:
        x0, x1, x2 = batch
        logits, ex_logits, = sess.run([task.logits, explain_predictor.get_score()],
                              feed_dict={
                                  task.x_list[0]: x0,
                                  task.x_list[1]: x1,
                                  task.x_list[2]: x2,
                              })

        for i in range(len(x0)):
            e = x0[i], logits[i], tuple_list_select(ex_logits, i)
            out_entries.append(e)

    save_to_pickle(out_entries, "save_view_{}_{}".format(run_name, data_id))


def run(args):
    hp = hyperparams.HPTrain()
    nli_setting = BertNLI()
    data_loader = get_modified_data_loader2(hp)

    predict_for_view(hp, nli_setting, data_loader,
                       args.data_id,
                       args.model_path,
                       args.run_name,
                       args.modeling_option,
                       tags,
                       )


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    run(args)