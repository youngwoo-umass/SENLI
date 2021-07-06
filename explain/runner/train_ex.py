import os
import pickle
import sys
from functools import partial

import numpy as np
import tensorflow as tf

import data_generator.NLI.nli_info
from attribution.eval import eval_explain
from data_generator.NLI.nli import get_modified_data_loader
from data_generator.shared_setting import BertNLI
from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.ex_train_modules import NLIExTrainConfig, get_informative_fn_by_name
from explain.explain_model import CorrelationModeling, CrossEntropyModeling
from explain.explain_trainer import ExplainTrainerM
from explain.nli_common import train_classification_factory, save_fn_factory, valid_fn_factory
from explain.runner.nli_ex_param import ex_arg_parser
from explain.setups import init_fn_generic
from explain.train_nli import get_nli_data
from transformer import hyperparams
from transformer.nli_base import transformer_nli_pooled
from transformer.transformer_cls import transformer_pooled
from tf_util.tf_logging import tf_logging, set_level_debug
from trainer.model_saver import setup_summary_writer
from trainer.multi_gpu_support import get_multiple_models, get_avg_loss, \
    get_avg_tensors_from_models, get_train_op, get_batch2feed_dict_for_multi_gpu, get_concat_tensors_from_models, \
    get_other_train_op_multi_gpu, get_concat_tensors_list_from_models
from trainer.np_modules import get_batches_ex
from trainer.tf_module import step_runner
from trainer.tf_train_module import init_session, get_train_op2, get_train_op_wo_gstep_update2


def train_self_explain(hparam, train_config, save_dir,
                       data, data_loader,
                       tags, modeling_option, init_fn, tag_informative_fn,
                       ):
    print("train_self_explain")
    max_steps = train_config.max_steps
    num_gpu = train_config.num_gpu
    train_batches, dev_batches = data

    ex_modeling_class = {
        'ce': CrossEntropyModeling,
        'co': CorrelationModeling
    }[modeling_option]
    lr_factor = 0.3

    def build_model():
        main_model = transformer_pooled(hparam, train_config.vocab_size)
        ex_model = ex_modeling_class(main_model.model.sequence_output, hparam.seq_max, len(tags),
                                     main_model.batch2feed_dict)
        return main_model, ex_model

    if num_gpu == 1:
        print("Using single GPU")
        main_model, ex_model = build_model()
        loss_tensor = main_model.loss
        acc_tensor = main_model.acc
        with tf.variable_scope("optimizer"):
            train_cls = get_train_op2(main_model.loss, hparam.lr, "adam", max_steps)
        batch2feed_dict = main_model.batch2feed_dict
        logits = main_model.logits

        ex_score_tensor = ex_model.get_scores()
        ex_loss_tensor = ex_model.get_loss()
        ex_per_tag_loss = ex_model.get_losses()
        ex_batch_feed2dict = ex_model.batch2feed_dict
        with tf.variable_scope("explain_optimizer"):
            train_ex_op = get_train_op_wo_gstep_update2(ex_loss_tensor, hparam.lr * lr_factor, "adam2", max_steps)
    else:
        main_models, ex_models = zip(*get_multiple_models(build_model, num_gpu))
        loss_tensor = get_avg_loss(main_models)
        acc_tensor = get_avg_tensors_from_models(main_models, lambda model: model.acc)
        with tf.variable_scope("optimizer"):
            train_cls = get_train_op([m.loss for m in main_models], hparam.lr, max_steps)
        batch2feed_dict = get_batch2feed_dict_for_multi_gpu(main_models)
        logits = get_concat_tensors_from_models(main_models, lambda model:model.logits)
        def get_loss_tensor(model):
            t = tf.expand_dims(tf.stack(model.get_losses()), 0)
            return t
        ex_per_tag_loss = tf.reduce_mean(get_concat_tensors_from_models(ex_models, get_loss_tensor), axis=0)

        ex_score_tensor = get_concat_tensors_list_from_models(ex_models, lambda model:model.get_scores())
        print(ex_score_tensor)
        ex_loss_tensor = get_avg_tensors_from_models(ex_models, ex_modeling_class.get_loss)
        ex_batch_feed2dict = get_batch2feed_dict_for_multi_gpu(ex_models)
        with tf.variable_scope("explain_optimizer"):
            train_ex_op = get_other_train_op_multi_gpu(([m.get_loss() for m in ex_models]), hparam.lr * lr_factor, max_steps)

    global_step = tf.train.get_or_create_global_step()

    if data_loader is not None:
        explain_dev_data_list = {tag: data_loader.get_dev_explain(tag) for tag in tags}

    run_name = os.path.basename(save_dir)
    train_writer, test_writer = setup_summary_writer(run_name)

    information_fn_list = list([partial(tag_informative_fn, t) for t in tags])

    def forward_run(batch):
        result, = sess.run([logits], feed_dict=batch2feed_dict(batch))
        return result

    explain_trainer = ExplainTrainerM(information_fn_list,
                                      len(tags),
                                      action_to_label=ex_modeling_class.action_to_label,
                                      get_null_label=ex_modeling_class.get_null_label,
                                      forward_run=forward_run,
                                      batch_size=hparam.batch_size,
                                      num_deletion=train_config.num_deletion,
                                      g_val=train_config.g_val,
                                      drop_thres=train_config.drop_thres
                                      )

    sess = init_session()
    sess.run(tf.global_variables_initializer())
    init_fn(sess)

    def eval_tag():
        if data_loader is None:
            return
        print("Eval")
        for label_idx, tag in enumerate(tags):
            print(tag)
            enc_explain_dev, explain_dev = explain_dev_data_list[tag]
            batches = get_batches_ex(enc_explain_dev, hparam.batch_size, 3)
            try:
                ex_logit_list = []
                for batch in batches:
                    ex_logits,  = sess.run([ex_score_tensor[label_idx],], feed_dict=batch2feed_dict(batch))
                    print(ex_logits.shape)
                    ex_logit_list.append(ex_logits)

                ex_logit_list = np.concatenate(ex_logit_list, axis=0)
                print(ex_logit_list.shape)
                assert len(ex_logit_list) == len(explain_dev)

                scores = eval_explain(ex_logit_list, data_loader, tag)

                for metric in scores.keys():
                    print("{}\t{}".format(metric, scores[metric]))

                p_at_1, MAP_score = scores["P@1"], scores["MAP"]
                summary = tf.Summary()
                summary.value.add(tag='{}_P@1'.format(tag), simple_value=p_at_1)
                summary.value.add(tag='{}_MAP'.format(tag), simple_value=MAP_score)
                train_writer.add_summary(summary, fetch_global_step())
                train_writer.flush()
            except ValueError as e:
                print(e)
                for ex_logits in ex_logit_list:
                    print(ex_logits.shape)

    # make explain train_op does not increase global step

    def train_explain(batch, step_i):
        def commit_ex_train(batch):
            if train_config.save_train_payload:
                save_payload(batch, step_i)
            fd = ex_batch_feed2dict(batch)
            ex_loss, _ =  sess.run([ex_per_tag_loss, train_ex_op], feed_dict=fd)
            return ex_loss
        summary = explain_trainer.train_batch(batch, commit_ex_train)
        train_writer.add_summary(summary, fetch_global_step())

    def fetch_global_step():
        step, = sess.run([global_step])
        return step

    train_classification = partial(train_classification_factory, sess, loss_tensor, acc_tensor, train_cls, batch2feed_dict)
    eval_acc = partial(valid_fn_factory, sess, dev_batches[:20], loss_tensor, acc_tensor, global_step, batch2feed_dict)

    save_fn = partial(save_fn_factory, sess, save_dir, global_step)
    init_step,  = sess.run([global_step])

    def train_fn(batch, step_i):
        step_before_cls = fetch_global_step()
        loss_val, acc = train_classification(batch, step_i)
        summary = tf.Summary()
        summary.value.add(tag='acc', simple_value=acc)
        summary.value.add(tag='loss', simple_value=loss_val)
        train_writer.add_summary(summary, fetch_global_step())
        train_writer.flush()
        tf_logging.debug("{}".format(step_i))

        step_after_cls = fetch_global_step()

        assert step_after_cls == step_before_cls + 1
        train_explain(batch, step_i)
        step_after_ex = fetch_global_step()
        assert step_after_cls == step_after_ex
        return loss_val, acc

    def valid_fn():
        eval_acc()
        eval_tag()
    print("Initialize step to {}".format(init_step))
    print("{} train batches".format(len(train_batches)))
    valid_freq = 1000
    save_interval = 5000
    loss, _ = step_runner(train_batches, train_fn, init_step,
                              valid_fn, valid_freq,
                              save_fn, save_interval, max_steps)
    return save_fn()


def train_from(start_model_path, start_type, save_dir,
               modeling_option, tags, info_fn_name, num_deletion,
               g_val=0.5,
               drop_thres=0.3,
               num_gpu=1):

    num_deletion = int(num_deletion)
    num_gpu = int(num_gpu)
    tf_logging.info("train_from : nli_ex")
    data, data_loader, hp, informative_fn, init_fn, train_config\
        = get_params(start_model_path, start_type, info_fn_name, num_gpu)

    train_config.num_deletion = num_deletion
    train_config.g_val = float(g_val)
    train_config.drop_thres = float(drop_thres)

    train_self_explain(hp, train_config, save_dir,
                       data, data_loader, tags, modeling_option,
                       init_fn, informative_fn)


def get_params(start_model_path, start_type, info_fn_name, num_gpu):
    hp = hyperparams.HPTrain()
    nli_setting = BertNLI()
    set_level_debug()
    train_config = NLIExTrainConfig()
    train_config.num_gpu = num_gpu
    train_config.save_train_payload = True

    tokenizer = get_tokenizer()
    tf_logging.info("Intializing dataloader")
    data_loader = get_modified_data_loader(tokenizer, hp.seq_max)
    tf_logging.info("loading batches")
    data = get_nli_data(hp, nli_setting)

    def init_fn(sess):
        start_type_generic = {
            'nli': 'cls',
            'nli_ex': 'cls_ex',
            'bert': 'bert',
            'cold': 'cold'
        }[start_type]
        return init_fn_generic(sess, start_type_generic, start_model_path)

    informative_fn = get_informative_fn_by_name(info_fn_name)
    return data, data_loader, hp, informative_fn, init_fn, train_config


if __name__  == "__main__":
    args = ex_arg_parser.parse_args(sys.argv[1:])
    train_from(args.start_model_path,
               args.start_type,
               args.save_dir,
               args.modeling_option,
               data_generator.NLI.nli_info.tags,
               args.info_fn,
               args.num_deletion,
               args.g_val,
               args.drop_thres,
               args.num_gpu)
