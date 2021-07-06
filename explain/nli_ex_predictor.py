import numpy as np
import tensorflow as tf

from data_generator.NLI import nli_info
from explain.explain_model import ExplainPredictor
from transformer.transformer_cls import transformer_pooled
from trainer.model_saver import load_model
from trainer.np_modules import get_batches_ex
from trainer.tf_train_module import init_session


class NLIExPredictor:
    def __init__(self, hparam, nli_setting, model_path, modeling_option, tags_list=nli_info.tags):
        self.num_tags = len(tags_list)
        self.tags = tags_list
        self.define_graph(hparam, nli_setting, modeling_option)
        self.sess = init_session()
        self.sess.run(tf.global_variables_initializer())
        self.batch_size = hparam.batch_size
        load_model(self.sess, model_path)

    def define_graph(self, hparam, train_config, modeling_option):
        self.task = transformer_pooled(hparam, train_config.vocab_size, False)
        self.sout = tf.nn.softmax(self.task.logits, axis=-1)
        self.explain_predictor = ExplainPredictor(self.num_tags, self.task.model.get_sequence_output(), modeling_option)

    def predict_ex(self, explain_tag, batches):
        tag_idx = self.tags.index(explain_tag)
        ex_logits_list = []
        for batch in batches:
            x0, x1, x2 = batch
            ex_logits, = self.sess.run([self.explain_predictor.get_score()[tag_idx]],
                                  feed_dict={
                                      self.task.x_list[0]: x0,
                                      self.task.x_list[1]: x1,
                                      self.task.x_list[2]: x2,
                                  })
            ex_logits_list.append(ex_logits)
        ex_logits = np.concatenate(ex_logits_list)
        return ex_logits

    def predict_both(self, explain_tag, batches):
        tag_idx = self.tags.index(explain_tag)
        ex_logits_list = []
        sout_list = []
        for batch in batches:
            x0, x1, x2 = batch
            sout, ex_logits, = self.sess.run([self.sout, self.explain_predictor.get_score()[tag_idx]],
                                  feed_dict={
                                      self.task.x_list[0]: x0,
                                      self.task.x_list[1]: x1,
                                      self.task.x_list[2]: x2,
                                  })
            ex_logits_list.append(ex_logits)
            sout_list.append(sout)
        ex_logits = np.concatenate(ex_logits_list, axis=0)
        sout = np.concatenate(sout_list, axis=0)
        return sout, ex_logits

    def predict_ex_from_insts(self, explain_tag, insts):
        batches = get_batches_ex(insts, self.batch_size, 3)
        return self.predict_ex(explain_tag, batches)

    def predict_both_from_insts(self, explain_tag, insts):
        batches = get_batches_ex(insts, self.batch_size, 3)
        r = self.predict_both(explain_tag, batches)
        return r

    def forward_run(self, inputs):
        batches = get_batches_ex(inputs, self.batch_size, 3)
        logit_list = []
        task = self.task
        for batch in batches:
            x0, x1, x2 = batch
            soft_out, = self.sess.run([self.sout, ],
                                  feed_dict={
                                      task.x_list[0]: x0,
                                      task.x_list[1]: x1,
                                      task.x_list[2]: x2,
                                  })
            logit_list.append(soft_out)
        return np.concatenate(logit_list)