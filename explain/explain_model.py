import abc

import numpy as np
import tensorflow as tf

from tf_v2_support import placeholder
from trainer import tf_module


class ExplainPredictor:
    def __init__(self, num_tags, sequence_output, modeling_option="default"):
        self.num_tags = num_tags
        self.sequence_output = sequence_output
        if modeling_option == "co":
            self.modeling_co()
        else:
            self.modeling_ce()

    def modeling_ce(self):
        self.ex_probs = []
        self.ex_logits = []

        with tf.variable_scope("explain"):
            for tag_idx in range(self.num_tags):
                with tf.variable_scope("ex_{}".format(tag_idx)):
                    ex_logits = tf.layers.dense(self.sequence_output, 2, name="ex_{}".format(tag_idx))
                    ex_prob = tf.nn.softmax(ex_logits)

                self.ex_logits.append(ex_logits)
                self.ex_probs.append(ex_prob[:, :, 1])

    def modeling_co(self):
        self.ex_probs = []
        self.ex_logits = []

        with tf.variable_scope("explain"):
            for tag_idx in range(self.num_tags):
                with tf.variable_scope("ex_{}".format(tag_idx)):
                    ex_logits = tf.layers.dense(self.sequence_output, 1, name="ex_{}".format(tag_idx))
                    ex_logits = tf.squeeze(ex_logits, axis=2)
                self.ex_logits.append(ex_logits)
                self.ex_probs.append(ex_logits)

    def get_score(self):
        return self.ex_probs


class ExplainModelingInterface(abc.ABC):
    @abc.abstractmethod
    def get_losses(self):
        pass

    @abc.abstractmethod
    def get_per_inst_losses(self):
        pass

    @abc.abstractmethod
    def batch2feed_dict(self, labels):
        pass

    @abc.abstractmethod
    def get_scores(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def action_to_label(good_action):
        pass


class ExplainModeling(ExplainModelingInterface):
    def __init__(self, sequence_output, seq_max, num_tags, origin_batch2feed_dict):
        self.tag_info = []
        with tf.variable_scope("explain"):
            for tag_idx in range(num_tags):
                self.tag_info.append(self.model_tag(sequence_output, seq_max, "ex_{}".format(tag_idx)))
        self.origin_batch2feed_dict = origin_batch2feed_dict

    @abc.abstractmethod
    def model_tag(self, sequence_output, seq_max, var_name):
        pass

    def feed_ex_batch(self, labels):
        feed_dict = {}
        for tag_idx, info in enumerate(self.tag_info):
            feed_dict[info['labels']] = labels[tag_idx * 2]
            feed_dict[info['mask']] = labels[tag_idx * 2 + 1]
        return feed_dict

    def get_losses(self):
        return [info['loss'] for info in self.tag_info]

    # return : List[ [300], [300], [300] ]
    def get_scores(self):
        return [info['score'] for info in self.tag_info]

    def get_per_inst_losses(self):
        return [info['losses'] for info in self.tag_info]
        pass

    def get_loss(self):
        return tf.reduce_mean(self.get_losses())

    def batch2feed_dict(self, batch):
        x0, x1, x2, y = batch[:4]
        labels = batch[4:]
        feed_dict = self.origin_batch2feed_dict((x0, x1, x2, y))
        feed_dict.update(self.feed_ex_batch(labels))
        return feed_dict

    def get_ex_scores(self, label_idx):
        return self.get_scores()[label_idx]


class CorrelationModeling(ExplainModeling):
    def __init__(self, sequence_output, seq_max, num_tags, origin_batch2feed_dict):
        super(CorrelationModeling, self).__init__(sequence_output, seq_max, num_tags, origin_batch2feed_dict)

    def model_tag(self, sequence_output, seq_max, var_name):
        ex_labels = placeholder(tf.float32, [None, seq_max])
        valid_mask = placeholder(tf.float32, [None, 1])
        with tf.variable_scope(var_name):
            ex_logits = tf.layers.dense(sequence_output, 1, name=var_name)
            ex_logits = tf.reshape(ex_logits, [-1, seq_max])
            labels_ = tf.cast(tf.greater(ex_labels, 0), tf.float32)
            losses = tf_module.correlation_coefficient_loss(ex_logits, -labels_)
            losses = valid_mask * losses
        loss = tf.reduce_mean(losses)
        score = ex_logits

        return {
            'labels': ex_labels,
            'mask':valid_mask,
            'ex_logits': ex_logits,
            'score': score,
            'losses':losses,
            'loss': loss
        }

    @staticmethod
    def action_to_label(good_action):
        pos_reward_indice = np.int_(good_action)
        loss_mask = -pos_reward_indice + np.ones_like(pos_reward_indice) * 0.1
        return loss_mask, [1]

    @staticmethod
    def get_null_label(any_action):
        return np.int_(any_action) , [0]


class CrossEntropyModeling(ExplainModeling):
    def __init__(self, sequence_output, seq_max, num_tags, origin_batch2feed_dict):
        super(CrossEntropyModeling, self).__init__(sequence_output, seq_max, num_tags, origin_batch2feed_dict)

    def model_tag(self, sequence_output, seq_max, var_name):
        ex_label = placeholder(tf.int32, [None, seq_max])
        valid_mask = placeholder(tf.float32, [None, 1])
        with tf.variable_scope(var_name):
            ex_logits = tf.layers.dense(sequence_output, 2, name=var_name)
            ex_prob = tf.nn.softmax(ex_logits)[:, :, 1]
            losses = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(ex_label, 2), logits=ex_logits)
            losses = valid_mask * losses
            loss = tf.reduce_mean(losses)

        return {
            'labels': ex_label,
            'mask': valid_mask,
            'ex_logits': ex_logits,
            'score': ex_prob,
            'losses': losses,
            'loss': loss
        }

    @staticmethod
    def action_to_label(good_action):
        pos_reward_indice = np.int_(good_action)
        loss_mask = pos_reward_indice
        return loss_mask, [1]

    @staticmethod
    def get_null_label(any_action):
        return np.int_(any_action) , [0]