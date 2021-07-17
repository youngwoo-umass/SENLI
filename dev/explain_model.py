import abc

import numpy as np
import tensorflow as tf


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
        for tag_idx in range(self.num_tags):
            ex_logits = tf.keras.layers.Dense(2, name="ex_{}".format(tag_idx))(self.sequence_output)
            ex_prob = tf.nn.softmax(ex_logits)

            self.ex_logits.append(ex_logits)
            self.ex_probs.append(ex_prob[:, :, 1])

    def modeling_co(self):
        self.ex_probs = []
        self.ex_logits = []
        for tag_idx in range(self.num_tags):
            ex_logits = tf.keras.layers.Dense(1, name="ex_{}".format(tag_idx))(self.sequence_output)
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
    def get_scores(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def action_to_label(good_action):
        pass


class ExplainModeling(ExplainModelingInterface):
    def __init__(self, seq_max, num_tags):
        self.seq_max = seq_max
        self.num_tags = num_tags
        self.tag_info = []

    def call(self, sequence_output):
        for tag_idx in range(self.num_tags):
            self.tag_info.append(self.model_tag(
                sequence_output, self.seq_max, "ex_{}".format(tag_idx)))


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

    def get_ex_scores(self, label_idx):
        return self.get_scores()[label_idx]

    @abc.abstractmethod
    def model_loss(self, ex_labels, valid_mask):
        pass


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=1, keepdims=True)
    my = tf.reduce_mean(y, axis=1, keepdims=True)
    xm, ym = x-mx, y-my
    r_num = tf.reduce_sum(tf.multiply(xm,ym), axis=1, keepdims=True)
    #r_den = tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(xm), axis=1, keep_dims=True)), tf.reduce_sum(tf.square(ym), axis=1, keep_dims=True)))
    r_den = tf.sqrt(tf.reduce_sum(tf.square(xm), axis=1, keepdims=True))
    r = r_num / (r_den + 0.00000001)
    r = tf.maximum(tf.minimum(r, 1.0), -1.0)
    return -r



class CorrelationModeling(ExplainModeling):
    def __init__(self, seq_max, num_tags, input_tensors):
        super(CorrelationModeling, self).__init__(seq_max, num_tags, input_tensors)

    def model_tag(self, ex_labels, valid_mask, sequence_output, seq_max, var_name):
        with tf.variable_scope(var_name):
            ex_logits = tf.keras.layers.Dense(1, name=var_name)(sequence_output)
            ex_logits = tf.reshape(ex_logits, [-1, seq_max])
            labels_ = tf.cast(tf.greater(ex_labels, 0), tf.float32)
            losses = correlation_coefficient_loss(ex_logits, -labels_)
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
    def __init__(self, seq_max, num_tags):
        super(CrossEntropyModeling, self).__init__(
            seq_max, num_tags)

    def model_tag(self, sequence_output, seq_max, var_name):
        ex_logits = tf.keras.layers.Dense(2, name=var_name)(sequence_output)
        ex_prob = tf.nn.softmax(ex_logits)[:, :, 1]
        self.ex_logits = ex_logits
        return {
            'ex_logits': ex_logits,
            'score': ex_prob,
        }

    def model_loss(self, ex_labels, valid_mask):
        ex_logits = self.ex_logits
        ce = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE,
            from_logits=True
        )
        losses = ce(ex_labels, ex_logits)
        losses = tf.cast(valid_mask, tf.float32) * losses
        loss = tf.reduce_mean(losses)
        return {
            'labels': ex_labels,
            'mask': valid_mask,
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


def action_penalty(action):
    num_tag = np.count_nonzero(action)
    penalty = (num_tag - 3) * 0.1 if num_tag > 3 else 0
    return penalty


def tag_informative_eq(explain_tag, before_prob, after_prob, action):
    if explain_tag == 'conflict':
        score = before_prob[2] - after_prob[2]
    elif explain_tag == 'match':
        score = before_prob[0] - after_prob[0]
    elif explain_tag == 'mismatch':
        score = before_prob[1] - after_prob[1]
    else:
        assert False
    score = score - action_penalty(action)
    return score

