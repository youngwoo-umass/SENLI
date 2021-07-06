import tensorflow as tf

from transformer import bert
from transformer.bert import dropout
from tf_v2_support import placeholder, tf1
from trainer import tf_module


class ClassificationB:
    def __init__(self, is_training, hidden_size, num_classes):
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.is_training = is_training

    def call(self, pooled_output, label_ids):
        if self.is_training:
            pooled_output = dropout(pooled_output, 0.1)

        self.pooled_output = pooled_output
        #self.logits = tf.layers.dense(pooled_output, self.num_classes, name="cls_dense")
        output_weights = tf1.get_variable(
            "output_weights", [self.num_classes, self.hidden_size],
            initializer=tf1.truncated_normal_initializer(stddev=0.02)
        )

        output_bias = tf1.get_variable(
            "output_bias", [self.num_classes],
            initializer=tf1.zeros_initializer()
        )

        logits = tf.matmul(pooled_output, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        self.logits = logits

        preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
        labels = tf.one_hot(label_ids, self.num_classes)
        # self.loss_arr = tf.nn.softmax_cross_entropy_with_logits_v2(
        #     logits=self.logits,
        #     labels=labels)
        self.loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_ids)

        self.loss = tf.reduce_mean(self.loss_arr)
        self.acc = tf_module.accuracy(self.logits, label_ids)


class transformer_nli_pooled:
    def __init__(self, hp, voca_size, is_training=True):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False

        input_ids = placeholder(tf.int64, [None, seq_length])
        input_mask = placeholder(tf.int64, [None, seq_length])
        segment_ids = placeholder(tf.int64, [None, seq_length])
        label_ids = placeholder(tf.int64, [None])
        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = label_ids

        use_one_hot_embeddings = use_tpu
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        pooled_output = self.model.get_pooled_output()

        task = ClassificationB(is_training, hp.hidden_units, 3)
        task.call(pooled_output, label_ids)
        self.loss = task.loss
        self.logits = task.logits
        self.acc = task.acc

    def batch2feed_dict(self, batch):
        if len(batch) == 3:
            x0, x1, x2 = batch
            feed_dict = {
                self.x_list[0]: x0,
                self.x_list[1]: x1,
                self.x_list[2]: x2,
            }
        else:
            x0, x1, x2, y = batch
            feed_dict = {
                self.x_list[0]: x0,
                self.x_list[1]: x1,
                self.x_list[2]: x2,
                self.y: y,
            }
        return feed_dict

