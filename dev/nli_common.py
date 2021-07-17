import csv
import os
from typing import List, Tuple

import tensorflow as tf

from dev.bert_common import encode_pair_inst_w_label, createTokenizer
from path_manager import mnli_dir
from tf_logging import senli_logging


label_list = ["entailment", "neutral", "contradiction", ]
tags = ["conflict", "match", "mismatch"]


def read_nli_data(file_name) -> List[Tuple[str, str, int]]:
    file_path = os.path.join(mnli_dir, file_name)
    data = []
    with open(file_path, encoding='utf-8', ) as csvFile:
        csv_reader = csv.reader(csvFile, delimiter="\t", quoting=csv.QUOTE_NONE)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            if line_count == 1:
                continue

            prem = row[8]
            hypo = row[9]
            label_str = row[-1]
            label = label_list.index(label_str)
            out_row = prem, hypo, label
            data.append(out_row)
    csvFile.close()
    senli_logging.info("{} datapoints from {}".format(len(data), file_name))
    return data


def get_nli_data(max_seq_length, name):
    text_data = read_nli_data(name)
    tokenizer = createTokenizer()

    def encode(inst: Tuple[str, str, int]):
        return encode_pair_inst_w_label(tokenizer, max_seq_length, inst)

    def gen():
        for t in text_data:
            yield encode(t)

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32),
            tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )
    return dataset


