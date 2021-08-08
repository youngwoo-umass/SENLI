import os
from typing import List, Iterable, Callable, Dict, Tuple, Set, NamedTuple, NewType

from bert.tokenization.bert_tokenization import FullTokenizer
from keras_preprocessing.sequence import pad_sequences

from dev.bert_common import createTokenizer
from path_manager import mnli_dir, mnli_ex_dir
import tensorflow as tf


tag_prefixes = ["c", "e", "n"]


def parse_indice_str(s):
    s = s.strip()
    if s:
        return list(map(int, s.split(",")))
    else:
        return []


def load_ex_eval_data(split) -> Dict[str, List]:
    output = {}
    for tag_prefix in tag_prefixes:
        tag = {
            'c': "conflict",
            "e": "match",
            "n": "mismatch"
        }[tag_prefix]
        output[tag] = load_ex_eval_for_tag(split, tag_prefix)
    return output


def load_ex_eval_for_tag(split, tag_prefix):
    file_path = os.path.join(mnli_ex_dir, "{}_{}.tsv".format(tag_prefix, split))
    data = []
    with open(file_path, encoding='utf-8', ) as csvFile:
        for idx, line in enumerate(csvFile):
            if idx == 0:
                continue
            data_id, p_text, h_text, p_indices, h_indices = line.split("\t")

            p_indices_: List[int] = parse_indice_str(p_indices)
            h_indices_: List[int] = parse_indice_str(h_indices)
            e = data_id, p_text, h_text, p_indices_, h_indices_
            data.append(e)

    return data


def tokenize_with_indices(tokenizer: FullTokenizer, text: str) -> Tuple[List[str], Dict[int, int]]:
    space_split_tokens = text.split()
    subword_idx_to_space_idx = {}
    output_tokens = []
    subword_idx = 0
    for space_token_idx, space_token in enumerate(space_split_tokens):
        tokens = tokenizer.tokenize(space_token)
        for token in tokens:
            subword_idx_to_space_idx[subword_idx] = space_token_idx
            subword_idx += 1
            output_tokens.append(token)
    return output_tokens, subword_idx_to_space_idx


def encode_pair_inst(tokenizer: FullTokenizer, max_seq_length, paired: Tuple[str, str]) \
        -> Tuple[List, List, Dict]:
    t1, t2 = paired
    tokens1, mapping1 = tokenize_with_indices(tokenizer, t1)
    tokens2, mapping2 = tokenize_with_indices(tokenizer, t2)

    mapping_combined = {}

    offset = 1
    for sb_idx, space_idx in mapping1.items():
        mapping_combined[sb_idx+offset] = 0, space_idx

    offset = len(tokens1) + 2
    for sb_idx, space_idx in mapping2.items():
        mapping_combined[sb_idx+offset] = 1, space_idx

    combined_tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
    seg1_len = 2 + len(tokens1)
    seg2_len = 1 + len(tokens2)
    token_ids = tokenizer.convert_tokens_to_ids(combined_tokens)
    segment_ids = [0] * seg1_len + [1] * seg2_len
    token_ids = pad_sequences([token_ids], max_seq_length, padding='post')[0]
    segment_ids = pad_sequences([segment_ids], max_seq_length, padding='post')[0]
    return token_ids, segment_ids, mapping_combined


DataID = NewType('DataID', int)
class EvalSet(NamedTuple):
    dataset: tf.data.Dataset
    tokenize_mapping_d: Dict[DataID, Dict[int, Tuple[int, int]]]
    data_id_mapping_d: Dict[DataID, str]
    label_d: Dict[DataID, Tuple[List[int], List[int]]]


def process_nli_ex_data(data: List[Tuple[str, str, str, List, List]],
                        max_seq_length) -> EvalSet:
    tokenizer = createTokenizer()

    encoded_data = []
    data_id_itr: int = 0
    tokenize_mapping_d = {}
    label_d = {}
    data_id_mapping: Dict[DataID, str] = {}
    for data_id_s, p, h, pi, hi in data:
        input_ids, seg_ids, tokenize_mapping = encode_pair_inst(tokenizer, max_seq_length, (p, h))
        data_id = DataID(data_id_itr)
        encoded_data.append((input_ids, seg_ids, data_id))
        tokenize_mapping_d[data_id] = tokenize_mapping
        data_id_mapping[data_id] = data_id_s
        label_d[data_id] = pi, hi
        data_id_itr += 1

    def gen():
        for t in encoded_data:
            yield t

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32),
            tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )
    return EvalSet(dataset, tokenize_mapping_d, data_id_mapping, label_d)


def load_nli_ex_eval_encoded(split, max_seq_length)\
        -> List[Tuple[str, EvalSet]]:
    output = []
    for tag, entries in load_ex_eval_data(split).items():
        eval_set: EvalSet = process_nli_ex_data(entries, max_seq_length)
        output.append((tag, eval_set))
    return output


def scores_to_ap(label: List[int], token_scores: List[Tuple[int, float]]) -> float:
    tp = 0
    sum = 0
    n_pred_pos = 0
    for idx, score in token_scores:
        n_pred_pos += 1
        if idx in label:
            tp += 1
            prec = (tp / n_pred_pos)
            assert prec <= 1
            sum += prec
    assert sum <= len(label)
    return sum / len(label) if label else 1


def something():
    d = load_nli_ex_eval_encoded("dev", 300)
    for tag, dataset, tokenize_mapping, data_id_mapping in d:
        print(tag, type(dataset), len(data_id_mapping))


def main():
    data = load_ex_eval_for_tag("dev", "n")
    for e in data:
        print(e)


if __name__ == "__main__":
    main()
