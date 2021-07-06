import numpy as np


class NLIExTrainConfig:
    vocab_filename = "bert_voca.txt"
    vocab_size = 30522
    seq_length = 300
    max_steps = 73630
    num_gpu = 1
    num_deletion = 20
    g_val = 0.5
    save_train_payload = False
    drop_thres = 0.3


def tag_informative_eq2(explain_tag, before_prob, after_prob, action):
    penalty = action_penalty(action)
    if explain_tag == 'conflict':
        score = (before_prob[2] - before_prob[0]) - (after_prob[2] - after_prob[0])
    elif explain_tag == 'match':
        # Increase of neutral
        score = (before_prob[2] + before_prob[0]) - (after_prob[2] + after_prob[0])
        # ( 1 - before_prob[1] ) - (1 - after_prob[1]) = after_prob[1] - before_prob[1] = increase of neutral
    elif explain_tag == 'mismatch':
        score = before_prob[1] - after_prob[1]
    else:
        assert False
    score = score - penalty
    return score


def action_penalty(action):
    num_tag = np.count_nonzero(action)
    penalty = (num_tag - 3) * 0.1 if num_tag > 3 else 0
    return penalty


def tag_informative_eq1_max(explain_tag, before_prob, after_prob, action):
    label_idx = np.argmax(before_prob)
    score = before_prob[label_idx] - after_prob[label_idx]
    score = score - action_penalty(action)
    return score


def tag_informative_eq1_per_label(explain_tag, before_prob, after_prob, action):
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


def get_informative_fn_by_name(name):
    return {
        "eq2": tag_informative_eq2,
        "eq1_max": tag_informative_eq1_max,
        "eq1_per_label": tag_informative_eq1_per_label,
    }[name]