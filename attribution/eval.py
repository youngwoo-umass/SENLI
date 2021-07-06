import numpy as np

from attribution.deleter_trsfmr import token_delete_with_indice
from data_generator.NLI import nli
from data_generator.NLI.enlidef import get_target_class
from evaluation import *


def eval_explain(conf_score, data_loader, tag):
    if tag == 'conflict':
        return eval_explain_0(conf_score, data_loader)
    else:
        return eval_explain_1(conf_score, data_loader, tag)


def eval_pairing(pair_logits_p, pair_logits_h, data_loader, enc_list, pair_infos):
    max_k = 9999
    pred_list = []
    gold_list = []
    for idx, entry in enumerate(pair_infos):
        _, score_h = data_loader.split_p_h(pair_logits_p[idx], enc_list[idx])  # If we mark premise, we check hypothesis parts only
        score_p, _ = data_loader.split_p_h(pair_logits_h[idx], enc_list[idx])
        prem, hypo, p_indice, h_indice = entry
        input_ids = enc_list[idx][0]
        p_enc, h_enc = data_loader.split_p_h(input_ids, enc_list[idx])

        p_explain = [] # indice of h which match given p
        h_explain = []

        h_set = set()
        for i in top_k_idx(score_h, max_k):
            # Convert the index of model's tokenization into space tokenized index
            v_i = data_loader.convert_index_out(hypo, h_enc, i)
            score = score_h[i]

            if v_i not in h_set:
                p_explain.append((score, v_i))
                h_set.add(v_i)

        p_set = set()
        for i in top_k_idx(score_p, max_k):
            v_i = data_loader.convert_index_out(prem, p_enc, i)
            score = score_p[i]

            if v_i not in p_set:
                h_explain.append((score, v_i))
                p_set.add(v_i)
        pred_list.append((p_explain, h_explain))
        gold_list.append((h_indice, p_indice))  # p_explain should be matched to p_indice

    p1_p, p1_h = p_at_k_list_ind(pred_list, gold_list, 1)
    # p_at_20 = p_at_k_list(pred_list, gold_list, 20)
    p_auc, h_auc = PR_AUC_ind(pred_list, gold_list)
    scores_p = {
        "P@1": p1_p,
        "AUC": p_auc,
    }
    scores_h = {
        "P@1": p1_h,
        "AUC": h_auc,
    }
    return scores_p, scores_h

def predict_translate(conf_score, data_loader, enc_payload, plain_payload):
    max_k = 999
    pred_list = []
    for idx, entry in enumerate(plain_payload):
        conf_p, conf_h = data_loader.split_p_h(conf_score[idx], enc_payload[idx])
        prem, hypo = entry
        input_ids = enc_payload[idx][0]
        p_enc, h_enc = data_loader.split_p_h(input_ids, enc_payload[idx])
        p_explain = []
        h_explain = []

        p_set = set()
        for i in top_k_idx(conf_p, max_k):
            # Convert the index of model's tokenization into space tokenized index
            v_i = data_loader.convert_index_out(prem, p_enc, i)
            score = conf_p[i]

            if v_i not in p_set:
                p_explain.append((score, v_i))
                p_set.add(v_i)

        h_set = set()
        for i in top_k_idx(conf_h, max_k):
            v_i = data_loader.convert_index_out(hypo, h_enc, i)
            score = conf_h[i]

            if v_i not in h_set:
                h_explain.append((score, v_i))
                h_set.add(v_i)

        pred_list.append((p_explain, h_explain))
    return pred_list

def eval_explain_1(conf_score, data_loader, tag):
    enc_explain_dev, explain_dev = data_loader.get_dev_explain_1(tag)
    num_inst = min(len(explain_dev), len(conf_score))
    max_k = 999
    pred_list = []
    gold_list = []
    for idx in range(num_inst):
        entry = explain_dev[idx]
        conf_p, conf_h = data_loader.split_p_h(conf_score[idx], enc_explain_dev[idx])
        prem, hypo, p_indice, h_indice = entry
        input_ids = enc_explain_dev[idx][0]
        p_enc, h_enc = data_loader.split_p_h(input_ids, enc_explain_dev[idx])

        p_explain = []
        h_explain = []

        p_set = set()
        for i in top_k_idx(conf_p, max_k):
            # Convert the index of model's tokenization into space tokenized index
            v_i = data_loader.convert_index_out(prem, p_enc, i)
            score = conf_p[i]

            if v_i not in p_set:
                p_explain.append((score, v_i))
                p_set.add(v_i)

        h_set = set()
        for i in top_k_idx(conf_h, max_k):
            v_i = data_loader.convert_index_out(hypo, h_enc, i)
            score = conf_h[i]

            if v_i not in h_set:
                h_explain.append((score, v_i))
                h_set.add(v_i)

        pred_list.append((p_explain, h_explain))
        gold_list.append((p_indice, h_indice))

    if tag == "match":
        p1_p, p1_h = p_at_k_list_ind(pred_list, gold_list, 1)
        #p_at_20 = p_at_k_list(pred_list, gold_list, 20)
        p_auc, h_auc = PR_AUC_ind(pred_list, gold_list)
        MAP_score = MAP(pred_list, gold_list)
        MAP_p, MAP_h = MAP_ind(pred_list, gold_list)
        scores = {
            "P@1": p1_p,
            "AUC": p_auc,
            #"P@20": p_at_20,
            "MAP":MAP_score,
            "MAP_p": MAP_p,
            "MAP_h": MAP_h,
        }
        return scores
    elif tag == "conflict":
        p1 = p_at_k_list(pred_list, gold_list, 1)
        # p_at_20 = p_at_k_list(pred_list, gold_list, 20)
        auc_score = PR_AUC(pred_list, gold_list)
        MAP_score = MAP(pred_list, gold_list)
        MAP_p, MAP_h = MAP_ind(pred_list, gold_list)
        scores = {
            "P@1": p1,
            "AUC": auc_score,
            # "P@20": p_at_20,
            "MAP": MAP_score,
            "MAP_p": MAP_p,
            "MAP_h": MAP_h,
        }
        return scores
    elif tag == 'mismatch':
        p1_p, p1_h = p_at_k_list_ind(pred_list, gold_list, 1)
        #p_at_20 = p_at_k_list(pred_list, gold_list, 20)
        p_auc, h_auc = PR_AUC_ind(pred_list, gold_list)
        MAP_score = MAP(pred_list, gold_list)
        MAP_p, MAP_h = MAP_ind(pred_list, gold_list)
        scores = {
            "P@1": p1_h,
            "AUC": h_auc,
            "MAP":MAP_h,
        }
        return scores

def eval_explain_0(conf_score, data_loader):
    enc_explain_dev, explain_dev = data_loader.get_dev_explain_0()

    pred_list = []
    gold_list = []
    end = 999
    for idx, entry in enumerate(explain_dev):
        attr_p, attr_h = data_loader.split_p_h(conf_score[idx], enc_explain_dev[idx])

        input_ids = enc_explain_dev[idx][0]
        p, h = data_loader.split_p_h(input_ids, enc_explain_dev[idx])

        p_explain = []
        p_set = set()
        for i in top_k_idx(attr_p, end):
            v_i = data_loader.convert_index_out(entry['p'], p, i)
            score = attr_p[i]

            if v_i not in p_set:
                p_explain.append((score, v_i))
                p_set.add(v_i)

        h_explain = []
        h_set = set()
        for i in top_k_idx(attr_h, end):
            v_i = data_loader.convert_index_out(entry['h'], h, i)
            score = attr_h[i]

            if v_i not in h_set:
                h_explain.append((score, v_i))
                h_set.add(v_i)

        pred_list.append((p_explain, h_explain))
        gold_list.append((entry['p_explain'], entry['h_explain']))

    p_at_1 = p_at_k_list(pred_list, gold_list, 1)
    MAP_score = MAP(pred_list, gold_list)
    auc = PR_AUC(pred_list, gold_list)
    scores = {
        "P@1": p_at_1,
        "MAP": MAP_score,
        "AUC": auc,
    }
    return scores


# contrib_score = numpy array
def eval_fidelity(contrib_score, input_data, forward_run, target_tag):
    target_class = get_target_class(target_tag)
    sorted_arg = np.flip(np.argsort(contrib_score, axis=1), axis=1)
    num_inst = len(input_data)

    def accuracy(logit_list):
        t = np.argmax(logit_list, axis=1) == target_class
        return np.count_nonzero(np.argmax(logit_list, axis=1) == target_class) / num_inst


    def get_delete_indice(x0, sorted_arg):
        indice = []
        for idx in sorted_arg:
            if x0[idx] == nli.SEP_ID or x0[idx] == nli.CLS_ID:
                None
            else:
                indice.append(idx)
        return indice

    # 0-deletion run

    acc_list = dict()

    acc_list[0] = accuracy(forward_run(input_data))

    for num_delete in range(1,20):
        new_data = []
        for i in range(num_inst):
            x0, x1, x2 = input_data[i]
            delete_indice = get_delete_indice(x0, sorted_arg[i])[:num_delete]
            x0_new, x1_new, x2_new = token_delete_with_indice(delete_indice, x0, x1, x2)
            new_data.append((x0_new, x1_new, x2_new))

        acc_list[num_delete] = accuracy(forward_run(new_data))

    return acc_list
