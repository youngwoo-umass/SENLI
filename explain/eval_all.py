from functools import partial

import cpath
from data_generator.NLI.nli import *
from data_generator.data_parser import esnli
from explain.eval_pr import *


def eval_nli_explain(pred_list, gold_list, only_prem, only_hypo = False):
    if len(pred_list) != len(gold_list):
        print("Warning")
        print("pred len={}".format(len(pred_list)))
        print("gold len={}".format(len(gold_list)))

    if only_prem:
        p1_p, p1_h = p_at_k_list_ind(pred_list, gold_list, 1)
        #p_at_20 = p_at_k_list(pred_list, gold_list, 20)
        p_auc, h_auc = PR_AUC_ind(pred_list, gold_list)
        MAP_p, MAP_h = MAP_ind(pred_list, gold_list)
        scores = {
            "P@1": p1_p,
            "MAP":MAP_p,
            "AUC": p_auc,
        }
        return scores
    elif only_hypo:
        p1_p, p1_h = p_at_k_list_ind(pred_list, gold_list, 1)
        # p_at_20 = p_at_k_list(pred_list, gold_list, 20)
        p_auc, h_auc = PR_AUC_ind(pred_list, gold_list)
        MAP_p, MAP_h = MAP_ind(pred_list, gold_list)
        scores = {
            "P@1": p1_h,
            "MAP": MAP_h,
            "AUC": h_auc,
        }
        return scores
    else:
        p1 = p_at_k_list(pred_list, gold_list, 1)
        # p_at_20 = p_at_k_list(pred_list, gold_list, 20)
        auc_score = PR_AUC(pred_list, gold_list)
        MAP_score = MAP(pred_list, gold_list)
        scores = {
            "P@1": p1,
            "MAP": MAP_score,
            "AUC": auc_score,
        }
        return scores




def eval_acc(pred_list, gold_list, only_prem, only_hypo = False):
    if len(pred_list) != len(gold_list):
        print("Warning")
        print("pred len={}".format(len(pred_list)))
        print("gold len={}".format(len(gold_list)))

    if only_prem:
        p1_p, p1_h = p_at_k_list_ind(pred_list, gold_list, 1)
        p_auc, h_auc = PR_AUC_ind(pred_list, gold_list)
        MAP_p, MAP_h = MAP_ind(pred_list, gold_list)
        scores = {
            "P@1": p1_p,
            "MAP":MAP_p,
            "AUC": p_auc,
        }
        return scores
    elif only_hypo:
        p1_p, p1_h = p_at_k_list_ind(pred_list, gold_list, 1)
        # p_at_20 = p_at_k_list(pred_list, gold_list, 20)
        p_auc, h_auc = PR_AUC_ind(pred_list, gold_list)
        MAP_p, MAP_h = MAP_ind(pred_list, gold_list)
        scores = {
            "P@1": p1_h,
            "MAP": MAP_h,
            "AUC": h_auc,
        }
        return scores
    else:
        p1 = p_at_k_list(pred_list, gold_list, 1)
        # p_at_20 = p_at_k_list(pred_list, gold_list, 20)
        auc_score = PR_AUC(pred_list, gold_list)
        MAP_score = MAP(pred_list, gold_list)
        scores = {
            "P@1": p1,
            "MAP": MAP_score,
            "AUC": auc_score,
        }
        return scores


def get_suc_list(score_list, cut):
    l = []
    total = len(score_list)

    for score, label in score_list:
        p = score >= cut

        if label == p:
            l.append(1)
        else:
            l.append(0)

    return l




def eval_pr(pred_list, gold_list, cut_off, only_prem, only_hypo = False):
    if len(pred_list) != len(gold_list):
        print("Warning")
        print("pred len={}".format(len(pred_list)))
        print("gold len={}".format(len(gold_list)))

    (cut_p, _), (cut_h, _) = cut_off
    p_info, h_info = get_all_score_label(pred_list, gold_list)


    if only_prem:
        p,r,f1 = get_pr(p_info, cut_p)
        acc = get_acc(p_info, cut_p)
        scores = {
            "P": p,
            "R":r,
            "F1":f1,
            "ACC":acc
        }
        return scores
    elif only_hypo:
        p, r, f1 = get_pr(h_info, cut_h)
        acc = get_acc(h_info, cut_h)
        scores = {
            "P": p,
            "R": r,
            "F1": f1,
            "ACC": acc
        }
        return scores
    else:
        pp, pr, pf1 = get_pr(p_info, cut_p)
        hp, hr, hf1 = get_pr(h_info, cut_h)
        #acc_p = get_acc(p_info, cut_p)
        #acc_h = get_acc(h_info, cut_h)

        correct_p = get_suc_list(p_info, cut_p)
        correct_h = get_suc_list(h_info, cut_h)
        all_c= correct_p + correct_h
        acc_p = sum(correct_p)/ len(correct_p)
        acc_h = sum(correct_h) / len(correct_h)
        acc = sum(all_c) / len(all_c)
        acc = (acc_p + acc_h)/2
        scores = {
            "Prem P": pp,
            "Prem R": pr,
            "Prem F1": pf1,
            "Hypo P": hp,
            "Hypo R": hr,
            "Hypo F1": hf1,
            "ACC": acc,
            "Acc_p":acc_p,
            "Acc_h":acc_h,
        }
        return scores

def load_gold(data_id):
    if data_id == 'conflict':
        data= load_mnli_explain_0()
        result = []
        for entry in data:
            result.append((entry['p_explain'],  entry['h_explain']))
    elif data_id == 'match' or data_id == 'mismatch':
        data = load_nli_explain_1(data_id)
        result = []
        for entry in data:
            result.append((entry[2], entry[3]))
    elif data_id.startswith("conflict_"):
        data = load_nli_explain_3("conflict_0_99", "conflict")
        result = []
        for entry in data:
            result.append((entry[2], entry[3]))
    elif data_id.startswith("test_"):
        data = load_nli_explain_3(data_id + "_idx", data_id)
        result = []
        for entry in data:
            result.append((entry[2], entry[3]))

    return result




def run_analysis():
    data_id = "conflict_0_99"
    data_id = "conflict"
    data_id = "match"
    prem_only = data_id.startswith("match")
    gold_list = load_gold(data_id)
    def p_at_k(rank_list, gold_set, k):
        tp = 0
        for score, e in rank_list[:k]:
            if e in gold_set:
                tp += 1
        return tp / k

    def AP(pred, gold):
        n_pred_pos = 0
        tp = 0
        sum = 0
        for score, e in pred:
            n_pred_pos += 1
            if e in gold:
                tp += 1
                sum += (tp / n_pred_pos)
        return sum / len(gold)

    #runs_list = ["pred_O_conflict_conflict"]
    runs_list = ["pred_P_match_match"]
    for run_name in runs_list:
        predictions = load_from_pickle(run_name)

        score_list_h = []
        score_list_p = []
        idx = 0
        for pred, gold in zip(predictions, gold_list):
            pred_p, pred_h = pred
            gold_p, gold_h = gold
            fail = False

            if prem_only:
                s1 = AP(pred_p, gold_p)
                if s1 < 0.96:
                    fail = True
                if fail:
                    print("-------------------")
                    print("id : ", idx)
                    print("AP : ", s1)
                    print("pred_p:", pred_p)
                    print("gold_p", gold_p)
            else:
                if gold_p:
                    s1 = p_at_k(pred_p, gold_p, 1)
                    if s1 < 0.99:
                        fail = True
                    score_list_p.append(s1)
                if gold_h :
                    s2 = p_at_k(pred_h, gold_h, 1)
                    if s2 < 0.99:
                        fail = True
                    score_list_h.append(s2)

                if fail:
                    print("-------------------")
                    print("id : ", idx)
                    print("pred_p:", pred_p)
                    print("gold_p", gold_p)
                    print("pred_h:", pred_h)
                    print("gold_h", gold_h)
            idx += 1


def run_eval():
    data_id = "conflict_0_99"
    data_id = "match"

    #runs_list = ["pred_O_conflict_conflict", "pred_deeplift_conflict", "pred_grad*input_conflict",
    #             "pred_intgrad_conflict", "pred_saliency_conflict"]

    runs_list = ["pred_P_match_match"]

    gold_list = load_gold(data_id)

    for run_name in runs_list:
        predictions = load_from_pickle(run_name)
        only_prem = False if data_id == 'conflict' else False
        scores = eval_nli_explain(predictions, gold_list, only_prem)
        print(run_name)
        for key in scores:
            print("{}\t{}".format(key, scores[key]))


def load_prediction(name):
    file_path = os.path.join(cpath.output_path, "prediction", "nli", name + ".pickle")
    return pickle.load(open(file_path, "rb"))





def paired_p_test(scorer, predictions1, predictions2, gold_list, only_prem, only_hypo):
    from scipy import stats

    p_score_list1, h_score_list1 = scorer(predictions1, gold_list)
    p_score_list2, h_score_list2 = scorer(predictions2, gold_list)

    assert len(p_score_list1) == len(p_score_list2)
    assert len(h_score_list1) == len(h_score_list2)

    if only_prem:
        score_list_1 = p_score_list1
        score_list_2 = p_score_list2
    elif only_hypo:
        score_list_1 = h_score_list1
        score_list_2 = h_score_list2
    else:
        score_list_1 = p_score_list1 + h_score_list1
        score_list_2 = p_score_list2 + h_score_list2

    _, p = stats.ttest_rel(score_list_1, score_list_2)
    return p


def bootstrap_paired_t(score_list1, score_list2, trial):
    from scipy import stats

    total = len(score_list1)

    sample_size = int(0.1 * total)

    l1 = []
    l2 = []
    for j in range(trial):
        sample1 = []
        sample2 = []
        for i in range(sample_size):
            idx = random.randint(0, total-1)
            sample1.append(score_list1[idx])
            sample2.append(score_list2[idx])

        l1.append(average(sample1))
        l2.append(average(sample2))
    return stats.ttest_rel(l1, l2)


def McNemar(score_list1, score_list2):
    import scipy
    def mcnemar_p(b, c):
        """Computes McNemar's test.
        Args:
          b: the number of "wins" for the first condition.
          c: the number of "wins" for the second condition.
        Returns:
          A p-value for McNemar's test.
        """
        n = b + c
        x = min(b, c)
        dist = scipy.stats.binom(n, .5)
        return 2. * dist.cdf(x)

    l = len(score_list1)
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(l):
        p1 = score_list1[i]
        p2 = score_list2[i]

        if p1 and p2:
            a += 1
        elif p1 and not p2:
            b += 1
        elif not p1 and p2:
            c += 1
        elif not p1 and not p2:
            d += 1
        else:
            assert False
    return None, mcnemar_p(b, c)

def paired_p_test2(scorer1, scorer2, predictions1, predictions2, gold_list, only_prem, only_hypo):
    p_score_list1, h_score_list1 = scorer1(predictions1, gold_list)
    p_score_list2, h_score_list2 = scorer2(predictions2, gold_list)

    assert len(p_score_list1) == len(p_score_list2)
    assert len(h_score_list1) == len(h_score_list2)

    if only_prem:
        score_list_1 = p_score_list1
        score_list_2 = p_score_list2
    elif only_hypo:
        score_list_1 = h_score_list1
        score_list_2 = h_score_list2
    else:
        score_list_1 = p_score_list1 + h_score_list1
        score_list_2 = p_score_list2 + h_score_list2

    # print("pp2", average(score_list_1))
    # print("pp2", average(score_list_2))
    _, p = McNemar(score_list_1, score_list_2)
    #_, p = stats.ttest_rel(score_list_1, score_list_2)
    return p



def paired_p_test_runner():
    best_runner = {
            'match':['CE_match','Y_match'],
            'mismatch':['CE_mismatch','V_mismatch'],
            'conflict':['Y_conflict', 'CE_conflict'],
        }

    for target_label in ["mismatch", "match", "conflict"]:
        data_id = "test_{}".format(target_label)
        gold_list = load_gold(data_id)

        only_prem = True if target_label == 'match' else False
        only_hypo = True if target_label == 'mismatch' else False

        predictions_list = []
        for method_name in best_runner[target_label]:
            run_name = "pred_" + method_name + "_" + data_id
            predictions_list.append(load_prediction(run_name))

        def p_at_1(pred, gold):
            return p_at_k_list_inner(pred, gold, 1)

        def AP(pred, gold):
            n_pred_pos = 0
            tp = 0
            sum = 0
            for score, e in pred:
                n_pred_pos += 1
                if e in gold:
                    tp += 1
                    sum += (tp / n_pred_pos)
            return sum / len(gold)

        print(best_runner[target_label])
        p_pat1 = paired_p_test(p_at_1, predictions_list[0], predictions_list[1], gold_list, only_prem, only_hypo)
        print("p-value for p@1", p_pat1)
        p_AP = paired_p_test(MAP_inner, predictions_list[0], predictions_list[1], gold_list, only_prem, only_hypo)
        print("p-value for MAP", p_AP)



def paired_p_test_runner_new():
    best_runner = {
            'match':['CE_match','X_match'],
            'mismatch':['CE_mismatch','W_mismatch'],
            'conflict':['Y_conflict', 'CE_conflict'],
        }
    best_runner = {
            'match':['NLIEx_AnyA', 'CE_match'],
            'mismatch':['CE_mismatch','NLIEx_AnyA'],
            'conflict':['Y_conflict', 'NLIEx_AnyA'],
        }

    for target_label in ["conflict", "match", "mismatch"]:
        data_id = "{}_1000".format(target_label)

        #gold_list = load_gold(data_id)
        gold_list = read_gold_label("gold_{}_100_700.csv".format(target_label))
        only_prem = True if target_label == 'match' else False
        only_hypo = True if target_label == 'mismatch' else False

        predictions_list = []
        for method_name in best_runner[target_label]:
            run_name = "pred_" + method_name + "_" + data_id
            predictions = load_part_from_prediction(run_name, 600)
            predictions_list.append(predictions)

        def p_at_1(pred, gold):
            return p_at_k_list_inner(pred, gold, 1)

        def AP(pred, gold):
            n_pred_pos = 0
            tp = 0
            sum = 0
            for score, e in pred:
                n_pred_pos += 1
                if e in gold:
                    tp += 1
                    sum += (tp / n_pred_pos)
            return sum / len(gold)

        first, second =best_runner[target_label]
        print("{} - {} > {}".format(target_label, first, second))
        p_pat1 = paired_p_test(p_at_1, predictions_list[0], predictions_list[1], gold_list, only_prem, only_hypo)
        print("p-value for p@1", p_pat1)
        p_AP = paired_p_test(MAP_inner, predictions_list[0], predictions_list[1], gold_list, only_prem, only_hypo)
        print("p-value for MAP", p_AP)


def paired_p_test_runner_acc():
    best_runner = {
            'conflict':['Y_conflict', 'LIME'],
            'match':['CE_match','X_match'],
            'mismatch':['CE_mismatch', 'W_mismatch'],
        }
    best_runner = {
            'match':['NLIEx_AnyA', 'CE_match'],
            'mismatch':['CE_mismatch','NLIEx_AnyA'],
            'conflict':['Y_conflict', 'NLIEx_AnyA'],
        }


    for target_label in ["mismatch", "match", "conflict"]:
        data_id = "{}_1000".format(target_label)

        dev_data_id = {
            'match': 'match',
            'mismatch': 'mismatch',
            'conflict': 'conflict_0_99'
        }[target_label]
        dev_label = load_gold(dev_data_id)

        def get_tune(run_name):
            predictions = load_prediction_head(run_name)
            if target_label == 'match':
                predictions = predictions[:56]
            elif target_label == 'mismatch':
                predictions = predictions[:34]
                
            return tune_cut(predictions, dev_label)


        #gold_list = load_gold(data_id)
        gold_list = read_gold_label("gold_{}_100_700.csv".format(target_label))
        only_prem = True if target_label == 'match' else False
        only_hypo = True if target_label == 'mismatch' else False

        predictions_list = []
        cut = {}
        for method_name in best_runner[target_label]:
            run_name = "pred_" + method_name + "_" + data_id
            predictions = load_part_from_prediction(run_name, 600)
            predictions_list.append(predictions)
            cut[method_name] = get_tune(run_name)

        def acc_compare(cut, pred, gold):

            (cut_p, _), (cut_h, _) = cut

            p_info, h_info = get_all_score_label(pred, gold)

            if only_prem:
                acc_p = acc_compare_inner(cut_p, p_info)
                acc_h = []
            elif only_hypo:
                acc_h = acc_compare_inner(cut_h, h_info)
                acc_p = []
            else:
                acc_p = acc_compare_inner(cut_p, p_info)
                acc_h = acc_compare_inner(cut_h, h_info)
            print(acc_p)
            print(acc_h)
            return acc_p,acc_h

        def acc_compare_inner(cut, pred):
            res = []
            for score, e in pred:
                d = score > cut
                res.append(int(d == e))
            return res

        first, second =best_runner[target_label]
        print("{} - {} > {}".format(target_label, first, second))
        p_acc = paired_p_test2(partial(acc_compare, cut[first]),
                               partial(acc_compare, cut[second]),
                               predictions_list[0], predictions_list[1], gold_list, only_prem, only_hypo)
        print("p-value for acc", p_acc)


def mismatch_p():
    methods = ['V_mismatch', 'W_mismatch']
    target_label = "mismatch"
    data_id = "test_{}".format(target_label)
    gold_list = load_gold(data_id)

    only_prem = True if target_label == 'match' else False
    only_hypo = True if target_label == 'mismatch' else False

    predictions_list = []
    for method_name in methods:
        run_name = "pred_" + method_name + "_" + data_id
        predictions_list.append(load_prediction(run_name))

    print(methods)
    p_AP = paired_p_test(MAP_inner, predictions_list[0], predictions_list[1], gold_list, only_prem, only_hypo)
    print("p-value for MAP", p_AP)


def run_cut_off_small():
    target_label =  "match"

    model_name = {
        'match':'Y_match',
        'mismatch':'V_mismatch',
        'conflict':'Y_conflict',
    }[target_label]

    ###data_id = target_label
    data_id = "test_{}".format(target_label)
    ###dev_label = load_gold(target_label)

    dev_label = load_gold(data_id)
    all_method = ["random", "idf", "saliency",  "grad*input", "intgrad", "deletion", "deletion_seq", "LIME", model_name]

    run_names = []
    for method_name in all_method:
        ###run_name = "pred_dev_" + method_name + "_" + data_id
        run_name = "pred_" + method_name + "_" + data_id
        run_names.append(run_name)

    best_cuts = {}
    for run_name in run_names:
        predictions = load_prediction(run_name)
        best_cuts[run_name] = tune_cut(predictions, dev_label)
        print(run_name, best_cuts[run_name])

    data_id = "test_{}".format(target_label)
    gold_list = load_gold(data_id)

    run_names = []
    for method_name in all_method:
        run_name = "pred_" + method_name + "_" + data_id
        run_names.append(run_name)

    for run_name in run_names:
        predictions = load_prediction(run_name)
        print(run_name)
        cut = best_cuts[run_name]
        if target_label == 'conflict':
            scores = eval_pr(predictions, gold_list, cut, False, False)
        elif target_label == 'match':
            scores = eval_pr(predictions, gold_list, cut, True, False)
        elif target_label == 'mismatch':
            scores = eval_pr(predictions, gold_list, cut, False, True)
        for key in scores:
            print("{}".format(key), end="\t")
        print()
        for key in scores:
            print("{}".format(scores[key]), end="\t")
        print()




def run_cut_off(target_label):
    dev_data_id = {
        'match':'match',
        'mismatch':'mismatch',
        'conflict':'conflict_0_99'
    }[target_label]
    print(dev_data_id)
    dev_label = load_gold(dev_data_id)

    model_name = {
        'match':'X_match',
        'mismatch':'W_mismatch',
        'conflict':'Y_conflict',
    }[target_label]
    ce_run = {
        'match':'CE_match',
        'mismatch':'CE_mismatch',
        'conflict':'CE_conflict',
    }[target_label]

    data_id = "{}_1000".format(target_label)

    all_method = ["random", "idf", "saliency",  "grad*input", "intgrad", "LIME", "deletion", "deletion_seq", model_name, ce_run, "NLIEx_AnyA"]
    all_method = ["X_match_del_0.1", "X_match_del_0.2", "X_match_del_0.3", "X_match_del_0.4", "X_match_del_0.5",
                   "X_match_del_0.6", "X_match_del_0.7", "X_match_del_0.8", "X_match_del_0.9"]

    run_names = []
    for method_name in all_method:
        run_name = "pred_" + method_name + "_" + data_id
        run_names.append(run_name)

    best_cuts = {}
    for run_name in run_names:
        predictions = load_prediction_head(run_name)
        if target_label == 'match':
            predictions = predictions[:56]
        elif target_label == 'mismatch':
            predictions = predictions[:34]
        best_cuts[run_name] = tune_cut(predictions, dev_label)
        print(run_name, best_cuts[run_name])

    gold_list = read_gold_label("gold_{}_100_700.csv".format(target_label))

    run_names = []
    for method_name in all_method:
        run_name = "pred_" + method_name + "_" + data_id
        run_names.append(run_name)

    cnt = 0
    for run_name in run_names:
        predictions = load_part_from_prediction(run_name, 600)

        print(run_name, end="\t")
        cut = best_cuts[run_name]
        if target_label == 'conflict':
            scores = eval_pr(predictions, gold_list, cut, False, False)
        elif target_label == 'match':
            scores = eval_pr(predictions, gold_list, cut, True, False)
        elif target_label == 'mismatch':
            scores = eval_pr(predictions, gold_list, cut, False, True)

        for key in scores:
            print("{}".format(scores[key]), end="\t")
        print()
        cnt += 1
        if cnt == len(run_names):
            print("\t", end="")
            for key in scores:
                print("{}".format(key), end="\t")
            print()



def eval_snli():
    model_name = "SNLIEx_B"

    ###data_id = target_label
    ###dev_label = load_gold(target_label)
    dev_label = esnli.load_gold("dev")[:50]
    all_method = ["LIME"]#[model_name]

    run_names = []
    for method_name in all_method:
        ###run_name = "pred_dev_" + method_name + "_" + data_id
        run_name = "pred_" + method_name + "_" + "dev" + "__0_50"
        run_names.append(run_name)

    best_cuts = {}
    for run_name in run_names:
        predictions = load_prediction(run_name)
        best_cuts[method_name] = tune_cut(predictions, dev_label)
        print(run_name, best_cuts[method_name])

    data_id = "test"
    gold_list = esnli.load_gold(data_id)[1500:4200]

    run_names = []
    for method_name in all_method:
        run_name = "pred_" + method_name + "_" + data_id
        run_names.append(run_name)

    for run_name in run_names:
        predictions = load_prediction(run_name)
        print(run_name)
        cut = best_cuts[method_name]
        scores = eval_pr(predictions, gold_list, cut, False, False)
        for key in scores:
            print("{}".format(key), end="\t")
        print()
        for key in scores:
            print("{}".format(scores[key]), end="\t")
        print()


def merge_lime():
    for target in ['mismatch']:#'conflict', 'match']:#,
        out_name = "pred_LIME_{}".format(target)
        data = []
        for step in range(0, 1000, 100):
            name = "pred_LIME_{}_1000__{}_{}".format(target, step, step+100)
            print(name)
            data.extend(load_cache(name))
        print(out_name)
        save_to_pickle(data, out_name)



def run_eval_acl():
    target_label =  "mismatch"
    data_id = "test_{}".format(target_label)
    label_name = "test_{}_idx".format(target_label)
    gold_list = load_gold(data_id)

    model_name = {
        'match':'Y_match',
        'mismatch':'V_mismatch',
        'conflict':'Y_conflict',
    }[target_label]



    run_names = []
    for method_name in ["random", "idf", "saliency",  "grad*input", "intgrad",
                        "deletion", "deletion_seq", model_name, 'W_mismatch', "AnyA"]:
        run_name = "pred_" + method_name + "_" + data_id
        run_names.append(run_name)

    for run_name in run_names:
        predictions = load_prediction(run_name)
        print(run_name)
        if target_label =='conflict':
            scores = eval_nli_explain(predictions, gold_list, False, False)
        elif target_label == 'match':
            scores = eval_nli_explain(predictions, gold_list, True, False)
        elif target_label == 'mismatch':
            scores = eval_nli_explain(predictions, gold_list, False, True)
        for key in scores:
            print("{}".format(key), end="\t")
        print()
        for key in scores:
            print("{}".format(scores[key]), end="\t")
        print()

def load_part_from_prediction(run_name, data_size):
    file_path = os.path.join(cpath.output_path, "prediction", "nli", run_name + ".pickle")
    data = pickle.load(open(file_path, "rb"))
    if len(data) == 900:
        return data[:data_size]
    elif len(data) == 1000:
        return data[100:100+data_size]


def load_prediction_head(run_name):
    file_path = os.path.join(cpath.output_path, "prediction", "nli", run_name + ".pickle")
    data = pickle.load(open(file_path, "rb"))
    if len(data) == 1000:
        return data[:100]
    else:
        assert False

def run_test_eval_emnlp(target_label, all_methods_str):
    gold_list = read_gold_label("gold_{}_100_700.csv".format(target_label))
    pred_id = "{}_1000".format(target_label)
    model_name = {
        'match': 'X_match',
        'mismatch': 'W_mismatch',
        'conflict': 'Y_conflict',
    }[target_label]
    ce_run = {
        'match':'CE_match',
        'mismatch':'CE_mismatch',
        'conflict':'CE_conflict',
    }[target_label]

    all_methods = all_methods_str.split(",")

    # all_methods = ["random", "idf", "saliency",  "grad*input", "intgrad", "LIME", "deletion", "deletion_seq", "NLIEx_AnyA", model_name, ce_run]
    #all_methods= ["X_match_del_0.1", "X_match_del_0.2","X_match_del_0.3","X_match_del_0.4", "X_match_del_0.5",
#                  "X_match_del_0.6","X_match_del_0.7", "X_match_del_0.8", "X_match_del_0.9"]
    run_names = []
    for method_name in all_methods:
        run_name = "pred_" + method_name + "_" + pred_id
        run_names.append(run_name)

    idx = 0
    for run_name in run_names:
        predictions = load_part_from_prediction(run_name, 600)

        if target_label == 'conflict':
            scores = eval_nli_explain(predictions, gold_list, False, False)
        elif target_label == 'match':
            scores = eval_nli_explain(predictions, gold_list, True, False)
        elif target_label == 'mismatch':
            scores = eval_nli_explain(predictions, gold_list, False, True)

        print(run_name, end="\t")
        for key in scores:
            print("{}".format(scores[key]), end="\t")
        print()

        idx += 1
        if idx == len(run_names):
            print("\t", end="")
            for key in scores:
                print("{}".format(key), end="\t")
            print()

def debuglime():
    run_name = "pred_LIME_conflict_1000"
    predictions = load_part_from_prediction(run_name, 200)

    p,h = predictions[0]

    for p,h in predictions:
        print(p)
        print(h)
        print(len(p))
        print(len(h))



if __name__ == '__main__':
    #run_eval()
    #run_eval_acl()
    #mismatch_p()
    #merge_lime()
    #paired_p_test_runner_new()
    paired_p_test_runner_acc()
    #run_cut_off("conflict")
    #run_cut_off("match")
    #run_cut_off("mismatch")
    #paired_p_test_runner()
    run_test_eval_emnlp("mismatch")
    run_test_eval_emnlp("match")
    run_test_eval_emnlp("conflict")

    #eval_snli()
    #debuglime()