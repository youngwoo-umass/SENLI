import argparse
import sys
from functools import partial

from data_generator.NLI.nli import read_gold_label
from explain.eval_all import paired_p_test, p_at_k_list_inner, MAP_inner, paired_p_test2, get_all_score_label
from explain.runner.run_test_eval_acc import load_predictions, get_best_cut


def paired_p_test_map_p1(metric, dir_path, run_name_A, run_name_B, target_label):
    gold_list = read_gold_label("gold_{}_100_700.csv".format(target_label))
    only_prem = True if target_label == 'match' else False
    only_hypo = True if target_label == 'mismatch' else False

    predictions_list = []
    for run_name in [run_name_A, run_name_B]:
        _, predictions = load_predictions(dir_path, run_name, target_label)
        predictions_list.append(predictions)

    def p_at_1(pred, gold):
        return p_at_k_list_inner(pred, gold, 1)

    if metric == "P1":
        p_value = paired_p_test(p_at_1, predictions_list[0], predictions_list[1], gold_list, only_prem, only_hypo)
    elif metric == "MAP":
        p_value = paired_p_test(MAP_inner, predictions_list[0], predictions_list[1], gold_list, only_prem, only_hypo)
    else:
        assert False

    print(metric, run_name_A, run_name_B, target_label, p_value < 0.01 , p_value)


def paired_p_test_acc(dir_path, run_name_A, run_name_B, target_label):
    dev_predictions = {}
    test_predictions = {}
    for run_name in [run_name_A, run_name_B]:
        dev_predictions[run_name], test_predictions[run_name] = load_predictions(dir_path, run_name, target_label)

    cut = get_best_cut(dev_predictions, target_label)

    gold_list = read_gold_label("gold_{}_100_700.csv".format(target_label))
    only_prem = True if target_label == 'match' else False
    only_hypo = True if target_label == 'mismatch' else False

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
        return acc_p, acc_h

    def acc_compare_inner(cut, pred):
        res = []
        for score, e in pred:
            d = score > cut
            res.append(int(d == e))
        return res

    p_acc = paired_p_test2(partial(acc_compare, cut[run_name_A]),
                           partial(acc_compare, cut[run_name_B]),
                           test_predictions[run_name_A],
                           test_predictions[run_name_B],
                           gold_list, only_prem, only_hypo)
    print("Accuracy", run_name_A, run_name_B, target_label, p_acc < 0.01, p_acc)


parser = argparse.ArgumentParser(description='File should be stored in ')
parser.add_argument("--metric", help="Your input file.")
parser.add_argument("--dir_path")
parser.add_argument("--run_name_A")
parser.add_argument("--run_name_B")
parser.add_argument("--target_label")

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    metric = args.metric
    if metric == "MAP":
        paired_p_test_map_p1(metric, args.dir_path, args.run_name_A, args.run_name_B, args.target_label)
    elif metric == "P1":
        paired_p_test_map_p1(metric, args.dir_path, args.run_name_A, args.run_name_B, args.target_label)
    elif metric == "accuracy":
        paired_p_test_acc(args.dir_path, args.run_name_A, args.run_name_B, args.target_label)
    else:
        print(metric)
        assert False
