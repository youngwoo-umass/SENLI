import os
import pickle
import sys

from data_generator.NLI.nli import read_gold_label
from explain.eval_all import load_gold, eval_pr
from explain.eval_pr import tune_cut
from explain.runner.run_test_eval import parser


def select_dev(target_label, predictions):
    if target_label == 'match':
        predictions = predictions[:56]
    elif target_label == 'mismatch':
        predictions = predictions[:34]
    return predictions


def load_predictions(dir_path, run_name, target_label):
    data_id = "{}_1000".format(target_label)

    def get_path(run_name, pred_id):
        file_name = "pred_{}_{}.pickle".format(run_name, pred_id)
        return os.path.join(dir_path, file_name)

    all_predictions = pickle.load(open(get_path(run_name, data_id), "rb"))
    dev_predictions = select_dev(target_label, all_predictions[:100])
    test_predictions = all_predictions[100:700]
    return dev_predictions, test_predictions


def run_eval_acc(target_label, dir_path, run_name_list_str):
    all_run_names = run_name_list_str.split(",")
    gold_list = read_gold_label("gold_{}_100_700.csv".format(target_label))
    dev_predictions = {}
    test_predictions = {}
    for run_name in all_run_names:
        dev_predictions[run_name], test_predictions[run_name] = load_predictions(dir_path, run_name, target_label)

    best_cuts = get_best_cut(dev_predictions, target_label)

    for run_name, predictions in test_predictions.items():
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

    print("\t", end="")
    for key in scores:
        print("{}".format(key), end="\t")
    print()


def get_best_cut(dev_predictions, target_label):
    dev_data_id = {
        'match': 'match',
        'mismatch': 'mismatch',
        'conflict': 'conflict_0_99'
    }[target_label]
    dev_label = load_gold(dev_data_id)

    best_cuts = {}
    for run_name, predictions in dev_predictions.items():
        best_cuts[run_name] = tune_cut(predictions, dev_label)
    return best_cuts


def run(args):
    run_eval_acc(args.tag, args.dir_path, args.run_name_list)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    run(args)