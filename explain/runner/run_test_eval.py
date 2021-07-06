import argparse
import os
import pickle
import sys

from data_generator.NLI.nli import read_gold_label
from explain.eval_all import eval_nli_explain

parser = argparse.ArgumentParser(description='File should be stored in ')
parser.add_argument("--tag", help="Your input file.")
parser.add_argument("--run_name_list")
parser.add_argument("--dir_path")


def load_predictions_corresponding_to_test_data(file_path):
    labeled_range = (100, 700)
    st, ed = labeled_range
    all_predictions = pickle.load(open(file_path, "rb"))
    predictions = all_predictions[st:ed]
    return predictions


def run_test_eval_tois(target_label, dir_path, run_name_list_str):
    labeled_range = (100, 700)
    st, ed = labeled_range
    gold_list = read_gold_label("gold_{}_{}_{}.csv".format(target_label, st, ed))
    pred_id = "{}_1000".format(target_label)
    all_run_names = run_name_list_str.split(",")
    print(target_label)

    def get_path(run_name):
        file_name = "pred_{}_{}.pickle".format(run_name, pred_id)
        return os.path.join(dir_path, file_name)

    for run_name in all_run_names:
        file_path = get_path(run_name)
        predictions = load_predictions_corresponding_to_test_data(file_path)
        if target_label == 'conflict':
            only_prem = False
            only_hypo = False
        elif target_label == 'match':
            only_prem = True
            only_hypo = False
        elif target_label == 'mismatch':
            only_prem = False
            only_hypo = True
        else:
            assert False

        scores = eval_nli_explain(predictions, gold_list, only_prem, only_hypo)
        print(run_name, end="\t")
        for key in scores:
            print("{}".format(scores[key]), end="\t")
        print()

    print("\t", end="")
    for key in scores:
        print("{}".format(key), end="\t")
    print()


def run(args):
    run_test_eval_tois(args.tag, args.dir_path, args.run_name_list)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    run(args)