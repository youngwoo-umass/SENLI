import sys

from explain.runner.eval_accuracy import eval_original_task

if __name__  == "__main__":
    eval_original_task(sys.argv[1], sys.argv[2], "snli")