import sys

from explain.runner.nli_ex_param import ex_arg_parser
from explain.runner.train_msmarco import train_from

if __name__  == "__main__":
    args = ex_arg_parser.parse_args(sys.argv[1:])
    train_from(args.start_model_path,
               args.start_type,
               args.save_dir,
               args.modeling_option,
               args.num_deletion,
               args.g_val,
               args.num_gpu)
