import sys

from explain.runner.nli_ex_param import ex_arg_parser
from explain.runner.train_ex import train_from

ex_arg_parser.add_argument("--tag")

if __name__  == "__main__":
    args = ex_arg_parser.parse_args(sys.argv[1:])
    train_from(args.start_model_path,
               args.start_type,
               args.save_dir,
               args.modeling_option,
               [args.tag],
               args.info_fn,
               args.num_deletion,
               args.num_gpu)
