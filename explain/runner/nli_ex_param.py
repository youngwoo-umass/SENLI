import argparse

ex_arg_parser = argparse.ArgumentParser(description='File should be stored in ')
ex_arg_parser.add_argument("--start_model_path", help="Your input file.")
ex_arg_parser.add_argument("--start_type")
ex_arg_parser.add_argument("--save_dir")
ex_arg_parser.add_argument("--modeling_option")
ex_arg_parser.add_argument("--info_fn", default="eq2")
ex_arg_parser.add_argument("--num_gpu", default=1)
ex_arg_parser.add_argument("--num_deletion", default=20)
ex_arg_parser.add_argument("--g_val", default=0.5)
ex_arg_parser.add_argument("--drop_thres", default=0.3)
ex_arg_parser.add_argument("--save_name")
