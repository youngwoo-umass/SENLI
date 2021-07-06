from trainer.model_saver import load_model_with_blacklist, load_model, load_model_w_scope


def init_fn_generic(sess, start_type, start_model_path):
    if start_type == "cls":
        load_model_with_blacklist(sess, start_model_path, ["explain", "explain_optimizer"])
    elif start_type == "cls_new":
        load_model_with_blacklist(sess, start_model_path, ["explain", "explain_optimizer", "optimizer"])
    elif start_type == "cls_ex":
        load_model(sess, start_model_path)
    elif start_type == "as_is":
        load_model(sess, start_model_path)
    elif start_type == "cls_ex_for_pairing":
        load_model_with_blacklist(sess, start_model_path, ["match_predictor", "match_optimizer"])
    elif start_type == "bert":
        load_model_w_scope(sess, start_model_path, ["bert"])
    elif start_type == "cold":
        pass
    else:
        assert False