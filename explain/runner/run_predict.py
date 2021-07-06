import sys

from attribution.eval import predict_translate
from cache import save_to_pickle
from data_generator.NLI.nli import get_modified_data_loader2
from data_generator.shared_setting import BertNLI
from explain.nli_ex_predictor import NLIExPredictor
from explain.runner.predict_params import parser
from transformer import hyperparams
from trainer.np_modules import get_batches_ex


def predict_nli_ex(hparam, nli_setting, data_loader,
                   explain_tag, data_id, model_path, run_name, modeling_option):
    print("predict_nli_ex")
    print("Modeling option: ", modeling_option)
    enc_payload, plain_payload = data_loader.load_plain_text(data_id)
    batches = get_batches_ex(enc_payload, hparam.batch_size, 3)

    predictor = NLIExPredictor(hparam, nli_setting, model_path, modeling_option)
    ex_logits = predictor.predict_ex(explain_tag, batches)
    pred_list = predict_translate(ex_logits, data_loader, enc_payload, plain_payload)
    save_to_pickle(pred_list, "pred_{}_{}".format(run_name, data_id))


def run(args):
    hp = hyperparams.HPTrain()
    nli_setting = BertNLI()
    data_loader = get_modified_data_loader2(hp)

    predict_nli_ex(hp, nli_setting, data_loader,
                         args.tag,
                         args.data_id,
                         args.model_path,
                         args.run_name,
                         args.modeling_option,
                         )

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    run(args)