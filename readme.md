# Explaining Text Matching on Neural Natural Language
Inference

This is the implementation for the model that was proposed 
in our paper "[Explaining Text Matching on Neural Natural Language](https://dl.acm.org/doi/abs/10.1145/3418052)", which is published in ACM Transactions on Information Systems (September 2020).


## Datasets

This code use GLUE version of MultiNLI dataset 
and our own annotation for the token-level explanation.

* MultiNLI dataset can be downloaded with the following command.
```
python dev/download_mnli.py
```
* The annotation for the token-level explanation is included in this repository.
* For the training, you need to separately download [BERT-Base](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip)
and set the checkpoint path in the `bert_model_folder` variable in `path_manager.py`.
  
## Training

```angular2html
PYTHONPATH=. python3 -u dev/run_train_nli_ex.py \
    --init_checkpoint=saved_model/run1 \
    --checkpoint_type=nli_saved_model \
    --model_save_path=saved_model/ex_run5 
PYTHONPATH=. python3 -u dev/run_ex_eval.py \
    saved_model/ex_run5
```

## Models


## Reference

```angular2html
@article{10.1145/3418052,
author = {Kim, Youngwoo and Jang, Myungha and Allan, James},
title = {Explaining Text Matching on Neural Natural Language Inference},
year = {2020},
issue_date = {October 2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {38},
number = {4},
issn = {1046-8188},
url = {https://doi.org/10.1145/3418052},
doi = {10.1145/3418052},
month = sep,
articleno = {39},
numpages = {23},
keywords = {interpretable machine learning, rationale, neural network explanation, Natural language inference}
}
```