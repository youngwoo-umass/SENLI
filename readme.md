# Explaining Text Matching on Neural Natural Language Inference

This is the implementation for the model that was proposed 
in our paper "[Explaining Text Matching on Neural Natural Language](https://dl.acm.org/doi/abs/10.1145/3418052)", which was published in ACM Transactions on Information Systems (September 2020).


## Datasets

This code uses GLUE version of MultiNLI dataset 
and our own annotation for the token-level explanation.
* The annotation for the token-level explanation is included in this repository.

* BERT checkpoint and MultiNLI dataset need to be downloaded with the following command.
```
python dev/download_data.py
```

## Training

Once the BERT checkpoint and MultiNLI dataset is downloaded you can start training with the following commands.

```angular2html
python3 -u run_train_nli_ex.py \
--init_checkpoint=data/bert_model.ckpt \
--checkpoint_type=bert \
--model_save_path=saved_model/ex_run
```

## Trained model

You can download the trained SENLI model from [here](https://drive.google.com/file/d/1c1G1sjLXh_brHnrfS-DXGcUvDOHiFA8P/view?usp=sharing).


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
