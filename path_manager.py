import os
from os.path import dirname

pjoin = os.path.join
project_root = os.path.abspath(dirname((os.path.abspath(__file__))))
data_path = pjoin(project_root, 'data')
bert_voca_path = pjoin(data_path, "vocab.txt")
mnli_dir = pjoin(data_path, "MNLI")
mnli_ex_dir = pjoin(data_path, "mnli_ex")
bert_model_folder = data_path
