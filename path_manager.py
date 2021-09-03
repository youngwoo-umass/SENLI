import os
from os.path import dirname

pjoin = os.path.join
project_root = os.path.abspath(dirname((os.path.abspath(__file__))))
data_path = pjoin(project_root, 'data')
bert_voca_path = pjoin(data_path, "bert_voca.txt")
mnli_dir = pjoin(data_path, "MNLI")
mnli_ex_dir = pjoin(data_path, "mnli_ex")
bert_model_folder = pjoin(data_path, "uncased_L-12_H-768_A-12")
