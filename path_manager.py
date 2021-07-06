import os
from os.path import dirname

from misc_lib import exist_or_mkdir
pjoin = os.path.join

project_root = os.path.abspath(dirname((os.path.abspath(__file__))))
data_path = pjoin(project_root, 'data')
bert_voca_path = pjoin(data_path, "bert_voca.txt")
mnli_dir = pjoin(data_path, "mnli")

bert_model_folder = "c:\\work\\code\\Chair\\output\\model\\runs\\uncased_L-12_H-768_A-12"
