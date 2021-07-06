import os

from cpath import data_path

num_classes = 3
corpus_dir = os.path.join(data_path, "nli")
tags = ["conflict", "match", "mismatch"]