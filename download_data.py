
import os
import sys
import shutil
import argparse
import tempfile
import urllib
import io

from misc_lib import exist_or_mkdir
from path_manager import data_path

if sys.version_info >= (3, 0):
    import urllib.request
import zipfile

URLLIB=urllib
if sys.version_info >= (3, 0):
    URLLIB=urllib.request


def download_and_extract_mnli(data_dir):
    url = 'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip'
    if os.path.exists(os.path.join(data_dir, "MNLI")):
        print("MNLI folder found. Skip download")
        return 
    print("Downloading and extracting MNLI dataset...")
    zip_save_path = os.path.join(data_dir, "mnli.zip")
    URLLIB.urlretrieve(url, zip_save_path)
    with zipfile.ZipFile(zip_save_path) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(zip_save_path)
    

def download_and_extract_bert(data_dir):
    url = "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip"
    if os.path.exists(os.path.join(data_dir, "bert_model.ckpt.index")):
        print("BERT model found. Skip download")
        return 
    print("Downloading and extracting BERT checkpoint...")
    zip_save_path = os.path.join(data_dir, "bert.zip")
    URLLIB.urlretrieve(url, zip_save_path)
    with zipfile.ZipFile(zip_save_path) as zip_ref:
        zip_ref.extractall(data_dir)


def main():
    exist_or_mkdir(data_path)
    download_and_extract_mnli(data_path)
    download_and_extract_bert(data_path)


if __name__ == '__main__':
    main()
