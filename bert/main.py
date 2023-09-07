import os
import random
import numpy as np
import torch
import datetime
import argparse
from transformers import BertConfig, BertModel
from modeling.modeling_compact_bert import CompactBertForSequenceClassification as Model

from pipeline.glue.entry import run as run_glue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="../cache/models")
    parser.add_argument('--dataset_dir', type=str, default="../cache/datasets")
    parser.add_argument('--dataset_name', type=str, default="glue")
    parser.add_argument('--task_name', type=str, default="sst2")

    return parser.parse_known_args()[0]


def setup_cache(args):
    os.environ["TRANSFORMERS_CACHE"] = args.model_dir


def main():
    args = parse_args()
    
    setup_cache(args)
    
    if args.dataset_name == "glue":
        run_glue()
    elif args.dataset_name == "squad" or args.dataset_name == "squad_v2":
        ...
    else:
        raise ValueError


if __name__ == '__main__':
    main()
