import os
import random
import numpy as np
import torch
import datetime
import argparse
from transformers import BertConfig, BertModel
from modeling.modeling_compact_bert import CompactBertForSequenceClassification as Model

from pipeline.glue.entry import run as run_glue
from pipeline.glue.perf_entry import run as run_glue_perf
from pipeline.squad.entry import run as run_squad
from pipeline.squad.perf_entry import run as run_squad_perf

# not support data-parallel right now

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="../cache/models")
    parser.add_argument('--dataset_dir', type=str, default="../cache/datasets")
    parser.add_argument('--dataset_name', type=str, default="glue")
    parser.add_argument('--task_name', type=str, default="sst2")
    parser.add_argument('--do_perf', action="store_true")

    return parser.parse_known_args()[0]


def setup_cache(args):
    os.environ["TRANSFORMERS_CACHE"] = args.model_dir


def main():
    args = parse_args()
    
    setup_cache(args)
    
    if not args.do_perf:
        if args.dataset_name == "glue":
            run_glue()
        elif args.dataset_name == "squad" or args.dataset_name == "squad_v2":
            run_squad()
        else:
            raise ValueError
    else:
        if args.dataset_name == "glue":
            run_glue_perf()
        elif args.dataset_name == "squad" or args.dataset_name == "squad_v2":
            run_squad_perf()
        else:
            raise ValueError        


if __name__ == '__main__':
    main()
