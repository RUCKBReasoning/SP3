import argparse
import time
import os
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    BertConfig as Config, 
    BertForSequenceClassification as TModel, 
    default_data_collator,
)
from modeling.modeling_packed_bert import (
    PackedBertForSequenceClassification as PModel
)
from .entry import parse_hf_args, prepare_dataset
from tqdm import tqdm

BATCH_SIZE: int = 32

def warmup(args):
    time1 = time.perf_counter()
    input = torch.randn(128, 1024).to(args.device)
    linear = torch.nn.Linear(1024, 1024).to(args.device)
    for i in range(1000):
        input = linear(input)

    time2 = time.perf_counter()
    print(round(time2 - time1, 2), "seconds for warmup")
    torch.cuda.synchronize()


@torch.no_grad()
def evaluate(args, model: torch.nn.Module, dataset: Dataset):
    model.to(args.device)
    torch.cuda.synchronize()
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        sampler=SequentialSampler(dataset),
        collate_fn=default_data_collator,
        drop_last=True
    )
    
    warmup(args)

    time_cost = 0
    for inputs in tqdm(loader, total=len(loader)):
        if "labels" in inputs:
            inputs.pop("labels")
        for key in inputs.keys():
            inputs[key] = inputs[key].to(args.device)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        _ = model(**inputs)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        time_cost += end_time - start_time

    time_cost /= len(loader)
    print("time_cost: {}".format(time_cost))

    model.to("cpu")
    torch.cuda.synchronize()
    
    return time_cost


def run():
    args, training_args = parse_hf_args()
    if training_args.pad_to_max_length is False:
        training_args.pad_to_max_length = True
    datasets, tokenizer, config, post_processing_function, compute_metrics = prepare_dataset(args, training_args)
    
    t_model: TModel = TModel.from_pretrained(training_args.output_dir, config=config)
    
    s_output_dir = "{}-[{:.2f}]".format(
        training_args.output_dir,
        training_args.structural_target_sparsity
    )
    p_model_path = os.path.join(s_output_dir, "pmodel_init.bin")
    p_model_config_path = os.path.join(s_output_dir, "pmodel_config.bin")
    p_config = torch.load(p_model_config_path)
    p_model = PModel(p_config)
    p_model.load_state_dict(torch.load(p_model_path, map_location="cpu"))
    
    eval_dataset = datasets["validation"]

    time1 = evaluate(t_model, eval_dataset)
    time2 = evaluate(p_model, eval_dataset)
    
    print("speed up = {}".format(time1 / time2))
