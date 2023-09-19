import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import transformers
from transformers import (
    HfArgumentParser,
    EvalPrediction,
    DataCollatorWithPadding,
)
from transformers.models.bert.tokenization_bert_fast import (
    BertTokenizer as Tokenizer
)
from transformers.models.bert.modeling_bert import (
    BertConfig as Config,
    BertForSequenceClassification as TModel,
)
from modeling.modeling_compact_bert import (
    CompactBertForSequenceClassification as SModel
)
from modeling.modeling_packed_bert import (
    PackedBertForSequenceClassification as PModel
)
from modeling.initializer import CompactorInitializer
from modeling.mixer import CompactorMixer, get_packed_bert_config
from datasets import DatasetDict, Dataset, load_from_disk, load_metric
from typing import Optional, Dict, List, Tuple, Callable, Union
import logging

from .trainer import DefaultTrainer, DistillTrainer
from .utils import (
    BertGlueUtils
)
from .args import (
    TrainingArguments,
    ModelArguments,
)
from copy import deepcopy

Trainers = Union[DefaultTrainer, DistillTrainer]

logger = logging.getLogger(__name__)

def parse_hf_args():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    args, training_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)

    return args, training_args


def setup_seed(training_args):
    seed: int = training_args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_logger(training_args):

    transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level
    )


def get_distill_args(args):
    distill_args = deepcopy(args)
    distill_args.num_train_epochs = args.distill_num_train_epochs
    distill_args.learning_rate = args.distill_learning_rate
    distill_args.evaluation_strategy = "epoch"
    
    return distill_args


def prepare_dataset(
    args,
    training_args,
) -> Tuple[DatasetDict, Tokenizer, Config, Callable]:
    try:
        raw_datasets = load_from_disk(
            os.path.join(
                training_args.dataset_dir, 
                training_args.dataset_name, 
                training_args.task_name
            )
        )
    except:
        raise ValueError('dataset [{}] does not exist.'.format(
                training_args.dataset_name))

    is_regression = training_args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    
    tokenizer = Tokenizer.from_pretrained(args.model_name, use_fast=args.use_fast)
    config = Config.from_pretrained(args.model_name, num_labels=num_labels)

    utils = BertGlueUtils()
    preprocess_fn = utils.get_map_fn(training_args, tokenizer)
    
    with training_args.main_process_first(desc="dataset map pre-processing"):
        datasets = raw_datasets.map(
            preprocess_fn,
            batched=True,
            desc="Running tokenizer on dataset",
        )
    for key, dataset in datasets.items():
        datasets[key] = dataset.remove_columns(
            set(dataset.column_names) - set(utils.column_names)
        )
    del_keys = [key for key in datasets.keys() if "test" in key]
    for key in del_keys:
        del datasets[key]
    
    # metric = evaluate.load(args.dataset, args.task_name)
    metric = load_metric("metric/" + training_args.dataset_name + ".py", training_args.task_name)
    
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    
    return datasets, tokenizer, config, compute_metrics

@torch.no_grad()
def evaluate(
    training_args,
    datasets: DatasetDict,
    evaluator: Trainers,
    metric_name: str = "eval"
):
    # Evaluation
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = datasets["validation_matched" if training_args.task_name == "mnli" else "validation"]
    tasks = [training_args.task_name]
    eval_datasets = [eval_dataset]
    if training_args.task_name == "mnli":
        tasks.append("mnli-mm")
        valid_mm_dataset = datasets["validation_mismatched"]
        eval_datasets.append(valid_mm_dataset)
        combined = {}

    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = evaluator.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)

        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if task is not None and "mnli" in task:
            combined.update(metrics)

        evaluator.log_metrics(metric_name, metrics)
        evaluator.save_metrics(metric_name, combined if task is not None and "mnli" in task else metrics)


def get_num_params(model: nn.Module):
    num_params = 0
    num_params_without_residual = 0
    for name, params in model.named_parameters():
        if 'encoder' in name:
            num_params += params.view(-1).shape[0]
            if 'residual' not in name:
                num_params_without_residual += params.view(-1).shape[0]
    return num_params, num_params_without_residual


def run():
    args, training_args = parse_hf_args()
    
    setup_seed(training_args)
    setup_logger(training_args)
    datasets, tokenizer, config, compute_metrics = prepare_dataset(args, training_args)

    data_collator = DataCollatorWithPadding(tokenizer)
    
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation_matched" if training_args.task_name == "mnli" else "validation"]

    # 1. Train teacher model
    if training_args.exit_stage == 1:
        exit(0)

    if training_args.train_teacher:
        t_model: TModel = TModel.from_pretrained(args.model_name, config=config)

        trainer = DefaultTrainer(
            t_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        evaluate(training_args, datasets, trainer)
    else:
        if not os.path.exists(training_args.output_dir):
            raise ValueError('checkpoints does not exist.')
        t_model: TModel = TModel.from_pretrained(training_args.output_dir, config=config)

    # 2. Init compactor
    if training_args.exit_stage == 2:
        exit(0)
     
    s_model_path = os.path.join(training_args.output_dir, "smodel_init.bin")
    s_model = SModel(config)
    if not training_args.skip_init_compactor:
        if training_args.init_compactor:
            s_model.load_state_dict(t_model.state_dict(), strict=False)
            train_dataset = datasets["train"]
            CompactorInitializer(
                s_model,
                train_dataset,
                data_collator,
            ).initialize()
            torch.save(s_model.state_dict(), s_model_path)
        else:
            s_model.load_state_dict(torch.load(s_model_path, map_location="cpu"), strict=False)
    else:
        s_model.load_state_dict(t_model.state_dict(), strict=False)

    s_output_dir = "{}-[{:.2f}]".format(
        training_args.output_dir,
        training_args.structural_target_sparsity
    )
    os.makedirs(s_output_dir, exist_ok=True)
    
    distill_args = get_distill_args(training_args)
    distill_args.output_dir = s_output_dir
    distill_trainer = DistillTrainer(
        s_model,
        t_model,
        args=distill_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,            
    )
    
    if training_args.init_compactor:
        evaluate(training_args, datasets, distill_trainer, "eval_smodel_init")
    
    # 3. Train Student with Knowledge Distill
    if training_args.exit_stage == 3:
        exit(0)
    
    s_model_path = os.path.join(s_output_dir, "smodel_distill.bin")
    if training_args.train_student:
        
        train_result = distill_trainer.train()
        metrics = train_result.metrics
        metrics["distill_samples"] = len(train_dataset)    

        distill_trainer.log_metrics("distill", metrics)
        distill_trainer.save_metrics("distill", metrics)
        
        if training_args.local_rank == 0:
            torch.save(s_model.state_dict(), s_model_path)
        evaluate(training_args, datasets, distill_trainer, "eval_smodel_distill")
    else:
        s_model.load_state_dict(torch.load(s_model_path, map_location="cpu"), strict=False)
    
    # 4. Mix compactor
    if training_args.exit_stage == 4:
        exit(0)

    p_model_path = os.path.join(s_output_dir, "pmodel_init.bin")
    p_model_config_path = os.path.join(s_output_dir, "pmodel_config.bin")
    if training_args.mix_compactor:
        p_model: PModel = CompactorMixer(
            training_args,
            s_model,
            PModel,
        ).mix()
        if training_args.local_rank == 0:
            torch.save(p_model.config, p_model_config_path)
            torch.save(p_model.state_dict(), p_model_path)
    else:
        p_config = torch.load(p_model_config_path)
        p_model = PModel(p_config)
        p_model.load_state_dict(torch.load(p_model_path, map_location="cpu"))
    
    p_model.config.per_layer_config = None
    training_args.num_train_epochs = 1.0
    training_args.output_dir = s_output_dir
    evaluator = DefaultTrainer(
        p_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    evaluate(training_args, datasets, evaluator, "eval_pmodel_init")

    p_params, p_params_without_residual = get_num_params(p_model)
    t_params, _ = get_num_params(t_model)
    if training_args.local_rank == 0:
        logger.info("p-params = {:.3f}".format(p_params))
        logger.info("p-params-without-residual = {:.3f}".format(p_params_without_residual))
        logger.info("t-params = {:.3f}".format(t_params))
        logger.info("real sparsity = {:.3f}".format(p_params / t_params))
        logger.info("real sparsity without residual = {:.3f}".format(p_params_without_residual / t_params))
    evaluator.save_metrics("sparsity", {
        "num_params": p_params,
        "num_params_without_residual": p_params_without_residual,
        "sparsity": p_params / t_params,
        "sparsity__without_residual": p_params_without_residual / t_params,
    })
