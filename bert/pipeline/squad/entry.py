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
from transformers import (
    BertTokenizerFast as Tokenizer
)
from transformers.models.bert.modeling_bert import (
    BertConfig as Config,
    BertForQuestionAnswering as TModel,
)
from modeling.modeling_compact_bert import (
    CompactBertForQuestionAnswering as SModel
)
from modeling.modeling_packed_bert import (
    PackedBertForQuestionAnswering as PModel
)
from modeling.initializer import CompactorInitializer
from modeling.mixer import CompactorMixer, get_packed_bert_config
from datasets import DatasetDict, Dataset, load_from_disk, load_metric
from typing import Optional, Dict, List, Tuple, Callable, Union
import logging

from .trainer import DefaultTrainer, DistillTrainer
from .utils import (
    BertSquadUtils
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
            )
        )
    except:
        raise ValueError('dataset [{}] does not exist.'.format(
                training_args.dataset_name))
        
    tokenizer = Tokenizer.from_pretrained(args.model_name, use_fast=args.use_fast)
    config = Config.from_pretrained(args.model_name)
    
    utils = BertSquadUtils()
    train_map_fn = utils.get_train_map_fn(training_args, tokenizer)
    validation_map_fn = utils.get_validation_map_fn(training_args, tokenizer)
    
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        raw_datasets["train"] = raw_datasets["train"].map(
            train_map_fn,
            batched=True,
            remove_columns=utils.column_names,
            desc="Running tokenizer on train dataset",
        )
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        raw_datasets["validation_examples"] = raw_datasets["validation"]
        raw_datasets["validation"] = raw_datasets["validation"].map(
            validation_map_fn,
            batched=True,
            remove_columns=utils.column_names,
            desc="Running tokenizer on validation dataset",
        )
    
    # metric = evaluate.load(args.dataset, args.task_name)
    metric = load_metric("metric/" + training_args.dataset_name + ".py")
    
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    
    post_processing_function = utils.get_post_processing_fn(training_args)
    
    return raw_datasets, tokenizer, config, post_processing_function, compute_metrics


def evaluate(
    training_args,
    datasets: DatasetDict,
    evaluator: Trainers,
    metric_name: str = "eval"
):
    # Evaluation
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = datasets["validation"]
    eval_examples = datasets["validation_examples"]

    metrics = evaluator.evaluate(eval_dataset=eval_dataset, eval_examples=eval_examples)
    metrics["eval_samples"] = len(eval_dataset)

    evaluator.log_metrics(metric_name, metrics)
    evaluator.save_metrics(metric_name, metrics)


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
    datasets, tokenizer, config, post_processing_function, compute_metrics = prepare_dataset(args, training_args)

    data_collator = DataCollatorWithPadding(tokenizer)
    
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    eval_examples = datasets["validation_examples"]

    # 1. Train teacher model
    if training_args.exit_stage == 1:
        exit(0)

    if training_args.train_teacher:
        t_model: TModel = TModel.from_pretrained(args.model_name, config=config)

        trainer = DefaultTrainer(
            t_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_examples=eval_examples,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=post_processing_function,
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
                sample_token_size=training_args.sample_token_size,
                sample_data_size=training_args.sample_data_size,
            ).initialize()
            torch.save(s_model.state_dict(), s_model_path)
        else:
            s_model.load_state_dict(torch.load(s_model_path, map_location="cpu"), strict=False)
    else:
        s_model.load_state_dict(t_model.state_dict(), strict=False)
    
    s_output_dir = "{}-[{:.2f}]".format(
        training_args.output_dir,
        training_args.target_sparsity
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
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
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

    p_model_path = os.path.join(s_output_dir, "pmodel.bin")
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
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )
    evaluate(training_args, datasets, evaluator, "eval_pmodel")

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
