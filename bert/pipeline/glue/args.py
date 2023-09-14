import os
from dataclasses import dataclass, field
from transformers import TrainingArguments as DefaultTrainingArguments
from transformers.training_args import (
    IntervalStrategy
)
from typing import Optional, Union


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
        default="bert-base-uncased"
    )
    use_fast: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    

@dataclass
class TrainingArguments(DefaultTrainingArguments):
    
    # Other
    exit_stage: int = field(default=5) # 4 stages
    skip_init_compactor: Optional[bool] = field(default=False)

    # Data
    dataset_dir: Optional[str] = field(default="../cache/datasets")
    
    dataset_name: Optional[str] = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
        default="glue",
    )
    task_name: Optional[str] = field(
        metadata={"help": "The name of the task to train on."},
        default="mrpc",
    )
    
    # Training
    
    train_teacher: Optional[bool] = field(default=False)
    init_compactor: Optional[bool] = field(default=False)
    train_student: Optional[bool] = field(default=False)
    mix_compactor: Optional[bool] = field(default=False)

    target_sparsity: Optional[float] = field(default=0.1)
    structural_target_sparsity: Optional[float] = field(default=0.05)
    # sparsity = (new params number) / (origin params number)
    
    distill_T: float = field(default=2.0)
    distill_lambda: float = field(default=0.3)  # lambda * loss_pred + (1 - lambda) * loss_layer
    
    reg_learning_rate: float = field(default=1e-1)
    
    distill_num_train_epochs: float = field(default=20.0, metadata={"help": "Total number of training epochs to perform."})
    distill_learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW."})
    
    pruning_start_epoch: int = field(default=2)
    structural_pruning_start_epoch: int = field(default=6)

    pruning_warmup_epoch: int = field(default=2)

    # Overwrite 
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW."})

    save_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    
    def get_file_name(self):
        return "[{}]-[{}]".format(
            self.dataset_name,
            self.task_name,
        )
    
    def __post_init__(self):
        # update output dir
        self.output_dir = os.path.join(self.output_dir, self.get_file_name())
        super().__post_init__()
