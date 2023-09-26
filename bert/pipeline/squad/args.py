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
    
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "The threshold used to select the null answer: if the best answer has a score that is less than "
                "the score of the null answer minus this threshold, the null answer is selected for this example. "
                "Only useful when `version_2_with_negative=True`."
            )
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )
    
    # Training
    target_score_field: Optional[str] = field(default=None)
    
    train_teacher: Optional[bool] = field(default=False)
    init_compactor: Optional[bool] = field(default=False)
    train_student: Optional[bool] = field(default=False)
    mix_compactor: Optional[bool] = field(default=False)

    target_sparsity: Optional[float] = field(default=0.12)
    # sparsity = (new params number) / (origin params number)
    
    distill_T: float = field(default=2.0)
    distill_lambda: float = field(default=0.3)  # lambda * loss_pred + (1 - lambda) * loss_layer
    
    reg_learning_rate: float = field(default=1e-1)
    
    distill_num_train_epochs: float = field(default=20.0, metadata={"help": "Total number of training epochs to perform."})
    distill_learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW."})
    
    pruning_start_epoch: int = field(default=2)
    pruning_warmup_epoch: int = field(default=2)

    use_structural_pruning: bool = field(default=False)
    structural_pruning_start_epoch: int = field(default=8)

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
        return "[{}]".format(
            self.dataset_name,
        )
    
    def __post_init__(self):
        # update output dir
        self.output_dir = os.path.join(self.output_dir, self.get_file_name())
        super().__post_init__()
