import numpy as np
from transformers import (
    BertTokenizer,
)


class BertGlueUtils:

    gelu_task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    column_names = [
        'input_ids',
        'attention_mask',
        'token_type_ids',
        'position_ids',
        'head_mask',
        'inputs_embeds',
        'output_attentions',
        'output_hidden_states',
        'return_dict',
        'label',
    ]

    def get_map_fn(self, training_args, tokenizer: BertTokenizer):
        task_name = training_args.task_name
        max_length = training_args.max_seq_length
        pad_to_max_length = training_args.pad_to_max_length
        sentence1_key, sentence2_key = self.gelu_task_to_keys[task_name]

        def map_fn(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, 
                max_length=max_length, 
                truncation=True,
                padding="max_length" if pad_to_max_length else False
            )
            return result

        return map_fn

bertGlueUtils = BertGlueUtils()

