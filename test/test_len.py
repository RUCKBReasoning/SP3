import os
from datasets import load_from_disk
from transformers import (
    BertTokenizer,
)

os.environ["TRANSFORMERS_CACHE"] = "/data2/hyx/projects/tcsp_v2/cache/models"

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

    def get_map_fn(self, task_name: str, tokenizer: BertTokenizer):
        sentence1_key, sentence2_key = self.gelu_task_to_keys[task_name]

        def map_fn(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, truncation=True)
            return result

        return map_fn

utils = BertGlueUtils()
tok = BertTokenizer.from_pretrained('bert-base-uncased')
dataset_dir = "/data2/hyx/projects/tcsp_v2/cache/datasets"
dataset_name = "glue"

for task_name in utils.gelu_task_to_keys.keys():

    raw_datasets = load_from_disk(
        os.path.join(
            dataset_dir, 
            dataset_name, 
            task_name
        )
    )


    map_fn = utils.get_map_fn(task_name, tok)

    datasets = raw_datasets.map(
        map_fn,
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

    max_len = 0
    for dataset in datasets.values():
        for sample in dataset:
            max_len = max(max_len, len(sample["input_ids"]))

    print("task_name={}, max_len={}".format(task_name, max_len))

"""
cola -> 47
mnli -> 444
mrpc -> 103
qnli -> 512
qqp -> 330
rte -> 289
sst -> 66
stsb -> 125
wnli -> 108
"""
