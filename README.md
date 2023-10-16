# DA-WID

## file structure

bert: source code

cache/checkpoints: save experiment results

cache/datasets: save datasets

cache/models: save pre-trained model

cache/svd_results: save svd results

## datasets

We download the dataset to a local disk beforehand, and load the dataset via huggingface's load_from_disk API.

For example, to read the mrpc dataset from the glue benchmark, we use the following command

```python
dataset = load_from_disk('[path to cache]/datasets/glue/mrpc')
```

and to read the SQuAD dataset, we use the following command

```python
dataset = load_from_disk('[path to cache]/datasets/squad')
```

When using this code, please download the dataset in advance in the format described above, or change the load_from_disk API to one that downloads the dataset directly over the network.

## scripts

### GLUE

```bash
# task_name=stsb, sst2, mnli, qnli, ...
task_name=mrpc

cd bert

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --train_student --mix_compactor --target_sparsity=0.06 --use_structural_pruning --task_name=${task_name}

# rte: --per_device_train_batch_size=24 --learning_rate=1e-5

```

### SQuAD

```bash
cd bert

CUDA_VISIBLE_DEVICES=3 python main.py --output_dir ../cache/checkpoints/bert --log_level info --train_student --mix_compactor --dataset_name=squad --target_sparsity=0.06 --use_structural_pruning
```
