cd bert

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --exit_stage=3 --task_name=mrpc

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --exit_stage=3 --task_name=cola

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --exit_stage=3 --task_name=rte

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --exit_stage=3 --task_name=stsb

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --exit_stage=3 --task_name=sst2

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --exit_stage=3 --task_name=qnli

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --exit_stage=3 --task_name=mnli

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --exit_stage=3 --task_name=qqp
