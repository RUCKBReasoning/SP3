cd bert

CUDA_VISIBLE_DEVICES=2 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --exit_stage=3 --task_name=cola

CUDA_VISIBLE_DEVICES=2 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --exit_stage=3 --task_name=rte

CUDA_VISIBLE_DEVICES=2 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --exit_stage=3 --task_name=stsb