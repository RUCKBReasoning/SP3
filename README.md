CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --train_student --mix_compactor

CUDA_VISIBLE_DEVICES=1 python main.py --output_dir ../cache/checkpoints/bert --log_level info --mix_compactor

CUDA_VISIBLE_DEVICES=1 python main.py --output_dir ../cache/checkpoints/bert --log_level info --train_teacher --init_compactor --train_student --mix_compactor --task_name=