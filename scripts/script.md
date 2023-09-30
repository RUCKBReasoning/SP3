# GLUE 

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --train_student --mix_compactor --task_name=...

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --train_student --mix_compactor --task_name=... --target_sparsity=0.06 --use_structural_pruning 

# SQuAD

CUDA_VISIBLE_DEVICES=2 python main.py --output_dir ../cache/checkpoints/bert --log_level info --train_student --mix_compactor --dataset_name=squad

CUDA_VISIBLE_DEVICES=3 python main.py --output_dir ../cache/checkpoints/bert --log_level info --train_student --mix_compactor --dataset_name=squad --target_sparsity=0.06 --use_structural_pruning

# TMP

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --train_student --mix_compactor --task_name=rte --target_sparsity=0.06 --use_structural_pruning --per_device_train_batch_size=32 --learning_rate=1e-5
