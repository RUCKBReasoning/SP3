CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --init_compactor --train_student --mix_compactor

CUDA_VISIBLE_DEVICES=1 python main.py --output_dir ../cache/checkpoints/bert --log_level info --mix_compactor

CUDA_VISIBLE_DEVICES=1 python main.py --output_dir ../cache/checkpoints/bert --log_level info --train_teacher --init_compactor --train_student --mix_compactor --task_name=...

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --train_teacher --exit_stage=2 --dataset_name=squad

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --train_teacher --exit_stage=2 --dataset_name=squad_v2 --version_2_with_negative


CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --train_teacher --init_compactor --train_student --mix_compactor --dataset_name=squad

CUDA_VISIBLE_DEVICES=3 python main.py --output_dir ../cache/checkpoints/bert --dataset_name=glue --task_name=mnli --do_perf


<!-- target_sparsity
structural_target_sparsity -->

CUDA_VISIBLE_DEVICES=0 python main.py --output_dir ../cache/checkpoints/bert --log_level info --train_student --mix_compactor --task_name=mrpc --target_sparsity=0.12

CUDA_VISIBLE_DEVICES=1 python main.py --output_dir ../cache/checkpoints/bert --log_level info --train_student --mix_compactor --task_name=mrpc --target_sparsity=0.06 --use_structural_pruning
