CUDA_VISIBLE_DEVICES=0 \
python3 trainAndEvaluate.py \
--batch_size 1 \
--base_lr 5e-6 \
--max_epochs 20 \
--save_path "/home/pjl307/ZZJ/CholeBleedingPoint/Cholec95-project/BlooDet/save/mytest" \
--data_path "/home/pjl307/ZZJ/CholeBleedingPoint/Cholec95-project/CholecBP95" \
--warmup \
--AdamW \

