python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 ../model_finetune.py \
--model_path  google/mt5-base \
--with_prompt 1 \
--with_cl 1 \
--lr 2e-5 \
--batch_size 16 \
--epoch 5 \
--data_dir ../../train_data/train_data_demo.txt.keywords \
--model_arch mt5 \
--output_path ../../saved_weights