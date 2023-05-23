python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 model_finetune.py \
--model_path PLM/mT5 \
--with_prompt 0 \
--with_cl 0 \
--lr 2e-5 \
--batch_size 32 \
--epoch 5 \
--data_dir train_data \
--model_arch mt5
