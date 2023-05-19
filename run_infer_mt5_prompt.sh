test_data=$1

python infer_prompt.py \
--model_path PLM/mT5 \
--weights_path output/base_prompt/mt5/model_epoch_5.pt \
--batch_size 1 \
--data_dir data \
--model_arch mt5 \
--beam_size 4 \
--with_cls 0 \
--no_repeat_ngram_size 3 \
--test_data ${test_data}
