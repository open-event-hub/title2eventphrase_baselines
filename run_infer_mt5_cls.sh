test_data=$1

python infer_cls.py \
--model_path PLM/mT5 \
--with_prompt 0 \
--weights_path output/base_cls/mt5/model_epoch_5.pt \
--batch_size 64 \
--data_dir data \
--model_arch mt5 \
--beam_size 4 \
--no_repeat_ngram_size 3 \
--test_data ${test_data}
