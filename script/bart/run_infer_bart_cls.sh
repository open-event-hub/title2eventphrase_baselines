test_data=$1

python ../infer_cls.py \
--model_path fnlp/bart-large-chinese \
--with_prompt 0 \
--weights_path ../../saved_weights/base_cls/bart/model_epoch_5.pt \
--batch_size 64 \
--data_dir data \
--model_arch bart \
--beam_size 4 \
--no_repeat_ngram_size 3 \
--test_data ${test_data}
