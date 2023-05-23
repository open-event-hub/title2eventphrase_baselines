test_data=$1

python ../infer_prompt.py \
--model_path fnlp/bart-large-chinese \
--weights_path ../../saved_weights/base_cls_prompt/bart/model_epoch_5.pt \
--batch_size 1 \
--data_dir data \
--model_arch bart \
--beam_size 4 \
--with_cls 1 \
--no_repeat_ngram_size 3 \
--test_data ${test_data}
