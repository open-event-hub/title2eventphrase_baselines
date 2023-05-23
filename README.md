# Title2EventPhrase
This is the repository for the paper "Event-Centric Query Expansion in Web Search".

# Quick Start
## Requirements
`pip install -r requirements.txt`
## Processing the dataset
We provide a demo of the training data:`train_data/train_data_demo.txt`,  It contains the following fields:
- id: Sample ID
- title:  Web page title, used as input
- event: Core event, used as the target
- topic: Event topic
Users can replace it with their own data, following the same format as the demo.

Use the following code for data preprocessing and extracting keywords from the title:
`cd script/preprocessing && sh run_keyword.sh ${filepath}`

# Model Training and Inference

## BART based Model Finetune：

`cd script/bart`
- BART：`sh run_finetune_bart.sh`
- BART + CL (contrastive learning) ：`sh run_finetune_bart_cls.sh`
- BART + PG (prompt guidance)：`sh run_finetune_bart_prompt.sh`
- BART + CL + PG：`sh run_finetune_bart_prompt_cls.sh`

## mT5 based Model Finetune：

`cd script/mt5`
- mT5`sh run_finetune_mt5.sh`
- mT5 + CL (contrastive learning) ：`sh run_finetune_mt5_cls.sh`
- mT5 + PG (prompt guidance)：`sh run_finetune_mt5_prompt.sh`
- mT5 + CL + PG：`sh run_finetune_mt5_prompt_cls.sh`

## Generation Inference

`sh run_infer_*.sh ${test_path}`