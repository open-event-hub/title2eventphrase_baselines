# Title2EventPhrase
This repository hosts the codebase for the paper titled "Event-Centric Query Expansion in Web Search".

# Quick Start
## Requirements
Install the necessary packages with the following command:
`pip install -r requirements.txt`
## Processing the dataset
We offer a demonstration of the training data: `train_data/train_data_demo.txt`,  It contains the following fields:
- id: Sample ID
- title:  Web page title, used as input
- event: Core event, used as the target
- topic: Event topic
Users can replace it with their own data, following the same format as the demo. Alternatively, you can use the dataset available on our  [webpage](https://open-event-hub.github.io/eqe/title2eventphrase/).

To preprocess the data and extract keywords from the title, use the following code:
`cd script/preprocessing && sh run_keyword.sh ${filepath}`

## Model Training and Inference

### BART based Model Finetune：

Navigate to the BART script folder with: `cd script/bart`
- BART：`sh run_finetune_bart.sh`
- BART + CL (contrastive learning) ：`sh run_finetune_bart_cls.sh`
- BART + PG (prompt guidance)：`sh run_finetune_bart_prompt.sh`
- BART + CL + PG：`sh run_finetune_bart_prompt_cls.sh`

### mT5 based Model Finetune：

Navigate to the mT5 script folder with: `cd script/mt5`
- mT5: `sh run_finetune_mt5.sh`
- mT5 + CL (contrastive learning) ：`sh run_finetune_mt5_cls.sh`
- mT5 + PG (prompt guidance)：`sh run_finetune_mt5_prompt.sh`
- mT5 + CL + PG：`sh run_finetune_mt5_prompt_cls.sh`

## Generation Inference
The test set should be formatted similarly to the training set and preprocessed with `run_keyword.sh`.

 The code for inference is as follows: `sh run_infer_*.sh ${test_path}`