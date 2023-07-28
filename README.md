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
 
# Citation
```
@inproceedings{zhang-etal-2023-event,
    title = "Event-Centric Query Expansion in Web Search",
    author = "Zhang, Yanan  and
      Cui, Weijie  and
      Zhang, Yangfan  and
      Bai, Xiaoling  and
      Zhang, Zhe  and
      Ma, Jin  and
      Chen, Xiang  and
      Zhou, Tianhua",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 5: Industry Track)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-industry.45",
    pages = "464--475",
    abstract = "In search engines, query expansion (QE) is a crucial technique to improve search experience. Previous studies often rely on long-term search log mining, which leads to slow updates and is sub-optimal for time-sensitive news searches. In this work, we present Event-Centric Query Expansion (EQE), the QE system used in a famous Chinese search engine. EQE utilizes a novel event retrieval framework that consists of four stages, i.e., event collection, event reformulation, semantic retrieval and online ranking, which can select the best expansion from a significant amount of potential events rapidly and accurately. Specifically, we first collect and filter news headlines from websites. Then we propose a generation model that incorporates contrastive learning and prompt-tuning techniques to reformulate these headlines to concise candidates. Additionally, we fine-tune a dual-tower semantic model to serve as an encoder for event retrieval and explore a two-stage contrastive training approach to enhance the accuracy of event retrieval. Finally, we rank the retrieved events and select the optimal one as QE, which is then used to improve the retrieval of event-related documents. Through offline analysis and online A/B testing, we observed that the EQE system has significantly improved many indicators compared to the baseline. The system has been deployed in a real production environment and serves hundreds of millions of users.",
}
```
