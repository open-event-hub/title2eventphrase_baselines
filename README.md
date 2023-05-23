## Title2EventPhrase

目录： `./title2event/Title2EventPhrase.txt

包含字段：
- id: 样本id
- title: 网页标题，生成输入
- event: 核心事件，生成目标
- topic: 事件主题

## 生产keyword

`sh run_keyword.sh ${filepath}`

## 模型训练

BART模型：
- BART：`sh run_finetune_bart.sh`
- BART + CL (contrastive learning) ：`sh run_finetune_bart_cls.sh`
- BART + PG (prompt guidance)：`sh run_finetune_bart_prompt.sh`
- BART + CL + PG：`sh run_finetune_bart_prompt_cls.sh`
mT5模型：
- mT5`sh run_finetune_mt5.sh`
- mT5 + CL (contrastive learning) ：`sh run_finetune_mt5_cls.sh`
- mT5 + PG (prompt guidance)：`sh run_finetune_mt5_prompt.sh`
- mT5 + CL + PG：`sh run_finetune_mt5_prompt_cls.sh`

## 生成推理

`sh run_infer_*.sh ${test_path}`