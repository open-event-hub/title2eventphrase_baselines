import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import numpy as np
import torch
import jieba
import json
import random
import argparse
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BartForConditionalGeneration
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW, get_scheduler
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",default='fnlp/bart-large-chinese',type=str)
parser.add_argument("--with_prompt",default=0, type=int)
parser.add_argument("--weights_path", default='output/cl_prompt/bart/model_epoch.pt', type=str)
parser.add_argument("--batch_size",default=8,type=int)
parser.add_argument("--data_dir",default="data",type=str)
parser.add_argument("--model_arch",default="bart",type=str)
parser.add_argument("--beam_size",default=4,type=int)
parser.add_argument("--no_repeat_ngram_size",default=3,type=int)
parser.add_argument("--test_data",default="",type=str)
args = parser.parse_args()
arg_dict=args.__dict__

max_input_length = 80
max_target_length = 80
test_data_path = arg_dict["test_data"]
output_file = "{}.{}_base_inferd".format(test_data_path, arg_dict['model_arch'])

with_prompt = int(args.with_prompt)
batch_size = arg_dict['batch_size']
beam_size = arg_dict['beam_size']
no_repeat_ngram_size = arg_dict['no_repeat_ngram_size']

arc = arg_dict['model_arch']
model_path = arg_dict['model_path']
weights_path = arg_dict['weights_path']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if  arc == 'bart':
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    eos_id = tokenizer.sep_token_id
else:
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    eos_id = tokenizer.eos_token_id
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model = model.to(device)

class NEWS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data_json = json.loads(line)
                Data[idx] = {
                    'title': data_json['title'],
                    'event': data_json['event'],
                    'keyword':data_json.get("keyword", "")
                }
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

test_data = NEWS(test_data_path)

def build_template(keyword):
    return "关于'{}'的短语是：".format(keyword)

prefix = "summarize: " if  args.model_arch == 'mt5' else ""
def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        title = sample['title']
        summary = sample['event']
        keyword = sample['keyword']
        if with_prompt > 0:
            batch_inputs.append(prefix + title+build_template(keyword))
        else:
            batch_inputs.append(prefix + title)
        batch_targets.append(summary)
        batch_targets.append(summary)
    batch_data = tokenizer(
        batch_inputs, 
        padding=True, 
        max_length=max_input_length,
        truncation=True, 
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets, 
            padding=True, 
            max_length=max_target_length,
            truncation=True, 
            return_tensors="pt"
        )["input_ids"]
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == eos_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data


test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)

def predict(dataloader, model):
    preds = []
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_target_length,
                num_beams=beam_size,
                no_repeat_ngram_size=no_repeat_ngram_size,
            ).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # preds += [' '.join(pred.strip()) for pred in decoded_preds]
        preds += [''.join(pred.strip()) for pred in decoded_preds]
    return preds

preds = predict(test_dataloader, model)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(preds))
