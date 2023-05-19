import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import torch
import jieba 
import random
import argparse
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BartForConditionalGeneration
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW, get_scheduler
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, BeamSearchScorer
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--model_path",default='fnlp/bart-large-chinese',type=str)
parser.add_argument("--weights_path", default='output/cl_prompt/bart/model_epoch.pt', type=str)
parser.add_argument("--batch_size",default=8,type=int)
parser.add_argument("--data_dir",default="data",type=str)
parser.add_argument("--model_arch",default="bart",type=str)
parser.add_argument("--beam_size",default=4,type=int)
parser.add_argument("--no_repeat_ngram_size",default=3,type=int)
parser.add_argument("--test_data",default="",type=str)
parser.add_argument("--with_cls",default=0,type=int)

args = parser.parse_args()
arg_dict=args.__dict__

max_input_length = 80
max_target_length = 80
test_data_path = arg_dict["test_data"]
batch_size = arg_dict['batch_size']
beam_size = arg_dict['beam_size']
no_repeat_ngram_size = arg_dict['no_repeat_ngram_size']

arc = arg_dict['model_arch']
model_path = arg_dict['model_path']
weights_path = arg_dict['weights_path']

output_file = "{}.{}_prompt_inferd".format(test_data_path, arg_dict['model_arch'])
if arg_dict["with_cls"] > 0:
    output_file = "{}.{}_cls_prompt_inferd".format(test_data_path, arg_dict['model_arch'])

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

prefix = "summarize: " if  arc == 'mt5' else ""

def collote_fn(batch_samples):
    batch_inputs, batch_prompt = [], []
    for sample in batch_samples:
        title = sample['title']
        summary = sample['event']
        keyword = sample['keyword']
        batch_inputs.append(prefix + title)
        batch_prompt.append(build_template(keyword))
            
    batch_data = tokenizer(
        batch_inputs,   
        padding=True, 
        max_length=max_input_length,
        truncation=True, 
        return_tensors="pt"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_prompt, 
            padding=True, 
            max_length=max_target_length,
            truncation=True, 
            return_tensors="pt"
        )["input_ids"]
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
    return batch_data



test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)

def predict(dataloader, model):
    preds = []
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        encoder_input_ids = batch_data["input_ids"]
        decoder_input_ids = batch_data["decoder_input_ids"]
        attention_mask = batch_data["attention_mask"]
        with torch.no_grad():
            model_kwargs = {
                "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(beam_size, dim=0), 
                attention_mask=attention_mask.repeat_interleave(beam_size, dim=0),
                return_dict=True)
            }
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=beam_size,
                device=device
            )

            logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(min_length=1, eos_token_id=eos_id)])
            outputs = model.beam_search(
                decoder_input_ids.repeat_interleave(beam_size, dim=0),
                beam_scorer,
                max_length=max_target_length,
                logits_processor=logits_processor,
                **model_kwargs,
            )
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds.extend(batch_preds)
    return preds


preds = predict(test_dataloader, model)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(preds))
