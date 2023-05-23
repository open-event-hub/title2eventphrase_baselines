import os
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",default='fnlp/bart-large-chinese',type=str)
parser.add_argument("--with_prompt",default=0, type=int)
parser.add_argument("--with_cl", default=0, type=int)
parser.add_argument("--lr",default=2e-5,type=float)
parser.add_argument("--batch_size",default=8,type=int)
parser.add_argument("--epoch",default=5,type=int)
parser.add_argument("--data_dir",default="data",type=str)
parser.add_argument("--model_arch",default="bart",type=str)
parser.add_argument("--output_path",default="output",type=str)
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
arg_dict=args.__dict__

torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)

torch.distributed.init_process_group(backend='nccl')
num_gpu = torch.cuda.device_count()

max_input_length = 80
max_target_length = 80
# train_data_path = os.path.join(args.data_dir, 'title_event_all_keywords.txt')
train_data_path = args.data_dir
epoch_num = args.epoch
learning_rate = arg_dict['lr']
batch_size = arg_dict['batch_size']
with_prompt = args.with_prompt
with_cl = int(args.with_cl) > 0
with_prompt = int(args.with_prompt) > 0
if with_prompt:
    if with_cl:
        output_dir = os.path.join(args.output_path, 'base_cls_prompt', args.model_arch)
    else:
        output_dir = os.path.join(args.output_path, 'base_prompt', args.model_arch)
else:
    if with_cl:
        output_dir = os.path.join(args.output_path, 'base_cls', args.model_arch)
    else:
        output_dir = os.path.join(args.output_path, 'base', args.model_arch)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
set_seed(42)


if args.model_arch == 'bart':
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BartForConditionalGeneration.from_pretrained(args.model_path)
    eos_id = tokenizer.sep_token_id
else:
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = MT5ForConditionalGeneration.from_pretrained(args.model_path)
    eos_id = tokenizer.eos_token_id
model = model.to(device)


class NEWS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
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

train_data = NEWS(train_data_path)


# 中文random_swap
def build_data_augments(text):
    text_segged = list(jieba.cut(text))
    if len(text_segged) > 2:
        idx = random.sample(list(range(len(text_segged))), 2)
        text_segged[idx[0]], text_segged[idx[1]] = text_segged[idx[1]], text_segged[idx[0]]
        text_segged = ''.join(text_segged)
        return text_segged
    elif len(text_segged) == 2:
        text_segged += text_segged
        random.shuffle(text_segged)
        text_segged = ''.join(text_segged)
        return text_segged
    return text + " " + text

def build_template(keyword):
    return "关于'{}'的短语是：".format(keyword)

prefix = "summarize: " if  args.model_arch == 'mt5' else ""

if with_cl:
    def collote_fn(batch_samples):
        batch_inputs, batch_targets = [], []
        for sample in batch_samples:
            title = sample['title']
            summary = sample['event']
            keyword = sample['keyword']
            batch_inputs.append(prefix + title)
            if with_prompt:
                batch_targets.append(build_template(keyword) + summary)
            else:
                batch_targets.append(summary)

        for sample in batch_samples:
            title = sample['title']
            summary = sample['event']
            keyword = sample['keyword']
            title_aug = build_data_augments(title)
            batch_inputs.append(prefix + title_aug)
            if with_prompt:
                batch_targets.append(build_template(keyword) + summary)
            else:
                batch_targets.append(summary)

        batch_data = tokenizer(
            batch_inputs, 
            padding=True, 
            max_length=max_input_length,
            truncation=True, 
            return_tensors="pt"
        )
        for fea in batch_data:
            batch_data[fea] = batch_data[fea].view(-1, 2, batch_data[fea].shape[-1])
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
            start_id = tokenizer.encode("：")[1]
            start_token_index = torch.argmax((labels == start_id).int(), dim=1)
            for idx, end_idx in enumerate(end_token_index):
                labels[idx][end_idx+1:] = -100
            for idx, start_idx in enumerate(start_token_index):
                labels[idx][:start_idx+1] = -100
            batch_data['labels'] = labels
        return batch_data
else:
    def collote_fn(batch_samples):
        batch_inputs, batch_targets = [], []
        for sample in batch_samples:
            title = sample['title']
            summary = sample['event']
            keyword = sample['keyword']
            batch_inputs.append(prefix + title)
            if with_prompt:
                batch_targets.append(build_template(keyword) + summary)
            else:
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
            start_id = tokenizer.encode("：")[1]
            start_token_index = torch.argmax((labels == start_id).int(), dim=1)
            for idx, end_idx in enumerate(end_token_index):
                labels[idx][end_idx+1:] = -100
            for idx, start_idx in enumerate(start_token_index):
                labels[idx][:start_idx+1] = -100
            batch_data['labels'] = labels
        return batch_data

train_sampler = DistributedSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, collate_fn=collote_fn)

# 对比学习Loss
def compute_infoceLoss(y_pred, tao=0.05, device="cpu"):  # y_pred:batch_size, hidden_size
    # 计算相似度矩阵
    y_pred = y_pred / torch.norm(y_pred, dim=1, keepdim=True)
    sim_matrix = torch.matmul(y_pred, y_pred.T)
    sim_matrix = sim_matrix - torch.eye(sim_matrix.shape[0], device=device) * 1e12
    sim_matrix = sim_matrix / tao  # batch_size, batch_size
    idx = torch.arange(sim_matrix.shape[0], device=device)
    # idx按中间位置逆转
    idx = torch.cat([idx[sim_matrix.shape[0]//2:], idx[:sim_matrix.shape[0]//2]])
    loss = F.cross_entropy(sim_matrix, idx)
    return loss

if with_cl:
    def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
        progress_bar = tqdm(range(len(dataloader)))
        progress_bar.set_description(f'loss: {0:>7f}')
        finish_batch_num = (epoch-1) * len(dataloader)
        
        model.train()
        for batch, batch_data in enumerate(dataloader, start=1):
            for fea in batch_data:
                batch_data[fea] = batch_data[fea].view(-1, batch_data[fea].shape[-1])
            batch_data = batch_data.to(device)
            outputs = model(input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                        decoder_input_ids=batch_data['decoder_input_ids'],
                        labels=batch_data['labels'],
                                output_hidden_states=True
                        )
            cls_embs = outputs['encoder_hidden_states'][-1][:, 0,:] 
            loss_s2s = outputs.loss
            loss_cl = compute_infoceLoss(cls_embs, device=device)
            loss = 0.8 * loss_s2s + 0.2 * loss_cl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
            progress_bar.update(1)
            # 分别打印两个loss
            if batch % 100 == 0:
                print("loss_s2s: {}, loss_cl: {}".format(loss_s2s.item(), loss_cl.item()))
        return total_loss
else:
    def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
        progress_bar = tqdm(range(len(dataloader)))
        progress_bar.set_description(f'loss: {0:>7f}')
        finish_batch_num = (epoch-1) * len(dataloader)
        
        model.train()
        for batch, batch_data in enumerate(dataloader, start=1):
            batch_data = batch_data.to(device)
            outputs = model(input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                        decoder_input_ids=batch_data['decoder_input_ids'],
                        labels=batch_data['labels'],
                                output_hidden_states=True
                        )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
            progress_bar.update(1)
            if batch % 100 == 0:
                print("loss: {}".format(loss.item()))
        return total_loss

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

model_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

total_loss = 0.
best_avg_rouge = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model_ddp, optimizer, lr_scheduler, t+1, total_loss)
    print(total_loss)
    if args.local_rank==0:
        model_to_save = model_ddp.module if hasattr(model_ddp, "module") else model_ddp
        torch.save(model_to_save.state_dict(), output_dir + '/model_epoch_{}.pt'.format(t+1))
