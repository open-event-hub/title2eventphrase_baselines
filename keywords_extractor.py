import sys
import random
import jieba
import json
import jieba.posseg as pseg
from keybert import KeyBERT
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

INPUT =  sys.argv[1]

kw_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
kw_model = KeyBERT(model=kw_model)
candidate_pos = ['n', 'nr', 'ns', 'nt', 'nz', 'nw', 'PER', 'LOC', 'ORG']

def tokenize_zh(text):
    words = pseg.cut(text)
    cutted_words = []
    candidate_words = []
    for word, flag in words:
        cutted_words.append(word)
        if flag in candidate_pos:
            candidate_words.append(word)
    return ' '.join(cutted_words), candidate_words
        
def extract_most_key_word(text, top_n=10):
    text, candidate_words = tokenize_zh(text)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=top_n)
    keywords = [keyword[0] for keyword in keywords]
    for keyword in keywords:
        if keyword in candidate_words:
            return keyword
    if len(keywords) > 0:
        return keywords[0]
    return ""

data = []
with open(INPUT, 'r') as fin:
    for line in fin.readlines():
        line = line.strip()
        if not line:
            continue
        data.append(json.loads(line.strip()))

# text = '广场上的人民英雄纪念碑，是一座纪念中国人民抗日战争胜利的纪念碑，位于北京市中心的天安门广场，是中国人民抗日战争胜利70周年纪念活动的主要场所之一。'
# keywords = extract_most_key_word(text)
# print(keywords)
OUTPUT = INPUT + ".keywords"
with open(OUTPUT, 'w') as fou:
    for line in tqdm(data):
        title = line['title']
        keyword = extract_most_key_word(title)
        line['keyword'] = keyword
        fou.write(json.dumps(line, ensure_ascii=False) + "\n")