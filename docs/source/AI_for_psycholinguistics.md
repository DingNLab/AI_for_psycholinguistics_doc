# 内容整合

## 0. 加载工具包

* 使用到的自然语言处理工具包
    - **srilm**: 计算频次、转移概率。[[官方文档]](https://srilm-python.readthedocs.io/en/latest/#)
    - **hanlp**: 适用于基本自然语言处理任务，包括分词、词性分析、句法分析、语义分析、静态词向量提取等等，对于中文比较友好。[[官方文档]](https://hanlp.hankcs.com/docs/)
    - **Huggingface系列**: 调用开源深度学习模型完成以上两者提到的，以及更多其他的任务，包括情感分析、文本生成等等。[[官网]](https://huggingface.co/)
* 其他常用的自然语言处理工具包
    - **nltk**：可以调用众多语料库（如wordnet等），也可以进行一系列的自然语言处理任务。[[官方文档]](https://www.nltk.org/)
    - **spacy**：速度快、功能全面的自然语言处理工具包。[[官方文档]](https://spacy.io/)
    - **stanza**：Stanford CoreNLP的python版本
    - **fastNLP**：复旦大学制作的NLP工具包


```python
# 如果在colab等服务器上运行，先用以下命令去掉#安装工具包
#!pip install srilm
#!pip install hanlp
#!pip install transformers, tokenizers
```


```python
# 如果只需要提取一部分特征，可以选择性地导入以下工具包
import os
import re
import json
from collections import Counter, OrderedDict
from tqdm import tqdm

# 数据处理及可视化
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc("font", family='SimHei') # 用来显示中文，对于macos系统需要换一个支持的字体

# 自然语言处理
from srilm import LM
import hanlp
import torch
from transformers import (
    BertTokenizer,
    GPT2LMHeadModel, 
    TextGenerationPipeline,
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForSeq2SeqLM,
    pipeline
    )

```

## 1. 词汇及语义特征提取

### 1.1 数据预处理：加载语料库以及进行分词


```python
def filter_str(astr, tokenizer):
    '''
    # 使用分词模型来分词
    输入: 
        astr: str, a sentence
        tokenizer: hanlp tokenizer
    输出:
        a sentence with words separated by space
    '''
    words = tokenizer(astr)
    return ' '.join(words)

def prepare_corpus(tokenizer, corpus, save_json_name):
    '''
    # 对语料库进行分词
    输入:
        tokenizer: hanlp tokenizer
        corpus: str, the path of corpus
        save_json_name: str, the path of saving json file
    输出: 
        
    '''
    with open(save_json_name, 'r', encoding='utf-8') as fp:
        wiki_texts = json.load(fp)
        wiki_texts_new = []
        for line in tqdm(wiki_texts):
            wiki_texts_new.append(filter_str(line, tokenizer))
        open(corpus, 'w').write('\n'.join(wiki_texts_new))

# 加载hanlp中的分词模型
hanlp_tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
# wiki语料
wiki_file = './srilm_data_model/wiki_demo/wiki_z.json'
# 分词后语料文件
wiki_file_tkd = './srilm_data_model/wiki_demo/wiki_z_word.txt'
# 执行
prepare_corpus(hanlp_tok, wiki_file_tkd, wiki_file)
```

    100%|██████████| 1/1 [00:00<00:00,  9.35it/s]                 


### 1.2 基于语料库统计的N-gram计算

#### 1.2.1 从语料库中生成N-gram模型
* 将语料库（corpus）和指定的模型设置（ngram）输入模型，在模型存储路径（model_path）中输出统计好的模型
* 现成的N-gram语料库：[google n-gram](http://storage.googleapis.com/books/ngrams/books/datasetsv2.html)


```python
def generate_model(model_path, ngram, corpus):
    '''
    输入:
        model_path: str, ngram模型的保存路径
        ngram: str, ngram-count路径
        corpus: str, corpus路径
    输出:
        
    '''
    cmd = '{} -text {} -order 3 -kndiscount3 -lm {}'.format(ngram, corpus, model_path)
    os.system(cmd)

ngram = '/home/zhang/acoustic_theory/workspace/21-12-30-srilm/srilm/bin/i686-m64/ngram-count'
wiki_file_tkd = './srilm_data_model/wiki_demo/wiki_z_word.txt'
model_path = './srilm_data_model/wiki_demo/wiki_z_word.lm'
generate_model(model_path, ngram, wiki_file_tkd)
```

    warning: discount coeff 1 is out of range: 0
    warning: discount coeff 7 is out of range: 1.91919



```python
model_path = './srilm_data_model/wiki/wiki_z_word.lm'
lm = LM(model_path, lower=True) # 加载N-gram模型
```

#### 1.2.2 采用N-gram模型计算词频
用srilm的LM来调用刚刚生成的模型，采用`lm.logprob_strings(word, context)`来生成 $\log{p \left( \rm{word} | context \right)}$，word是当前单词，当context是空列表`[]`时相当于1-gram即词频


```python
# 计算词频
print('*'*20 + ' 计算词频 ' + '*'*20)
word_freq0_ = lm.logprob_strings('的', [])
word_freq1_ = lm.logprob_strings('西瓜', [])
word_freq2_ = lm.logprob_strings('桌子', [])

# 输出结果
print('='*20 + 'P(的) vs P(西瓜) vs P(桌子)' + '='*20)
print('P(的): ' + str(word_freq0_))
print('P(西瓜): ' + str(word_freq1_))
print('P(桌子): ' + str(word_freq2_))
```

    ******************** 计算词频 ********************
    ====================P(的) vs P(西瓜) vs P(桌子)====================
    P(的): -1.3277089595794678
    P(西瓜): -5.5793938636779785
    P(桌子): -5.5162577629089355


#### 1.2.3 采用N-gram模型计算转移概率

当$n>1$时，在`context`中放入前$n-1$个词，顺序是从右到左。


```python
tp1_ = lm.logprob_strings('西瓜', ['吃', '喜欢'])
tp2_ = lm.logprob_strings('桌子', ['吃', '喜欢'])
print('='*10 + 'P(西瓜 | 吃, 喜欢) vs P(桌子 | 吃, 喜欢)' + '='*10)
print('P(西瓜 | 吃, 喜欢): ' + str(tp1_))
print('P(桌子 | 吃, 喜欢): ' + str(tp2_))
```

    ==========P(西瓜 | 吃, 喜欢) vs P(桌子 | 吃, 喜欢)==========
    P(西瓜 | 吃, 喜欢): -2.884925365447998
    P(桌子 | 吃, 喜欢): -6.211382865905762


#### 1.2.4 采用N-gram模型计算surprisal
$\rm{surprisal} = -\log{ \it{p} \left( \rm{word} | context \right)}$，所以只要取负即可。


```python
s1_ = -lm.logprob_strings('西瓜', ['吃', '喜欢'])
s2_ = -lm.logprob_strings('桌子', ['吃', '喜欢'])
print('='*10 + 'surprisal(西瓜 | 吃, 喜欢) vs surprisal(桌子 | 吃, 喜欢)' + '='*10)
print('surprisal(西瓜 | 吃, 喜欢): ' + str(s1_))
print('surprisal(桌子 | 吃, 喜欢): ' + str(s2_))
```

    ==========surprisal(西瓜 | 吃, 喜欢) vs surprisal(桌子 | 吃, 喜欢)==========
    surprisal(西瓜 | 吃, 喜欢): 2.884925365447998
    surprisal(桌子 | 吃, 喜欢): 6.211382865905762


#### 1.2.5 采用N-gram模型计算entropy
$\rm{entropy} = \sum \left( p*surprisal \right)$，所以对于给定的context，对所有的词来计算surprisal然后求期望


```python
model_path = './srilm_data_model/wiki/wiki_z_morpheme.lm'
lm = LM(model_path, lower=True) # 加载N-gram模型
def entropy_cal(lm, context):
    # entropy
    raw_text_idx = [lm.vocab.intern(w) for w in context]
    vocab_num = lm.vocab.max_interned() + 1
    logprobs = [lm.logprob(i, raw_text_idx) for i in range(vocab_num)]
    logprobs_np = np.array(logprobs)
    logprobs_np_ = logprobs_np[logprobs_np > -np.inf]
    entropy_ = sum(-np.power(10, logprobs_np_)*logprobs_np_)
    return entropy_

print('='*10 + 'entropy(蝴) vs entropy(。)' + '='*10)
e1_ = entropy_cal(lm, ['蝴'])
print('entropy(蝴): ' + str(e1_))
e2_ = entropy_cal(lm, ['。'])
print('entropy(。): ' + str(e2_))
```

    ==========entropy(蝴) vs entropy(。)==========
    entropy(蝴): 0.03182660213747036
    entropy(。): 2.5136258206385347


### 1.3 基于深度学习模型的转移概率计算

以gpt-2为例，采用的模型为[gpt2-chinese-cluecorpussmall](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)

#### 1.3.1 加载模型，包括分词模型与语言模型



```python
from transformers import BertTokenizer, GPT2LMHeadModel
ckpt_path = "uer/gpt2-chinese-cluecorpussmall" # checkpoint模型路径
tokenizer = BertTokenizer.from_pretrained(ckpt_path) # 分词器
model = GPT2LMHeadModel.from_pretrained(ckpt_path) # 语言模型
```

#### 1.3.2 获取模型的转移概率
后续的surprisal和entropy也可以通过转移概率算出来，与1.2部分类似


```python
model.config.output_hidden_states = True  # 在模型设置config中设置为True，可以让模型输出hidden states
inputs = tokenizer('小明喜欢吃西瓜。小明喜欢打篮球。小明经常去花店', return_tensors="pt") # 对句子进行分词
outputs = model(**inputs)  # 将分词后的句子输入模型，得到模型输出的结果

print('='*10 + '输入字数: ' + '='*10)
print(len(inputs['input_ids'][0]))

print('='*10 + '转移概率维度: ' + '='*10)
print(str(outputs.logits[0].shape) + '  输入字数 x 总字数')
```

    ==========输入字数: ==========
    25
    ==========转移概率维度: ==========
    torch.Size([25, 21128])  输入字数 x 总字数


### 1.4 词性


```python
## 0. 分词
sent_ex = '这个门被锁了，锁很难被打开。'
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
tks = tok(sent_ex)
print('0. 分词结果：')
print(tks)

## 1. 词性标注
pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
print('1. 词性标注：')
print(pos(tks))
```

    Building model [5m[33m...[0m[0m          

    0. 分词结果：
    ['这个', '门', '被', '锁', '了', '，', '锁', '很难', '被', '打开', '。']


                                                 

    1. 词性标注：
    ['DT', 'NN', 'SB', 'VV', 'SP', 'PU', 'VV', 'AD', 'SB', 'VV', 'PU']


### 1.5 词向量

#### 1.5.1 获取静态词向量：以word2vec为例
* hanlp支持调用各种静态词向量， 包括[word2vec](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/word2vec.html), [glove](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/glove.html)等等，具体的模型及文献可以在链接文档中进行选择，一般情况下维度越高越准确。


```python
word2vec = hanlp.load(hanlp.pretrained.word2vec.MERGE_SGNS_BIGRAM_CHAR_300_ZH) # 加载word2vec词向量
word2vec('中国')
```

                                                            




    tensor([ 1.4234e-02,  8.3600e-02,  2.4145e-02, -1.0256e-01, -1.0829e-01,
            -2.6786e-02, -9.6481e-02,  9.0537e-02, -5.4941e-02,  4.5936e-02,
            -4.2577e-02, -5.1776e-02,  4.9661e-02, -3.2703e-02, -6.6407e-03,
             9.8313e-03,  4.2377e-02, -7.1969e-02,  6.7363e-02, -1.2679e-01,
             1.3423e-03,  1.8129e-02,  1.3923e-02,  6.0298e-02,  2.9974e-02,
             3.4969e-02,  4.7053e-02, -1.4874e-02,  6.6235e-02, -1.5579e-01,
            -1.1716e-01,  8.8726e-02,  6.0976e-02, -8.0692e-02, -3.1017e-02,
            -1.3132e-02,  5.4841e-02,  4.0733e-02, -1.5295e-01, -7.8516e-02,
             6.6119e-02,  2.9393e-02, -3.0162e-02, -4.3704e-02,  8.3047e-03,
            -7.7654e-02, -1.5644e-02,  6.2678e-02,  7.3149e-02, -1.9128e-02,
             2.7543e-02, -1.4893e-02, -1.2223e-02,  9.6474e-02,  2.1985e-02,
             4.4640e-02, -2.4626e-02,  9.8536e-02, -1.3777e-01,  5.1621e-02,
             9.5042e-02, -3.2784e-02,  2.8697e-02, -1.3267e-02,  1.1536e-02,
            -9.0047e-02, -7.2654e-02, -8.7082e-04, -3.6991e-02,  1.6448e-03,
             2.6809e-02, -7.5198e-02, -2.6094e-02,  6.5516e-03, -7.2922e-02,
            -6.3720e-02, -6.4798e-03,  1.3006e-02,  1.7040e-02, -4.3527e-02,
             1.6448e-03, -4.0217e-02,  2.1293e-02, -4.1442e-02, -4.9964e-02,
             1.0784e-02,  1.2986e-01, -1.7174e-02,  9.0332e-02,  8.1890e-04,
            -4.3150e-02, -6.7029e-02, -4.6127e-02, -6.4486e-02, -1.8022e-02,
             1.3425e-02,  6.9962e-02, -1.4400e-02,  6.0225e-03, -3.7480e-03,
             8.5195e-03, -2.2870e-02, -4.1049e-02, -1.8603e-02, -5.3075e-02,
            -7.1510e-02,  9.2589e-03, -6.3029e-03, -2.4524e-02, -3.4340e-02,
            -8.8730e-02,  1.5332e-02,  2.8820e-02,  1.8295e-02, -5.8320e-02,
            -2.7167e-02, -1.7402e-02, -7.7428e-02, -1.0769e-01, -1.0446e-01,
             4.5363e-02, -6.3230e-02,  8.3784e-02,  5.3965e-02,  2.0121e-02,
            -3.7716e-02, -2.0752e-02, -6.2321e-02, -1.3778e-01,  5.0385e-02,
             8.9087e-06, -8.1429e-02,  6.1611e-02, -4.1132e-02,  7.4521e-02,
            -5.0390e-02, -1.6549e-02,  4.1053e-02, -1.7056e-02, -1.2268e-02,
            -1.3683e-02,  1.0725e-02, -5.9534e-02, -3.3246e-02,  3.8279e-02,
            -3.6564e-02,  6.8516e-02,  6.6845e-02,  4.3522e-02, -2.3375e-02,
            -1.3111e-02,  1.4433e-03,  3.9912e-02,  3.8543e-03,  8.9713e-02,
             1.9988e-02,  9.5058e-04, -7.2403e-02, -3.7107e-02, -6.4932e-02,
            -2.1959e-02,  3.4034e-02, -2.9596e-02, -6.8593e-02, -1.9584e-02,
             4.0717e-02, -1.0285e-01, -6.5889e-03,  9.2453e-03, -4.2289e-02,
            -5.7992e-02,  3.3845e-02,  1.3048e-02, -5.1361e-02,  7.8392e-02,
            -1.9344e-02, -1.0448e-01,  4.1529e-02, -9.7657e-02, -3.4509e-03,
             4.9083e-02,  5.5863e-02,  8.7877e-03, -1.1969e-01,  7.1582e-02,
             2.4624e-02, -2.8234e-03, -1.0275e-01, -8.0798e-02, -1.2945e-01,
             1.7228e-02, -8.7083e-02, -4.5541e-02, -3.6977e-02,  7.5634e-02,
             6.3264e-02, -1.0102e-01, -9.6761e-02, -1.7960e-02, -1.6474e-02,
             6.5089e-02, -5.6679e-02,  1.7903e-02, -6.3342e-02,  2.1894e-02,
            -8.5694e-03, -2.0418e-02,  9.6943e-02,  6.6336e-02,  5.3024e-02,
             7.7205e-02,  7.5687e-02, -2.4854e-02, -8.4196e-02,  7.2153e-02,
            -3.3994e-02,  2.7743e-02,  7.6132e-02,  1.2271e-01,  8.2420e-02,
             2.2781e-02,  6.0472e-03, -1.5400e-01, -1.1090e-01, -1.8680e-03,
             9.7762e-02,  3.7373e-03, -2.6415e-02,  1.7530e-02,  9.8943e-03,
            -4.3207e-02,  4.6805e-02,  1.3863e-02, -5.2318e-02, -3.4550e-03,
            -3.7918e-02,  2.9433e-02,  3.3142e-02,  8.7807e-03,  3.0049e-02,
             8.8094e-02,  1.4916e-03, -1.7431e-02, -2.5317e-02, -1.6277e-02,
             1.1268e-02,  9.4293e-02,  3.3744e-02, -3.4135e-02,  6.1734e-04,
            -5.8349e-02,  1.2800e-01,  2.4264e-03, -1.0573e-01, -2.0444e-02,
             3.9112e-02, -1.4461e-01,  6.4038e-02, -8.3256e-03, -4.6320e-02,
            -1.3400e-02,  1.2040e-02,  7.3522e-02, -1.6663e-02, -1.2628e-03,
            -2.7094e-02, -1.8414e-03,  6.0205e-02, -6.7361e-02,  5.6380e-02,
             2.3484e-03, -4.5203e-03,  4.1993e-02,  2.9977e-02, -1.2228e-02,
             2.8904e-03, -1.7870e-02, -1.3307e-02, -4.5424e-02, -3.1245e-02,
             4.0651e-03,  1.0091e-01,  6.3333e-02,  1.5903e-01,  9.9152e-02,
            -2.0661e-02,  6.4784e-03,  1.3163e-03,  2.6181e-02, -9.9187e-03,
             1.4386e-02, -4.5888e-02,  5.6548e-02,  3.5045e-02,  5.5262e-02,
             3.0622e-02,  9.1758e-03, -1.0747e-01,  5.5859e-03, -5.0639e-02],
           device='cuda:1')



- 捕获了性别信息
- 捕获了首都信息


```python
print(torch.nn.functional.cosine_similarity(
    word2vec('国王')-word2vec('王妃'), 
    word2vec('男')-word2vec('女'), dim=0)
      )
print(torch.nn.functional.cosine_similarity(
    word2vec('公主')-word2vec('王妃'), 
    word2vec('男')-word2vec('女'), dim=0)
      )
```

    tensor(0.1429, device='cuda:1')
    tensor(0.0366, device='cuda:1')



```python
print(torch.nn.functional.cosine_similarity(
    word2vec('日本')-word2vec('东京'), 
    word2vec('中国')-word2vec('北京'), dim=0)
      )
print(torch.nn.functional.cosine_similarity(
    word2vec('韩国')-word2vec('东京'), 
    word2vec('中国')-word2vec('北京'), dim=0)
      )

```

    tensor(0.4674, device='cuda:1')
    tensor(0.3933, device='cuda:1')


- 计算相似词


```python
# 单个词
print(word2vec.most_similar('北京')) 
print('\n')
```

    {'上海': 0.6443496942520142, '天津': 0.6384099721908569, '西安': 0.611718475818634, '南京': 0.6113559603691101, '北京市': 0.6093109846115112, '海淀': 0.6049214601516724, '广州': 0.5977935791015625, '京城': 0.5955069661140442, '沈阳': 0.5865166187286377, '深圳': 0.580772876739502}
    
    


#### 1.5.2 获取基于上下文的词向量：语言模型的隐藏层表征

同样以1.3中调用的[gpt2-chinese-cluecorpussmall](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)为例


```python
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
ckpt_path = "uer/gpt2-chinese-cluecorpussmall" # checkpoint模型路径
tokenizer = BertTokenizer.from_pretrained(ckpt_path) # 分词器
model = GPT2LMHeadModel.from_pretrained(ckpt_path) # 语言模型
```


```python
model.config.output_hidden_states = True
inputs = tokenizer('小明喜欢吃西瓜。小明喜欢打篮球。小明经常去花店', return_tensors="pt")
outputs = model(**inputs)

print('\n' + '='*10 + '最后一层输出的内隐表征维度: ' + '='*10)
print(str(outputs.hidden_states[-1].shape) + '  1 x 输入字数 x 表征维度')
```

    
    ==========最后一层输出的内隐表征维度: ==========
    torch.Size([1, 25, 768])  1 x 输入字数 x 表征维度


## 2. 句法特征提取

### 2.1 句法特征抽取


```python
import hanlp
```


```python
## workshop中的例子，研究中一般会把标点去掉，但是这里保留了标点，模型也是能够解析标点的
Hanlp = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH) # 选择使用的模型
doc = Hanlp('欢迎大家参加工作坊！', tasks=['dep', 'con']) # 在tasks中选择需要的任务，如果不设置就进行所有任务（运行起来会慢一点）
doc.pretty_print()
```

                                                 


<div style="display: table; padding-bottom: 1rem;"><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Dep&nbsp;Tre&nbsp;<br>───────&nbsp;<br>┌┬──┬──&nbsp;<br>││&nbsp;&nbsp;└─►&nbsp;<br>│└─►┌──&nbsp;<br>│&nbsp;&nbsp;&nbsp;└─►&nbsp;<br>└─────►&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Tok&nbsp;<br>───&nbsp;<br>欢迎&nbsp;&nbsp;<br>大家&nbsp;&nbsp;<br>参加&nbsp;&nbsp;<br>工作坊&nbsp;<br>！&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Relat&nbsp;<br>─────&nbsp;<br>root&nbsp;&nbsp;<br>dobj&nbsp;&nbsp;<br>dep&nbsp;&nbsp;&nbsp;<br>dobj&nbsp;&nbsp;<br>punct&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Tok&nbsp;<br>───&nbsp;<br>欢迎&nbsp;&nbsp;<br>大家&nbsp;&nbsp;<br>参加&nbsp;&nbsp;<br>工作坊&nbsp;<br>！&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">P&nbsp;&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7&nbsp;<br>───────────────────────────────────────<br>_──────────────────────────┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>_───────────────────►NP────┤&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>_──────────┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├►VP&nbsp;───┐&nbsp;&nbsp;&nbsp;<br>_───►NP&nbsp;───┴►VP&nbsp;────►IP&nbsp;───┘&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├►IP<br>_──────────────────────────────────┘&nbsp;&nbsp;&nbsp;</pre></div>


#### 2.1.1 成分句法

成分句法输出得到的是一个树结构的数据，可以看作一个嵌套的列表。我们可以：
* 访问句法树的一些属性
* 转换为括号表示法，计算括号数量
* 访问句法树的子树


```python
Hanlp = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
doc = Hanlp('欢迎大家参加工作坊！')
tree = doc['con']
```

                                                 


```python
# 叶结点的位置
for i in range(len(tree.leaves())):
    print(tree.leaf_treeposition(i))
```

    (0, 0, 0, 0)
    (0, 0, 1, 0, 0)
    (0, 0, 2, 0, 0, 0)
    (0, 0, 2, 0, 1, 0, 0)
    (0, 1, 0)



```python
tree[0, 0, 1, 0, 0]
```




    '大家'




```python
# 转为括号表示法
bracket_form = tree.pformat().replace ('\n', '').replace(' ', '') # 去掉换行和空格
bracket_form
```




    '(TOP(IP(VP(VV欢迎)(NP(PN大家))(IP(VP(VV参加)(NP(NN工作坊)))))(PU！)))'




```python
# 转换为Chomsky Normal Form，可以用tree.un_chomsky_normal_form()转换回来
tree.chomsky_normal_form() 
bracket_form = tree.pformat().replace ('\n', '').replace(' ', '')
print(bracket_form)
```

    (TOP(IP(VP(VV欢迎)(VP|<NP-IP>(NP(PN大家))(IP(VP(VV参加)(NP(NN工作坊))))))(PU！)))



```python
# 输出中有些节点只派生出一支，是冗余的（例如最外层的TOP根结点只派生出IP，以及句子中的IP只派生出VP），可以选择压缩节点
tree.collapse_unary(collapseRoot=True, joinChar='|') # 压缩冗余节点，压缩的节点用｜来表示
bracket_form = tree.pformat().replace ('\n', '').replace(' ', '')
bracket_form 
```




    '(TOP|IP(VP(VV欢迎)(VP|<NP-IP>(NP(PN大家))(IP|VP(VV参加)(NP(NN工作坊)))))(PU！))'




```python
import re
import pandas as pd
# 计算括号表示法中每个词的括号数
bracket_clean= re.sub("([^()])", "", bracket_form) # 只保留括号
print(bracket_clean)

# 计算左括号数
left_bracket = [len(re.findall("\(", i)) for i in bracket_clean] 
left_bracket_count = []
for i in left_bracket:
    if len(left_bracket_count) == 0 or (i == 1 and j != 1):
        left_bracket_count.append(1)
    elif i == 1 and j == 1:
        left_bracket_count[-1] += 1
    j = i
print("左括号数:", left_bracket_count)

# 计算右括号数
right_bracket = [len(re.findall("\)", i)) for i in bracket_clean] 
right_bracket_count = []; j = 0
for i in right_bracket:
    if i == 1 and j != 1:
        right_bracket_count.append(1)
    elif i == 1 and j == 1:
        right_bracket_count[-1] += 1
    j = i
print("右括号数:", right_bracket_count)

# 可以保存为 dataframe 进行进一步的句法特征分析
df_bracket = pd.DataFrame([tree.leaves(), left_bracket_count, right_bracket_count]).T
df_bracket.columns = ['word', 'left_bracket', 'right_bracket']
# df_bracket.to_csv('bracket.csv', index=False) # 保存为csv文件
df_bracket
```

    ((()((())(()(()))))())
    左括号数: [3, 3, 2, 2, 1]
    右括号数: [1, 2, 1, 5, 2]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>left_bracket</th>
      <th>right_bracket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>欢迎</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>大家</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>参加</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>工作坊</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>！</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 句法树的属性
print("Terminal nodes:", tree.leaves())
print("Tree depth:", tree.height())
print("Tree productions:", tree.productions())
print("Part of Speech:", tree.pos())
```

    Terminal nodes: ['欢迎', '大家', '参加', '工作坊', '！']
    Tree depth: 7
    Tree productions: [TOP|IP -> VP PU, VP -> VV VP|<NP-IP>, VV -> '欢迎', VP|<NP-IP> -> NP IP|VP, NP -> PN, PN -> '大家', IP|VP -> VV NP, VV -> '参加', NP -> NN, NN -> '工作坊', PU -> '！']
    Part of Speech: [('欢迎', 'VV'), ('大家', 'PN'), ('参加', 'VV'), ('工作坊', 'NN'), ('！', 'PU')]



```python
# 句法树的嵌套结构
for i in tree.subtrees():  # 根据Tree productions，遍历所有的子树，每一棵子树都是一个Tree对象，可以进行之前相同的操作
    print(i)
```

    (TOP|IP
      (VP
        (VV 欢迎)
        (VP|<NP-IP> (NP (PN 大家)) (IP|VP (VV 参加) (NP (NN 工作坊)))))
      (PU ！))
    (VP (VV 欢迎) (VP|<NP-IP> (NP (PN 大家)) (IP|VP (VV 参加) (NP (NN 工作坊)))))
    (VV 欢迎)
    (VP|<NP-IP> (NP (PN 大家)) (IP|VP (VV 参加) (NP (NN 工作坊))))
    (NP (PN 大家))
    (PN 大家)
    (IP|VP (VV 参加) (NP (NN 工作坊)))
    (VV 参加)
    (NP (NN 工作坊))
    (NN 工作坊)
    (PU ！)



```python
# 通过索引访问句法树的子树
treepositions = tree.treepositions() # 所有节点的索引
treepositions
```




    [(),
     (0,),
     (0, 0),
     (0, 0, 0),
     (0, 1),
     (0, 1, 0),
     (0, 1, 0, 0),
     (0, 1, 0, 0, 0),
     (0, 1, 1),
     (0, 1, 1, 0),
     (0, 1, 1, 0, 0),
     (0, 1, 1, 1),
     (0, 1, 1, 1, 0),
     (0, 1, 1, 1, 0, 0),
     (1,),
     (1, 0)]




```python
for i in treepositions: # 遍历所有节点
    print(tree[i])
```

    (TOP|IP
      (VP
        (VV 欢迎)
        (VP|<NP-IP> (NP (PN 大家)) (IP|VP (VV 参加) (NP (NN 工作坊)))))
      (PU ！))
    (VP (VV 欢迎) (VP|<NP-IP> (NP (PN 大家)) (IP|VP (VV 参加) (NP (NN 工作坊)))))
    (VV 欢迎)
    欢迎
    (VP|<NP-IP> (NP (PN 大家)) (IP|VP (VV 参加) (NP (NN 工作坊))))
    (NP (PN 大家))
    (PN 大家)
    大家
    (IP|VP (VV 参加) (NP (NN 工作坊)))
    (VV 参加)
    参加
    (NP (NN 工作坊))
    (NN 工作坊)
    工作坊
    (PU ！)
    ！


#### 2.1.2 依存句法
* 依存句法的数据结构更加简单，为一个列表`[(head, relation), ... ]`。列表中第$i$个值中包括了它的核心词的位置以及它与核心词之间的依存关系


```python
Hanlp = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
doc = Hanlp('欢迎大家参加工作坊！')
doc['dep']
```

                                                 




    [(0, 'root'), (1, 'dobj'), (1, 'dep'), (3, 'dobj'), (1, 'punct')]




```python
# 可以保存为 dataframe 进行进一步的句法特征分析
df_dep = pd.DataFrame(doc['dep'], columns=['head', 'rel'])
df_dep['word'] = doc['tok/fine']
df_dep = df_dep[['word', 'head', 'rel']]
df_dep
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>head</th>
      <th>rel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>欢迎</td>
      <td>0</td>
      <td>root</td>
    </tr>
    <tr>
      <th>1</th>
      <td>大家</td>
      <td>1</td>
      <td>dobj</td>
    </tr>
    <tr>
      <th>2</th>
      <td>参加</td>
      <td>1</td>
      <td>dep</td>
    </tr>
    <tr>
      <th>3</th>
      <td>工作坊</td>
      <td>3</td>
      <td>dobj</td>
    </tr>
    <tr>
      <th>4</th>
      <td>！</td>
      <td>1</td>
      <td>punct</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2 批量操作

只需要将要处理的句子放在list中，一起进行特征抽取即可。这对所有特征都适用，不仅是句法特征。


```python
sentences = ['2023年心理语言学会在广州召开。', '欢迎大家参加工作坊！']
docs = Hanlp(sentences)
docs.pretty_print()
```


<div style="display: table; padding-bottom: 1rem;"><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Dep&nbsp;Tree&nbsp;<br>────────&nbsp;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;┌─►&nbsp;<br>┌───►└──&nbsp;<br>│&nbsp;&nbsp;&nbsp;┌──►&nbsp;<br>│&nbsp;&nbsp;&nbsp;│┌─►&nbsp;<br>│┌─►└┴──&nbsp;<br>││┌─►┌──&nbsp;<br>│││&nbsp;&nbsp;└─►&nbsp;<br>└┴┴──┬──&nbsp;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─►&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Toke&nbsp;<br>────&nbsp;<br>2023&nbsp;<br>年&nbsp;&nbsp;&nbsp;&nbsp;<br>心理&nbsp;&nbsp;&nbsp;<br>语言&nbsp;&nbsp;&nbsp;<br>学会&nbsp;&nbsp;&nbsp;<br>在&nbsp;&nbsp;&nbsp;&nbsp;<br>广州&nbsp;&nbsp;&nbsp;<br>召开&nbsp;&nbsp;&nbsp;<br>。&nbsp;&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Relati&nbsp;<br>──────&nbsp;<br>nummod&nbsp;<br>nsubj&nbsp;&nbsp;<br>nn&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>nn&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>nsubj&nbsp;&nbsp;<br>prep&nbsp;&nbsp;&nbsp;<br>pobj&nbsp;&nbsp;&nbsp;<br>root&nbsp;&nbsp;&nbsp;<br>punct&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Po&nbsp;<br>──&nbsp;<br>NT&nbsp;<br>M&nbsp;&nbsp;<br>NN&nbsp;<br>NN&nbsp;<br>NN&nbsp;<br>P&nbsp;&nbsp;<br>NR&nbsp;<br>VV&nbsp;<br>PU&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Toke&nbsp;<br>────&nbsp;<br>2023&nbsp;<br>年&nbsp;&nbsp;&nbsp;&nbsp;<br>心理&nbsp;&nbsp;&nbsp;<br>语言&nbsp;&nbsp;&nbsp;<br>学会&nbsp;&nbsp;&nbsp;<br>在&nbsp;&nbsp;&nbsp;&nbsp;<br>广州&nbsp;&nbsp;&nbsp;<br>召开&nbsp;&nbsp;&nbsp;<br>。&nbsp;&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">NER&nbsp;Type&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>────────────&nbsp;<br>───►DATE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>───►LOCATION&nbsp;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Toke&nbsp;<br>────&nbsp;<br>2023&nbsp;<br>年&nbsp;&nbsp;&nbsp;&nbsp;<br>心理&nbsp;&nbsp;&nbsp;<br>语言&nbsp;&nbsp;&nbsp;<br>学会&nbsp;&nbsp;&nbsp;<br>在&nbsp;&nbsp;&nbsp;&nbsp;<br>广州&nbsp;&nbsp;&nbsp;<br>召开&nbsp;&nbsp;&nbsp;<br>。&nbsp;&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">SRL&nbsp;PA1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>────────────&nbsp;<br>◄─┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>◄─┴►ARGM-TMP&nbsp;<br>◄─┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>&nbsp;&nbsp;├►ARG1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>◄─┘&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>◄─┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>◄─┴►ARGM-LOC&nbsp;<br>╟──►PRED&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Toke&nbsp;<br>────&nbsp;<br>2023&nbsp;<br>年&nbsp;&nbsp;&nbsp;&nbsp;<br>心理&nbsp;&nbsp;&nbsp;<br>语言&nbsp;&nbsp;&nbsp;<br>学会&nbsp;&nbsp;&nbsp;<br>在&nbsp;&nbsp;&nbsp;&nbsp;<br>广州&nbsp;&nbsp;&nbsp;<br>召开&nbsp;&nbsp;&nbsp;<br>。&nbsp;&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Po&nbsp;&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6&nbsp;<br>────────────────────────────────<br>NT──────────┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>M&nbsp;───►CLP&nbsp;──┴►QP&nbsp;───┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>NN──┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├►NP&nbsp;───┐&nbsp;&nbsp;&nbsp;<br>NN&nbsp;&nbsp;├────────►NP&nbsp;───┘&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;<br>NN──┘&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;<br>P&nbsp;──────────┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├►IP<br>NR───►NP&nbsp;───┴►PP&nbsp;───┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;<br>VV───────────►VP&nbsp;───┴►VP────┤&nbsp;&nbsp;&nbsp;<br>PU──────────────────────────┘&nbsp;&nbsp;&nbsp;</pre></div><br><div style="display: table; padding-bottom: 1rem;"><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Dep&nbsp;Tre&nbsp;<br>───────&nbsp;<br>┌┬──┬──&nbsp;<br>││&nbsp;&nbsp;└─►&nbsp;<br>│└─►┌──&nbsp;<br>│&nbsp;&nbsp;&nbsp;└─►&nbsp;<br>└─────►&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Tok&nbsp;<br>───&nbsp;<br>欢迎&nbsp;&nbsp;<br>大家&nbsp;&nbsp;<br>参加&nbsp;&nbsp;<br>工作坊&nbsp;<br>！&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Relat&nbsp;<br>─────&nbsp;<br>root&nbsp;&nbsp;<br>dobj&nbsp;&nbsp;<br>dep&nbsp;&nbsp;&nbsp;<br>dobj&nbsp;&nbsp;<br>punct&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Po&nbsp;<br>──&nbsp;<br>VV&nbsp;<br>PN&nbsp;<br>VV&nbsp;<br>NN&nbsp;<br>PU&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Tok&nbsp;<br>───&nbsp;<br>欢迎&nbsp;&nbsp;<br>大家&nbsp;&nbsp;<br>参加&nbsp;&nbsp;<br>工作坊&nbsp;<br>！&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">SRL&nbsp;PA1&nbsp;&nbsp;<br>────────&nbsp;<br>╟──►PRED&nbsp;<br>───►ARG1&nbsp;<br>◄─┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>◄─┴►ARG2&nbsp;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Tok&nbsp;<br>───&nbsp;<br>欢迎&nbsp;&nbsp;<br>大家&nbsp;&nbsp;<br>参加&nbsp;&nbsp;<br>工作坊&nbsp;<br>！&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">SRL&nbsp;PA2&nbsp;&nbsp;<br>────────&nbsp;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>╟──►PRED&nbsp;<br>───►ARG1&nbsp;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Tok&nbsp;<br>───&nbsp;<br>欢迎&nbsp;&nbsp;<br>大家&nbsp;&nbsp;<br>参加&nbsp;&nbsp;<br>工作坊&nbsp;<br>！&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Po&nbsp;&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7&nbsp;<br>────────────────────────────────────────<br>VV──────────────────────────┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>PN───────────────────►NP────┤&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>VV──────────┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├►VP&nbsp;───┐&nbsp;&nbsp;&nbsp;<br>NN───►NP&nbsp;───┴►VP&nbsp;────►IP&nbsp;───┘&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├►IP<br>PU──────────────────────────────────┘&nbsp;&nbsp;&nbsp;</pre></div>



```python
# 提取出来的特征直接索引即可
print("句子数量为:", docs.count_sentences())
for i in range(docs.count_sentences()):
    print(docs['tok/fine'][i])
```

    句子数量为: 2
    ['2023', '年', '心理', '语言', '学会', '在', '广州', '召开', '。']
    ['欢迎', '大家', '参加', '工作坊', '！']


## 3.0 语言任务
在本小节中，我们以主题分析任务和上下文学习为例，演示语言模型的加载和推理过程。对于其他语言任务，均可在huggingface平台搜索到类似的教程文档以及代码。
### 3.1 主题分析任务
使用transformers管道pipeline快速实现语言任务


```python
# 从huggingface平台上找到对应的模型路径
model_path = 'model/roberta-base-finetuned-chinanews-chinese'

# 使用transformers工具包加载模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 利用pipeline快速进行语言任务
text = '欢迎参加工作坊！'
text_classification = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
res = text_classification(text)[0]
print("="*20, "单个句子主题分析计算", "="*20)
print(f"\nInput: {text}\nPrediction: {res['label']}, Score: {res['score']:.3f}")


# pipeline可以实现批量句子的计算
text_lst = ['2023年心理语言学会在广州召开', '湖人有意签保罗补强，联手詹姆斯追逐总冠军']
res_lst = text_classification(text_lst)
print("\n\n")
print("="*20, "多个句子批量进行主题分析计算", "="*20)
for text, res in zip(text_lst, res_lst):
    print(f"\nInput: {text}\nPrediction: {res['label']}, Score: {res['score']:.3f}")
```

    ==================== 单个句子主题分析计算 ====================
    
    Input: 欢迎参加工作坊！
    Prediction: culture, Score: 0.723
    
    
    
    ==================== 多个句子批量进行主题分析计算 ====================
    
    Input: 2023年心理语言学会在广州召开
    Prediction: culture, Score: 0.969
    
    Input: 湖人有意签保罗补强，联手詹姆斯追逐总冠军
    Prediction: sports, Score: 1.000


### 3.2 上下文学习
通过在上下文中给定任务描述和示例，通用的文本生成模型可以根据上下文快速学习语言任务。在这里我们不使用pipeline，直接调用模型方法进行计算。


```python
# 从huggingface平台上找到对应的模型路径
model_path = "model/flan-t5-large"

# 使用transformers工具包加载模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


print("\n\n")
print("="*20, "上下文学习实现文本翻译", "="*20)
text = "translate English to German: How old are you?"

# 调用模型分词器，对输入文本进行分词并转换为模型可处理的tensor形式
input_ids = tokenizer(text, return_tensors="pt").input_ids

# 调用模型的generate方法
outputs = model.generate(input_ids)
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens = True)
print(f"Input: {text}\nOutput: {decoded_output}")



print("\n\n")
print("="*20, "上下文学习实现主题文本生成", "="*20)
text = '''Generate sentences with the topic : 
sports => Lionel Messi and MLS club Inter Miami are discussing possible signing
entertainment => 
'''

# 调用模型分词器，对输入文本进行分词并转换为模型可处理的tensor形式
input_ids = tokenizer(text, return_tensors="pt").input_ids

# 调用模型的generate方法
outputs = model.generate(input_ids)
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens = True)
print(f"Input: {text}\nOutput: {decoded_output}")

```

    
    
    
    ==================== 上下文学习实现文本翻译 ====================


    /home/zhang/anaconda3/envs/ngram/lib/python3.7/site-packages/transformers/generation/utils.py:1278: UserWarning: Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to 20 (`generation_config.max_length`). Controlling `max_length` via the config is deprecated and `max_length` will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation.
      UserWarning,


    Input: translate English to German: How old are you?
    Output: Wie alte sind Sie?
    
    
    
    ==================== 上下文学习实现主题文本生成 ====================
    Input: Generate sentences with the topic : 
    sports => Lionel Messi and MLS club Inter Miami are discussing possible signing
    entertainment => 
    
    Output: a new tv series starring adrian sandler is 


### 3.3 文本生成超参数
在本小节中，我们会分析文本生成中的温度参数、搜索策略参数以及top-p参数对文本生成结果的影响。


```python
# 从huggingface平台上找到对应的模型路径
model_path = "model/flan-t5-large"

# 使用transformers工具包加载模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

text = 'Welcome to '

# 调用模型分词器，对输入文本进行分词并转换为模型可处理的tensor形式
input_ids = tokenizer(text, return_tensors="pt").input_ids

# 其余可修改参数包括top_k, top_p等, 可直接在.generate()方法中调用
# ref: https://huggingface.co/blog/how-to-generate
print(f'\nInput: {text}\n')
print("="*20, "贪婪搜索", "="*20)
for iter in range(5):
    outputs = model.generate(input_ids, max_length=10)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens = True)
    print(f"Iter {iter}: {decoded_output}")
    
print("="*20, "随机搜索, 温度参数=0.1", "="*20)
for iter in range(5):
    outputs = model.generate(input_ids, do_sample=True, temperature=0.1, max_length=10)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens = True)
    print(f"Iter {iter}: {decoded_output}")
    

print("="*20, "随机搜索, 温度参数=1.0", "="*20)
for iter in range(5):
    outputs = model.generate(input_ids, do_sample=True, temperature=1.0, max_length=10)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens = True)
    print(f"Iter {iter}: {decoded_output}")
```

    
    Input: Welcome to 
    
    ==================== 贪婪搜索 ====================
    Iter 0: Welcome to the e-commerce world!
    Iter 1: Welcome to the e-commerce world!
    Iter 2: Welcome to the e-commerce world!
    Iter 3: Welcome to the e-commerce world!
    Iter 4: Welcome to the e-commerce world!
    ==================== 随机搜索, 温度参数=0.1 ====================
    Iter 0: Welcome to the official website of the Institut
    Iter 1: Welcome to the e-commerce world!
    Iter 2: Welcome to the official website of the 
    Iter 3: Welcome to the official website of the 
    Iter 4: Welcome to the world of e-commerce
    ==================== 随机搜索, 温度参数=1.0 ====================
    Iter 0: To the Official Fanpage of Xs
    Iter 1: Browse through our collection of christian
    Iter 2: Gozki - the place to be
    Iter 3: Welcome to Cleo Group online-magazin
    Iter 4: Welcome to the world of The Tweeds



```python

```
