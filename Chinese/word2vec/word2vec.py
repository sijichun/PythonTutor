#!/usr/bin/env python
# coding: utf-8

import jieba

jieba.load_userdict("Econ_Dict.txt")

import re

with open('stopword.txt','rt') as f:
    stoplist=f.readlines()
    stoplist=[w.replace('\n','') for w in stoplist]

def not_digit(w):
    w=w.replace(',','')
    if re.match(r'\d+',w)!=None or re.match(r'\d%',w)!=None or re.match(r'\d*\.\d+',w)!=None:
        return False
    else:
        return True

def tokenize(w):
    cut_w=jieba.cut(w)
    ## 去除停用词
    cut_w=[w.strip().lower() for w in cut_w if ((w not in stoplist) and not_digit(w) and len(w.strip())>0)]
    return cut_w

import os

Words=[]

folder = os.listdir('./')
for fo in folder:
    if os.path.isdir('./'+fo):
        files = os.listdir('./'+fo)
        for fi in files:
            if '.txt' in fi:
                print('./'+fo+'/'+fi)
                with open('./'+fo+'/'+fi,'r', encoding='gb18030', errors='ignore') as f:
                    for t in f:
                        tokenized_words=tokenize(t)
                        with open('ALL_WORDS', 'at') as f:
                            f.write(' '.join(tokenized_words)+'\n')
                        Words.append(tokenized_words)
import gensim

print("共%s个词" % len(Words))

print('30 started')
w2v_model=gensim.models.Word2Vec(Words, window=6, size=30, min_count=10, workers=40, batch_words=30000)
w2v_model.save('word2vec30')

print('100 started')
w2v_model=gensim.models.Word2Vec(Words, window=6, size=100, min_count=10, workers=40, batch_words=30000)
w2v_model.save('word2vec100')

print('300 started')
w2v_model=gensim.models.Word2Vec(Words, window=6, size=300, min_count=10, workers=40, batch_words=30000)
w2v_model.save('word2vec300')

print('500 started')
w2v_model=gensim.models.Word2Vec(Words, window=6, size=500, min_count=10, workers=40, batch_words=30000)
w2v_model.save('word2vec500')
