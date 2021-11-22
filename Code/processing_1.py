#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 10:11:09 2021

@author: zihuiouyang
"""
# A python script for processing a single text file
import json
import numpy as np
from nltk.tokenize import word_tokenize
f = open('train-v1.1.json',)
data = json.load(f)
data_for_use = data["data"][0]["paragraphs"]
#%%
from nltk.tokenize import word_tokenize
s = word_tokenize(data_for_use[0]["context"])
#%%
embeddings_dict = {}
with open("/Users/zihuiouyang/Documents/glove.6B.300d.txt", 'r') as f: #This needs to be changed if running locally
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
#%%
context_len = 400
for i in range(context_len-len(s)):
    s.append("<pad>")
#%%
np.random.seed(0)
c = []
unk_vector = np.random.uniform(-1,1,300)
np.random.seed(1)
pad_vector = np.random.uniform(-1,1,300)
#%%
vocab = list(embeddings_dict.keys())
#%%
for words in s:
    if words == "<pad>":
        c.append(pad_vector)
    elif words in vocab:
        c.append(embeddings_dict[words])
    else:
        c.append(unk_vector)