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
from nltk.corpus import words
import time
#%%
f = open('train-v1.1.json',)
data = json.load(f)
data_for_use = data["data"][0]["paragraphs"]
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
np.random.seed(0)
c = []
unk_vector = np.random.uniform(-1,1,300)
np.random.seed(1)
pad_vector = np.random.uniform(-1,1,300)
vocab = list(embeddings_dict.keys())
#%%
for word in s:
    if word == "<pad>":
        c.append(pad_vector)
    elif word in vocab:
        c.append(embeddings_dict[word])
    else:
        c.append(unk_vector)
#%%
matrix = np.zeros((26,200))
for i in range(26):
    np.random.seed(i)
    matrix[i] = np.random.uniform(-1,1,200)
b = np.nanmax(matrix, axis = 0)
np.random.seed(26)
char_pad_vector = np.random.uniform(-1,1,200)
w = words.words()
l = len(c)
#%%
a = time.time()
for h in range(l):
    word = s[h]
    mat = np.zeros((16,200))
    if word in w:
        word = word.lower()
        word_length = len(word)
        if word_length<16:
            for i in range(word_length):
                n = ord(word[i]) - 97
                mat[i] = matrix[n]
            for j in range(16-word_length):
                mat[len(word)+j] = char_pad_vector
        else:
            for i in range(16):
                n = ord(word[i]) - 97
                mat[i] = matrix[n]
    else:
        for i in range(16):
            mat[i] = char_pad_vector
    c[h] = np.concatenate((c[h], np.nanmax(mat, axis = 0)))
d = time.time()-a           
            
            
            
            