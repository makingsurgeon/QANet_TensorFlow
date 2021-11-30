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
vectors = {}
with open("/Users/zihuiouyang/Documents/glove.6B.300d.txt", 'r') as f:
    for line in f:
        line_split = line.strip().split(" ")
        vec = np.array(line_split[1:], dtype=float)
        word = line_split[0]

        for char in word:
            if ord(char) < 128:
                if char in vectors:
                    vectors[char] = (vectors[char][0] + vec,
                                     vectors[char][1] + 1)
                else:
                    vectors[char] = (vec, 1)
a = {}
for word in vectors:
    avg_vector = np.round((vectors[word][0] / vectors[word][1]), 6).tolist()
    a[word] = avg_vector[:200]
embeddings_dict = {}
with open("/Users/zihuiouyang/Documents/glove.6B.300d.txt", 'r') as f: #This needs to be changed if running locally
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
#%%
f = open('train-v1.1.json',)
data = json.load(f)
data_for_use = data["data"][0]["paragraphs"]
#%%
context_len = 400
np.random.seed(0)
unk_vector = np.random.uniform(-1,1,300)
np.random.seed(1)
pad_vector = np.random.uniform(-1,1,300)
vocab = list(embeddings_dict.keys())
#%%
np.random.seed(26)
char_pad_vector = np.random.uniform(-1,1,200)
w = words.words()
#%%
v = []
t = time.time()
for i in range(len(data_for_use)):
    s = word_tokenize(data_for_use[i]["context"])
    for i in range(context_len-len(s)):
        s.append("<pad>")
    c = []
    for word in s:
        if word == "<pad>":
            c.append(pad_vector)
        elif word in vocab:
            c.append(embeddings_dict[word])
        else:
            c.append(unk_vector)
    l = len(c)
    d = []
    for h in range(l):
        word = s[h]
        mat = np.zeros((16,200))
        if word in w:
            word = word.lower()
            word_length = len(word)
            if word_length<16:
                for i in range(word_length):
                    mat[i] = a[word[i]]
                for j in range(16-word_length):
                    mat[len(word)+j] = char_pad_vector
            else:
                for i in range(16):
                    mat[i] = a[word[i]]
        else:
            for i in range(16):
                mat[i] = char_pad_vector
        d.append(np.nanmax(mat,axis = 0))
    v.append(d)
diff = time.time()-t



            
            