#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:37:01 2021

@author: zihuiouyang
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#%%
input_shape = (4,16,200)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv1D(
32, 3, activation='relu',input_shape=input_shape[1:])(x) #Embedding convolution
#%%
input_dim = (4,500)
x1 = tf.random.normal(input_shape)
tf.keras.layers.LayerNormalization()
y1 = tf.keras.layers.DepthwiseConv1D(7, 1, depth_multiplier = 128)*4
tf.keras.layers.LayerNormalization()
#Not sure whether you can import from tensorflow.models, but you have attention layer there
tf.keras.layers.LayerNormalization()
y2 = tf.keras.layers.Dense()