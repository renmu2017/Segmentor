# -*- coding: utf-8 -*-
'''
Created on 2018年  9月12日

@author: WQ
'''
import os,sys
import codecs
import numpy as np
import nltk
import pandas as pd
import pickle
import tensorflow as tf
import tensorboard as tb
from imp import reload
from sklearn.cross_validation import train_test_split
from gensim.models import word2vec
from prettyprinter import cpprint
from tensorflow.python.keras._impl.keras.models import Sequential
from tensorflow.python.keras._impl.keras.layers.embeddings import Embedding
from tensorflow.python.keras._impl.keras.layers.recurrent import LSTM
from tensorflow.python.keras._impl.keras.layers.core import Dropout

WORKPATH = 'E:/WorkSpace0906/Segmentor'
