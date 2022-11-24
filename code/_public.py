from bert_serving.client import BertClient
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from progressbar import *
from scipy import stats
from sklearn.manifold import TSNE
from time import sleep
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import _public as pb
import copy
import json
import logging
import math
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pickle
import random
import scipy.stats
import scipy.stats as stats
import sklearn.metrics as metrics
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re

# All Hyperparameters

Alpha = 0.05
# steretype words percentage
Lambda = 0.05
# fusion parameter
Sigma = 0.3
# debiased parameter

Dataset_Names = ['Twitter']
# ['HyperPartisan','Twitter','ARC','SCIERC','ChemProt','Economy','News','Parties','Yelp_Hotel','IMDB'，'Taobao','Suning']
Round = 1
Fusion = 'SUM_Sigmoid'
# ['SUM_Sigmoid', 'SUM_Tanh'， 'SUM_Linear'， 'None']
Base_Model = 'RoBERTa'
# TextCNN RoBERTa TextRCNN

Init_epoch = 1
# epoch for initial training
Epoch = 1
# epoch for fusion training

Learning_Rate_Init = 5e-4
# lr for initial training
Learning_Rate = 5e-4
# lr for fusion training

Lambda_1 = 0.2
Lambda_2 = 0.05
# parameter of fusion loss

Explain = False

EDA = False
Weight = False

Save_Path = './Results.txt'

# public
Top = 0.4
INF = 999999999
Epsilon = 1e-6
Lowercases = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
Start_Time = ''
Seed = 0
Dataset_Name = ''
XMaxLen = 0
XMaxLenLimit = 300
YList = []
Mask_Token = '[MASK]'
Train_Example_Num_Control = 80000
Use_GPU = True
Tqdm_Len = 80
Train_Distribution = []
# training
Operation_Times = 1
Counterfactual_Input_Control = 'Average'
DataLoader_Shuffle = True
Train_Batch_Size, DevTest_Batch_Size = 32, 32
Embedding_Dimension = 300
Dropout_Rate = 0.1

embedding = None
word2id = None

def random_setting(seed = 100):
    # random seed setting
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def Pickle_Save(variable, path):
    with open(path, 'wb') as file:
        pickle.dump(variable, file)

def Pickle_Read(filepath):
    with open(filepath, 'rb') as file:
        obj = pickle.load(file)
    return obj

def Max_Index(array):
    max_index = 0
    for i in range(len(array)):
        if(array[i]>array[max_index]):
            max_index = i
    return max_index

def Get_Report(true_labels, pred_labels):
    true_labels = [int(v) for v in true_labels]
    pred_labels = [int(v) for v in pred_labels]
    label_list = sorted(list(set(true_labels+pred_labels)))

    macro_f1 = metrics.f1_score(y_true=true_labels, y_pred=pred_labels, average='macro')

    acc = metrics.accuracy_score(true_labels, pred_labels)

    auc = metrics.roc_auc_score(true_labels, pred_labels) if len(label_list)==2 else -0.0

    report_map = {}

    report_map['macro_f1'] = macro_f1

    report_map['acc'] = acc
    report_map['auc'] = auc
    return report_map


def KL(x, y):
    return scipy.stats.entropy(x, y)

def JS(x, y):
    x = np.array(x)
    y = np.array(y)
    z = (x+y)/2.0
    js = 0.5*KL(x,z)+0.5*KL(y,z)
    return js

def Map_To_Sorted_List(map):
    x, y = [], []
    for item in sorted(map.items(), key=lambda item:item[1]):
        x.append(item[0])
        y.append(item[1])
    return x, y

def normalization(data):
    _range = np.max(abs(data))
    return data / _range


def Compute_Fairness(us, v):
    distance, fairness = 0.0, 0.0
    for u in us:
        distance += JS(u, v)
    distance /= len(us)
    fairness = distance * 100.0
    return fairness


def custom_tokenizer(s, return_offsets_mapping=True):

    pos = 0
    offset_ranges = []
    input_ids = []
    for m in re.finditer(r"\W", s):
        start, end = m.span(0)
        offset_ranges.append((pos, start))
        input_ids.append(s[pos:start])
        pos = end
    if pos != len(s):
        offset_ranges.append((pos, len(s)))
        input_ids.append(s[pos:])
    out = {}
    out["input_ids"] = input_ids
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    return out