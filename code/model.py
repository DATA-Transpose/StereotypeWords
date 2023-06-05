import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from gensim.models import KeyedVectors
from fairseq.models.roberta import RobertaModel
from transformers import RobertaTokenizerFast
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import os
import sys
import random
import math
import copy
import json
import numpy as np
import config as cf
import scipy.stats as stats
import matplotlib.pyplot as plt


class Train:

    def __init__(self, model_x, model_s):
        cf.random_setting()
        self.model_x = model_x.cuda()
        self.model_s = model_s.cuda()
        self.stage = 'Init'
        
    # Initialized training process for model_x using original data
    def Init_Train(self, train_loader, dev_loader, test_loader):
        self.stage = 'Init'
        cf.random_setting(5)
        optimizer = optim.Adam(self.model_x.parameters(), lr=cf.Learning_Rate_Init)
        criterion = nn.CrossEntropyLoss()

        best_dev_bmaf1 = 0
        f_test_bacc = 0
        f_test_bmaf1 = 0
        for i in range(cf.Init_epoch):
            Total_loss = []
            print('Initial training epoch: ', i)

            self.model_x.train()

            for batch_idx, (x, fcx, pcx, y, y_tensor) in enumerate(train_loader):
                y_tensor = y_tensor.cuda()
                optimizer.zero_grad()

                factual_output = self.model_x(x)
                loss = criterion(factual_output, y_tensor)
                loss.backward()
                optimizer.step()
                Total_loss.append(loss.cpu().item())
                    
            print('training_loss: ', np.mean(Total_loss))
            
            self.model_x.eval()
            # test on dev set to select the best model
            with torch.no_grad():  
                true_labels, factual_outputs = [], []       
                for batch_idx, (x, fcx, pcx, y, y_tensor) in enumerate(dev_loader):
                    true_labels.extend(y_tensor.cpu().data.numpy())

                    factual_outputs.extend(self.model_x(x).cpu().data.numpy())

                baseline_labels = torch.max(torch.Tensor(factual_outputs), 1)[1]
                baseline_labels = baseline_labels.numpy()
                if cf.Use_GPU == True:
                    torch.cuda.empty_cache()
    
                dev_bacc = cf.Get_Report(true_labels, baseline_labels)['acc']
                dev_bmaf1 = cf.Get_Report(true_labels, baseline_labels)['macro_f1']
                
                if dev_bmaf1 > best_dev_bmaf1:
                    true_labels, factual_outputs = [], []
                    best_dev_bmaf1 = dev_bmaf1
                    
                    torch.save(self.model_x.state_dict(),cf.Base_Model + cf.Dataset_Name + cf.Stereotype + 'xinit.pt')
                    
                    for batch_idx, (x, fcx, pcx, y, y_tensor) in enumerate(test_loader):
                        true_labels.extend(y_tensor.cpu().data.numpy())

                        factual_outputs.extend(self.model_x(x).cpu().data.numpy())
                        
                    baseline_labels = torch.max(torch.Tensor(factual_outputs), 1)[1]
                    baseline_labels = baseline_labels.numpy()
                    if cf.Use_GPU == True:
                        torch.cuda.empty_cache()
        
                    f_test_bacc = cf.Get_Report(true_labels, baseline_labels)['acc']
                    f_test_bmaf1 = cf.Get_Report(true_labels, baseline_labels)['macro_f1']
                        
                    init_factual_keyword_fairness, _ = self.Fairness(test_loader)
 
        self.model_x.train()
        print('******Initial Training finished**********')

        return f_test_bacc, f_test_bmaf1, init_factual_keyword_fairness
    
    # Training process for model_x and biased model model_s using masked data
    def Train(self, train_loader, dev_loader, test_loader, seed = 0):
        self.stage = 'Train'
        cf.random_setting(seed)

        optimizer = optim.Adam([{'params': self.model_x.parameters(), 'lr': cf.Learning_Rate},
                                 {'params': self.model_s.parameters(), 'lr': cf.Learning_Rate}])

        criterion = nn.CrossEntropyLoss()
        if cf.Use_GPU == True:
            criterion = criterion.cuda()
            torch.cuda.empty_cache()
            self.model_x = self.model_x.cuda()
            self.model_s = self.model_s.cuda()
        f_test_maf1 = 0
        f_test_bmaf1 = 0
        f_dev_fmaf1 = 0

        for i in range(cf.Epoch):
            Total_loss0 = []
            Total_loss1 = []
            Total_loss2 = []
            Total_loss = []
            if i == 0:
                self.Evaluate(dev_loader, test_loader, i)
            self.model_x.train()
            self.model_s.train()
            trainbar, true_labels, factual_labels = tqdm(total=len(train_loader), ncols=cf.Tqdm_Len), [], []
            for batch_idx, (x, fcx, pcx, y, y_tensor) in enumerate(train_loader):

                # if b.Base_Model == 'TextCNN' or cf.Base_Model == 'TextRCNN':
                #     if cf.Explain == False:
                #         x = self.model_x.token(x).cuda()
                #         pcx = self.model_s.token(pcx).cuda()
                #     else:
                #         x = self.model_x.token(x)
                #         pcx = self.model_s.token(pcx)

                optimizer.zero_grad()
                if cf.Fusion == 'SUM_Sigmoid':
                    factual_output = self.model_x(x) + cf.Lambda * torch.sigmoid(self.model_s(pcx))
                elif cf.Fusion == 'SUM_Tanh':
                    factual_output = self.model_x(x) + cf.Lambda * torch.tanh(self.model_s(pcx))
                elif cf.Fusion == 'SUM_Linear':
                    factual_output = self.model_x(x) + cf.Lambda * self.model_s(pcx)
                elif cf.Fusion == 'None':
                    factual_output = self.model_x(x)
                semantic_output = self.model_x(x)
                word_output = self.model_s(pcx)

                loss0 = criterion(factual_output, y_tensor)
                loss1 = criterion(semantic_output, y_tensor)
                loss2 = criterion(word_output, y_tensor)
                loss = loss0 + cf.Lambda_1 * loss1 + cf.Lambda_2 * loss2

                loss.backward()
                optimizer.step()

                if cf.Use_GPU == True:
                    torch.cuda.empty_cache()

                Total_loss0.append(loss0.item())
                Total_loss1.append(loss1.item())
                Total_loss2.append(loss2.item())
                Total_loss.append(loss.item())
                trainbar.set_description(cf.Dataset_Name + ' Training_Epoch={}'.format(i + 1))
                trainbar.update(1)

            trainbar.close()
            print('training_loss0: ', np.mean(Total_loss0))
            print('training_loss1: ', np.mean(Total_loss1))
            print('training_loss2: ', np.mean(Total_loss2))
            print('training_loss: ', np.mean(Total_loss))
            test_acc, test_bacc, test_maf1, test_bmaf1, dev_fmaf1, test_dev_maf1 = self.Evaluate(dev_loader, test_loader, i + 1)
            factual_keyword_fairness, counterfactual_keyword_fairness = self.Fairness(test_loader)
            if f_dev_fmaf1 <= dev_fmaf1:
                torch.save(self.model_x.state_dict(),cf.Base_Model + cf.Dataset_Name + cf.Stereotype + 'xdebias.pt')
                torch.save(self.model_s.state_dict(),cf.Base_Model + cf.Dataset_Name + cf.Stereotype + 'sdebias.pt')
                f_dev_fmaf1 = dev_fmaf1
                f_test_maf1 = test_maf1
                f_test_bmaf1 = test_bmaf1
                f_test_acc = test_acc
                f_test_bacc = test_bacc
                f_fairness = counterfactual_keyword_fairness
                f_bfairness = factual_keyword_fairness


        return f_test_acc, f_test_bacc, f_test_maf1, f_test_bmaf1, f_fairness, f_bfairness

    
    def Evaluate(self, dev_loader, test_loader, mark=''):
        if cf.Use_GPU == True:
            torch.cuda.empty_cache()
            self.model_x = self.model_x.cuda()
            self.model_s = self.model_s.cuda()

        best_sigma = 1
        base = 0
        for sigma in range(0,41):
            cf.Sigma = sigma/20.0
            dev_acc, dev_bacc, dev_maf1, dev_bmaf1 = self.Test_maF1(dev_loader)
            if dev_maf1 > base:
                base = dev_maf1
                best_sigma = cf.Sigma
        cf.Sigma = best_sigma

        test_acc, test_bacc, test_maf1, test_bmaf1 = self.Test_maF1(test_loader)

        self.Save(dev_maf1,test_acc, test_bacc, test_maf1, test_bmaf1, mark=mark)
        return test_acc, test_bacc, test_maf1, test_bmaf1, dev_maf1, dev_bmaf1


    def Test_maF1(self, test_loader):
        if cf.Use_GPU == True: torch.cuda.empty_cache()
        if cf.Use_GPU == True:
            torch.cuda.empty_cache()
            self.model_x = self.model_x.cuda()
            self.model_s = self.model_s.cuda()

        true_labels, factual_outputs, counterfactual_outputs = [], [], []
        self.model_x.eval()
        self.model_s.eval()

        with torch.no_grad():
            for batch_idx, (x, fcx, pcx, y, y_tensor) in enumerate(test_loader):
                true_labels.extend(y_tensor.cpu().data.numpy())

                if cf.Fusion == 'SUM_Sigmoid':
                    factual_outputs.extend((self.model_x(x) + cf.Lambda * torch.sigmoid(self.model_s(pcx))).cpu().data.numpy())
                    counterfactual_outputs.extend((self.model_x(fcx) + cf.Lambda * torch.sigmoid(self.model_s(pcx))).cpu().data.numpy())
                elif cf.Fusion == 'SUM_Tanh':
                    factual_outputs.extend((self.model_x(x) + cf.Lambda * torch.tanh(self.model_s(pcx))).cpu().data.numpy())
                    counterfactual_outputs.extend((self.model_x(fcx) + cf.Lambda * torch.tanh(self.model_s(pcx))).cpu().data.numpy())
                elif cf.Fusion == 'SUM_Linear':
                    factual_outputs.extend((self.model_x(x) + cf.Lambda * self.model_s(pcx)).cpu().data.numpy())
                    counterfactual_outputs.extend((self.model_x(fcx)+ cf.Lambda * self.model_s(pcx)).cpu().data.numpy())
                elif cf.Fusion == 'None':
                    factual_outputs.extend(self.model_x(x).cpu().data.numpy())
                    counterfactual_outputs.extend(self.model_x(fcx).cpu().data.numpy())
                if cf.Use_GPU == True:
                    torch.cuda.empty_cache()

            predict_labels, baseline_labels = self.Counterfactual_Predict(factual_outputs, counterfactual_outputs)
            bacc = cf.Get_Report(true_labels, baseline_labels)['acc']
            acc = cf.Get_Report(true_labels, predict_labels)['acc']
            bmaf1 = cf.Get_Report(true_labels, baseline_labels)['macro_f1']
            cmaf1 = cf.Get_Report(true_labels, predict_labels)['macro_f1']

        return acc, bacc, cmaf1, bmaf1

    def Counterfactual_Predict(self, factual_outputs, counterfactual_output):
        factual_outputs = np.array(factual_outputs)
        counterfactual_output = np.array(counterfactual_output)
        debiased_outputs = factual_outputs - cf.Sigma * counterfactual_output
        predicted_labels = torch.max(torch.Tensor(debiased_outputs), 1)[1]
        baseline_labels = torch.max(torch.Tensor(factual_outputs), 1)[1]
        return predicted_labels.numpy(), baseline_labels.numpy()

    def Fairness(self, test_loader):

        if cf.Use_GPU == True:
            torch.cuda.empty_cache()

        texts = []
        factual_outputs = []
        counterfactual_outputs = []
        self.model_x.eval()
        self.model_s.eval()
        with torch.no_grad():
            for batch_idx, (x, fcx, pcx, y, y_tensor) in enumerate(test_loader):
                
                texts.extend(x)
                if self.stage == 'Init':
                    factual_outputs.extend((self.model_x(x)).cpu().data.numpy())
                    counterfactual_outputs.extend((self.model_x(fcx)).cpu().data.numpy())
                else:
                    factual_outputs.extend((self.model_x(x) + cf.Lambda * torch.sigmoid(self.model_s(pcx))).cpu().data.numpy())
                    counterfactual_outputs.extend((self.model_x(fcx) + cf.Lambda * torch.sigmoid(self.model_s(pcx))).cpu().data.numpy())

                if cf.Use_GPU == True:
                    torch.cuda.empty_cache()

            counterfactual_labels, factual_labels = self.Counterfactual_Predict(factual_outputs, counterfactual_outputs)

            if cf.Use_GPU == True:
                    torch.cuda.empty_cache()

        factual_keyword_distributions, counterfactual_keyword_distributions, w2c = [], [], {}

        for i, text in enumerate(texts):

            for word in text.split(' '):
                
                if w2c.get(word) is None:
                    w2c[word] = [[0 for _ in cf.YList], [0 for _ in cf.YList]]


                w2c[word][0][factual_labels[i]] += 1

                w2c[word][1][counterfactual_labels[i]] += 1

        for word in w2c.keys():
            u = w2c[word][0]
            su = sum(u)
            u = [value / su for value in u]
            v = w2c[word][1]
            sv = sum(v)
            v = [value / sv for value in v]
            #print(u)
            #print(v)
            factual_keyword_distributions.append(u)
            counterfactual_keyword_distributions.append(v)

        uniform = [1.0 / len(cf.YList) for _ in cf.YList]

        factual_keyword_fairness = cf.Compute_Fairness(factual_keyword_distributions, uniform)
        counterfactual_keyword_fairness = cf.Compute_Fairness(counterfactual_keyword_distributions, uniform)

        return factual_keyword_fairness, counterfactual_keyword_fairness


    def Save(self, dev_maf1, test_acc, test_bacc, test_maf1, test_bmaf1, mark=''):
        id = '{}-{}-Seed={}-Epoch={} \n'.format(cf.Dataset_Name, self.model_x.name, cf.Seed, mark)

        report = '{} | \n'.format(id)
        report += 'dev_counterfactual_maf1={:.2%} | \n'.format(dev_maf1)
        report += 'test_counterfactual_acc={:.2%}| \n'.format(test_bacc)
        report += 'test_factual_acc={:.2%}| \n'.format(test_acc)
        report += 'test_counterfactual_maf1={:.2%}| \n'.format(test_maf1)
        report += 'test_factual_maf1={:.2%}| \n'.format(test_bmaf1)
        writer = open(cf.Save_Path, 'a+')
        writer.write(report + '\n')
        writer.close()
        print(report)



class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.name = 'TextCNN'
        self.sentence_max_size = cf.XMaxLen
        self.label_size = len(cf.YList)
        self.lr = cf.Learning_Rate
        self.emb_dim = 300
        self.shape = (self.sentence_max_size, self.emb_dim)

        self.filter_num = 100
        self.kernel_list = [1,2,3,4,5]
        self.chanel_num = 1

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.chanel_num, self.filter_num, (kernel, self.emb_dim)),
            nn.ReLU(),
            nn.MaxPool2d((self.sentence_max_size - kernel + 1, 1))
        ) for kernel in self.kernel_list])

        self.dropout = nn.Dropout(cf.Dropout_Rate)

        self.fc = nn.Linear(self.filter_num * len(self.kernel_list), self.label_size)

    def forward(self, text):

        if cf.Explain == False:
            xs = token(text).cuda()
        else:
            xs = token(text)

        xs = xs.unsqueeze(1)
        in_size = xs.size(0)

        #print(xs.grad)
        out = [conv(xs) for conv in self.convs]

        out = torch.cat(out, dim=1)

        #print(out.grad)
        out = out.view(in_size, -1)
        #print(out.grad)
        out = F.dropout(out)
        out = self.fc(out)
        #print(out.grad)
        return out

def token(text):

    xs = torch.zeros([len(text), cf.XMaxLen, 300])
    for b_idx in range(len(text)):
        for index in range(0, cf.XMaxLen):
            if index >= len(text[b_idx]):
                break
            else:
                word = text[b_idx][index]
                if word in cf.word2id:
                    vector = cf.embedding.weight[cf.word2id[word]]
                    xs[b_idx][index] = vector
                elif word.lower() in cf.word2id:
                    xs = cf.embedding.weight[cf.word2id[word.lower()]]
                    xs[b_idx][index] = vector

    return xs



class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return torch.max_pool1d(x, kernel_size=x.shape[-1])

class TextRCNN(nn.Module):
    def __init__(self):
        super(TextRCNN, self).__init__()
        self.name = 'TextRCNN'
        self.sentence_max_size = cf.XMaxLen
        self.label_size = len(cf.YList)

        self.lstm = nn.LSTM(input_size=300, hidden_size=256,
                            batch_first=True, bidirectional=True)
        self.globalmaxpool = GlobalMaxPool1d()
        self.dropout = nn.Dropout(.5)
        self.linear1 = nn.Linear(300 + 2 * 256, 256)
        self.linear2 = nn.Linear(256, self.label_size)


    def forward(self, text):

        if cf.Explain == False:
            xs = token(text).cuda()
        else:
            xs = token(text)

        last_hidden_state, (c, h) = self.lstm(xs)
        out = torch.cat((xs, last_hidden_state),2)
        out = F.relu(self.linear1(out))
        out = out.permute(dims=[0, 2, 1])
        out = self.globalmaxpool(out).squeeze(-1)
        out = self.dropout(out)
        out = self.linear2(out)
        return out


class RoBERTa(nn.Module):
    def __init__(self):
        super(RoBERTa, self).__init__()
        self.name = 'RoBERTa'
        self.sentence_max_size = cf.XMaxLen
        self.label_size = len(cf.YList)

        self.lr = cf.Learning_Rate
        self.emb_dim = 768
        self.shape = self.emb_dim

        self.hidden_layer_1 = 256
        self.hidden_layer_2 = 128
        self.hidden_layer = 100

        self.roberta = RobertaModel.from_pretrained('./code/roberta.base', checkpoint_file='model.pt')
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

        for p in self.parameters():
            p.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.hidden_layer),
            nn.Tanh(),
            nn.Dropout(cf.Dropout_Rate),
            nn.Linear(self.hidden_layer, self.label_size)
        )

    def forward(self, x):
        tokens = torch.tensor([self.tokenizer.encode(v, padding='max_length', max_length=256, truncation=True) for v in x]).cuda()

        last_layer_feature = self.roberta.extract_features(tokens)[:,-1,:]
        #print(last_layer_feature.grad)
        out = self.mlp(last_layer_feature)
        #print(out.grad)
        return out

