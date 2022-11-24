import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
# import logging
import _public as pb
# from progressbar import *
from tqdm import tqdm

import random
import math
import copy
import scipy.stats as stats
import json
import shap
from transformers import RobertaTokenizerFast


class Example:
    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.fully_counterfactual_text = []
        self.partial_counterfactual_text = []

class MyAllDataset():
    def __init__(self, dataset_name):
        pb.random_setting()
        self.dataset_name = dataset_name
        self.train_examples = []
        self.dev_examples = []
        self.test_examples = []
        self.Read_Data()

    # read data
    def Read_from_Datapath(self, data_path):

        examples = []
        for line in open(data_path).read().split('\n'):
            if '{' in line:
                linemap = json.loads(line.lower())
                if len(linemap['text'].strip()) > 0 and len(linemap['label'].strip()) > 0:
                    examples.append(Example(linemap['text'], linemap['label']))
        return examples

    # Conform
    def Conform_Dev_Test(self, dev_examples, test_examples):
        examples = dev_examples + test_examples
        label2examples = {}
        for example in examples:
            label = example.label
            if label not in label2examples:
                label2examples[label] = []
            label2examples[label].append(example)
        dev_examples_, test_examples_ = [], []
        for key in label2examples.keys():
            subexamples = label2examples[key]
            random.shuffle(subexamples)
            seperator = int(len(subexamples) / 2)
            dev_examples_.extend(subexamples[:seperator])
            test_examples_.extend(subexamples[seperator:])

        return dev_examples_, test_examples_

    # initialize
    def Init_Public(self, train_examples, dev_examples, test_examples):
        examples = train_examples + dev_examples + test_examples


        class_count = {}

        for i, example in enumerate(examples):
            if class_count.get(example.label) == None:
                class_count[example.label] = 1
            else:
                class_count[example.label] += 1

            example.text = example.text.split(' ')

            example.text = [word.strip() for word in example.text if len(word.strip()) > 0]
            example.text = ' '.join(example.text)

            for j in range(len(example.text)):
                example.fully_counterfactual_text.append(pb.Mask_Token)
                example.partial_counterfactual_text.append(pb.Mask_Token)


        for x in [example.text for example in examples]:
            pb.XMaxLen = min(max(pb.XMaxLen, len(x)), pb.XMaxLenLimit)

        self.train_examples = train_examples
        self.dev_examples = dev_examples
        self.test_examples = test_examples

    # initialize
    def Public(self, train, train_loader):

        class_count = {}

        for i, example in enumerate(self.train_examples):
            if class_count.get(example.label) == None:
                class_count[example.label] = 1
            else:
                class_count[example.label] += 1

        model = train.model_x
        if pb.Base_Model == 'RoBERTa' or pb.Base_Model == 'GPT2':
            masker = RobertaTokenizerFast.from_pretrained('roberta-base')
                #shap.maskers.Text(pb.custom_tokenizer)
        elif pb.Base_Model == 'TextCNN' or pb.Base_Model == 'TextRCNN':
            masker = shap.maskers.Text(r"\W")
            model = model.cpu()
            pb.Explain = True
        explainer = shap.Explainer(model, masker, output_names=pb.YList, seed=1)
        model.eval()
        word_shap = {}
        important_words = []
        for batch_idx, (x, fcx, pcx, y, y_tensor) in enumerate(train_loader):

            print(batch_idx)
            shap_values = explainer(x)
            for i in range(len(x)):
                shap_value = pb.normalization(shap_values.values[i][:, int(y[i])])
                # print(shap_value)
                # print(len(shap_value))
                # print(shap_values.data)
                # print(shap_values.data[i])
                # print(len(shap_values.data[i]))
                # for idx in range(len(shap_values.data[i])):
                #
                #     if shap_value[idx] > 0:
                #         word = shap_values.data[i][idx].strip()
                #         if len(word) > 0:
                #             word_shap[word] = shap_value[idx]
                important_words += list(set([shap_values.data[i][idx].strip() for idx in range(len(shap_value))
                                             if shap_value[idx] > 0 and len(shap_values.data[i][idx].strip()) > 0]))
        pb.Explain = False
        model = model.cuda()

        #important_words = list(word_shap.keys())



        stereotype_words = []
        keyword_entropy = {}
        for keyword in important_words:
            keyword_class_count = {}
            for i, example in enumerate(self.train_examples):
                if keyword not in example.text:
                    pass
                else:
                    if keyword_class_count.get(example.label) == None:
                        keyword_class_count[example.label] = 1
                    else:
                        keyword_class_count[example.label] += 1
            keyword_class_percentage = {}
            keyword_sum = sum(list(keyword_class_count.values()))
            entropy = 0
            for _class in class_count.keys():
                keyword_class_percentage[_class] = keyword_class_count.get(_class, 0) / keyword_sum
                entropy -= keyword_class_percentage[_class] * np.log(keyword_class_percentage[_class] + 1e-8)

            keyword_entropy[keyword] = entropy

        stereotype_word_list = list(keyword_entropy.values())
        stereotype_word_list.sort(reverse=True)
        boundary = stereotype_word_list[int(len(stereotype_word_list) * pb.Alpha)]

        for keyword in keyword_entropy:
            if keyword_entropy[keyword] > boundary:
                stereotype_words.append(keyword)
        print('stereotype_words lengths: ', len(stereotype_words))

        for i, example in enumerate(self.train_examples + self.dev_examples + self.test_examples):
            example.fully_counterfactual_text = []
            example.partial_counterfactual_text = []
            text = example.text.split(' ')
            for j in range(len(text)):
                word = text[j].strip()
                example.fully_counterfactual_text.append(pb.Mask_Token)
                if word not in important_words or word not in stereotype_words:
                    example.partial_counterfactual_text.append(pb.Mask_Token)
                else:
                    example.partial_counterfactual_text.append(word)



    def Read_Data(self, init_train = False, train = None, train_loader = None):

        if init_train == False:
            # output dataset's name
            print('Dataset:{}'.format(self.dataset_name))


            if pb.EDA==False:
                train_datapath = './data/' + self.dataset_name + '.train.jsonl'
            else:

                train_datapath = './data/data.eda/' + self.dataset_name + '.train.eda.jsonl'
            dev_datapath = './data/' + self.dataset_name + '.dev.jsonl'
            test_datapath = './data/' + self.dataset_name + '.test.jsonl'

            train_examples = self.Read_from_Datapath(train_datapath)
            dev_examples = self.Read_from_Datapath(dev_datapath)
            test_examples = self.Read_from_Datapath(test_datapath)

            pb.YList = sorted(list(set([example.label for example in train_examples + dev_examples + test_examples])))


            dev_examples, test_examples = self.Conform_Dev_Test(dev_examples, test_examples)

            # analysis
            random.shuffle(train_examples)
            random.shuffle(dev_examples)
            random.shuffle(test_examples)
            trLen, deLen, teLen = len(train_examples), len(dev_examples), len(test_examples)
            train_examples = train_examples[:min(len(train_examples), pb.Train_Example_Num_Control)]
            dev_examples = dev_examples[:min(len(dev_examples), int(len(train_examples)*1.0/trLen*deLen))]
            test_examples = test_examples[:min(len(test_examples), int(len(train_examples)*1.0/trLen*teLen))]
            trLen, deLen, teLen = len(train_examples), len(dev_examples), len(test_examples)
            alLen = trLen + deLen + teLen
            print('#train_examples: {}({:.2%})'.format(trLen, trLen * 1.0 / alLen))
            print('#dev_examples: {}({:.2%})'.format(deLen, deLen * 1.0 / alLen))
            print('#test_examples: {}({:.2%})'.format(teLen, teLen * 1.0 / alLen))


            self.Init_Public(train_examples, dev_examples, test_examples)

            print('pb.XMaxLen={}'.format(pb.XMaxLen))
            print('pb.YList={} {}'.format(len(pb.YList), pb.YList))


            # probability distributions
            train_distribution = pb.Train_Distribution = [0 for _ in range(len(pb.YList))]
            dev_distribution = [0 for _ in range(len(pb.YList))]
            test_distribution = [0 for _ in range(len(pb.YList))]
            for e in self.train_examples: train_distribution[pb.YList.index(e.label)] += 1
            for e in self.dev_examples:   dev_distribution[pb.YList.index(e.label)] += 1
            for e in self.test_examples:  test_distribution[pb.YList.index(e.label)] += 1
            train_distribution = [x * 1.0 / sum(train_distribution) for x in train_distribution]
            dev_distribution = [x * 1.0 / sum(dev_distribution) for x in dev_distribution]
            test_distribution = [x * 1.0 / sum(test_distribution) for x in test_distribution]
            print('train_distribution: [', end='')
            for v in train_distribution: print('{:.2%}'.format(v), end=' ')
            print('] {}'.format('Balanced' if pb.EDA == True else 'Raw'))
            print('dev_distribution:   [', end='')
            for v in dev_distribution: print('{:.2%}'.format(v), end=' ')
            print(']')
            print('test_distribution:  [', end='')
            for v in test_distribution: print('{:.2%}'.format(v), end=' ')
            print(']')



        else:
            self.Public(train, train_loader)

            # MASK ratio
            examples = self.train_examples + self.dev_examples + self.test_examples
            Ratio = 0.0
            for example in examples:
                up = len([word for word in example.partial_counterfactual_text if word == pb.Mask_Token])
                down = len(example.text)
                Ratio += up * 1.0 / down
            Ratio = Ratio * 1.0 / len(examples)
            print('{:.2%} MASKed ({:.2%} is context)'.format(Ratio, 1.0 - Ratio))


    def Word_Detection(self, train_loader, train):

        self.Read_Data(True, train, train_loader)

class MyDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        index %= self.__len__()

        x = self.examples[index].text
        fcx = self.examples[index].fully_counterfactual_text
        pcx = self.examples[index].partial_counterfactual_text
        fcx = " ".join(fcx)
        pcx = " ".join(pcx)

        y = self.examples[index].label
        y_tensor = self.Generate_Y_Tensor(y)
        return x, fcx, pcx, y,  y_tensor

    def Generate_Y_Tensor(self, label):
        tensor = torch.zeros(len(pb.YList))
        tensor[pb.YList.index(label)] = 1
        tensor = torch.argmax(tensor)
        if pb.Use_GPU == True:
            tensor = tensor.cuda()
        return tensor




