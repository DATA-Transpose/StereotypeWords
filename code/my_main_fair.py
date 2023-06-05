import data_process
import model
import torch
from torch.utils.data import Dataset, DataLoader
import os
import warnings
import copy
import config as cf
import sys

def MAIN():


    # settings
    cf.XMaxLen = 0
    cf.YList = []
    cf.Use_GPU = torch.cuda.is_available()

    # random seed setting
    cf.random_setting(0)

    # output configuration
    print('cf.Dataset_Name={}'.format(cf.Dataset_Name))
    print('cf.Base_Model={}'.format(cf.Base_Model))

    cf.Pretrained = True

    # prepare a specific dataset
    TextDataset = data_process.TextDataset(cf.Dataset_Name)
    # build model for intialized training
    if cf.Base_Model == 'TextCNN' or cf.Base_Model == 'TextRCNN':

        [cf.embedding, cf.word2id] = cf.Pickle_Read('./w2v/glove.300d.en.txt.pickle')
        cf.embedding.weight.requires_grad = False

    if cf.Base_Model == 'TextCNN':

        model_x = model.TextCNN()
        model_s = model.TextCNN()

    elif cf.Base_Model == 'TextRCNN':

        model_x = model.TextRCNN()
        model_s = model.TextRCNN()


    elif cf.Base_Model == 'RoBERTa':
        model_x = model.RoBERTa()
        model_s = model.RoBERTa()

    if cf.Use_GPU == True:
        model_x = model_x.cuda()
        model_s = model_s.cuda()

    tr = model.Train(model_x, model_s)
    tr.stage = 'Init'
    tr.model_x.load_state_dict(torch.load(cf.Base_Model + cf.Dataset_Name + cf.Stereotype + 'xinit.pt'))
    #tr.model_s.load_state_dict(torch.load(cf.Base_Model + cf.Dataset_Name + cf.Stereotype + 'sdebias.pt'))
    train_dataset = data_process.TrainDataset(TextDataset.train_examples)
    test_dataset = data_process.TrainDataset(TextDataset.test_examples)

    train_loader = DataLoader(train_dataset, batch_size=cf.Train_Batch_Size, shuffle=cf.DataLoader_Shuffle)
    test_loader = DataLoader(test_dataset, batch_size=cf.DevTest_Batch_Size, shuffle=cf.DataLoader_Shuffle)
    
    TextDataset.Word_Detection(train_loader, tr)

    train_dataset = data_process.TrainDataset(TextDataset.train_examples)
    test_dataset = data_process.TrainDataset(TextDataset.test_examples)

    train_loader = DataLoader(train_dataset, batch_size=cf.Train_Batch_Size, shuffle=cf.DataLoader_Shuffle)
    test_loader = DataLoader(test_dataset, batch_size=cf.DevTest_Batch_Size, shuffle=cf.DataLoader_Shuffle)
    factual_keyword_fairness1, counterfactual_keyword_fairness1 = tr.Fairness(test_loader)
    
    
    
    
    tr = model.Train(model_x, model_s)
    tr.stage = 'Train'
    tr.model_x.load_state_dict(torch.load(cf.Base_Model + cf.Dataset_Name + cf.Stereotype + 'xdebias.pt'))
    tr.model_s.load_state_dict(torch.load(cf.Base_Model + cf.Dataset_Name + cf.Stereotype + 'sdebias.pt'))
    train_dataset = data_process.TrainDataset(TextDataset.train_examples)
    dev_dataset = data_process.TrainDataset(TextDataset.dev_examples)
    test_dataset = data_process.TrainDataset(TextDataset.test_examples)

    train_loader = DataLoader(train_dataset, batch_size=cf.Train_Batch_Size, shuffle=cf.DataLoader_Shuffle)
    test_loader = DataLoader(test_dataset, batch_size=cf.DevTest_Batch_Size, shuffle=cf.DataLoader_Shuffle)
    
    TextDataset.Word_Detection(train_loader, tr)

    train_dataset = data_process.TrainDataset(TextDataset.train_examples)
    dev_dataset = data_process.TrainDataset(TextDataset.dev_examples)
    test_dataset = data_process.TrainDataset(TextDataset.test_examples)

    train_loader = DataLoader(train_dataset, batch_size=cf.Train_Batch_Size, shuffle=cf.DataLoader_Shuffle)
    test_loader = DataLoader(test_dataset, batch_size=cf.DevTest_Batch_Size, shuffle=cf.DataLoader_Shuffle)
    factual_keyword_fairness, counterfactual_keyword_fairness = tr.Fairness(test_loader)
    print('************factual fairness: ********', factual_keyword_fairness)
    print('************counterfactual fairness: ********', counterfactual_keyword_fairness)
    print('************base factual fairness: ********', factual_keyword_fairness1)
    print('************counterfactual fairness: ********', counterfactual_keyword_fairness1)
    

        
if __name__ == "__main__":

    # set path
    os.chdir("/home/code/")

    # read configuration
    print('sys.argv={}'.format(sys.argv))

    # Read Dataset and Model info
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--Dataset_Name' and i+1<len(sys.argv):
            cf.Dataset_Names = [sys.argv[i+1]]
        if sys.argv[i] == '--Base_Model' and i+1<len(sys.argv):
            cf.Base_Model = sys.argv[i+1]

    # Run model on datasets
    for name in cf.Dataset_Names:
        cf.Dataset_Name = name
        MAIN()
