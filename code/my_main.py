import torch
from torch.utils.data import Dataset, DataLoader
import os
import warnings
import copy
import sys
import config as cf
import data_process
import model



def MAIN():


    # settings
    cf.XMaxLen = 0
    cf.YList = []
    cf.Use_GPU = torch.cuda.is_available()   

    # output configuration
    print('Dataset_Name={}'.format(cf.Dataset_Name))
    print('Base_Model={}'.format(cf.Base_Model))

    # multiple rounds experiments
    for i in range(cf.Round):
        if i > 10:
            cf.Pretrained = True
            
        # random seed setting    
        cf.Seed = i        
        cf.random_setting(i)
        
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

        # prepare dataloader for initialized training
        train_dataset = data_process.TrainDataset(TextDataset.train_examples)
        dev_dataset = data_process.TrainDataset(TextDataset.dev_examples)
        test_dataset = data_process.TrainDataset(TextDataset.test_examples)

        train_loader = DataLoader(train_dataset, batch_size=cf.Train_Batch_Size, shuffle=cf.DataLoader_Shuffle)
        dev_loader = DataLoader(dev_dataset, batch_size=cf.DevTest_Batch_Size, shuffle=cf.DataLoader_Shuffle)
        test_loader = DataLoader(test_dataset, batch_size=cf.DevTest_Batch_Size, shuffle=cf.DataLoader_Shuffle)
        tr = model.Train(model_x, model_s)
        
        # check whether pre-trained model exists
        if cf.Pretrained == False:                        
            f_test_bacc, f_test_bmaf1, init_factual_keyword_fairness = tr.Init_Train(train_loader, dev_loader, test_loader)                                      
        else:                    
            tr.model_x.load_state_dict(torch.load(cf.Base_Model + cf.Dataset_Name + cf.Stereotype + 'xinit.pt'))           
            f_test_bacc, f_test_bmaf1, init_factual_keyword_fairness = 0,0,0
        # mask the stereotype words    
        TextDataset.Word_Detection(train_loader, tr)
        # prepare dataloader for biased training
        train_dataset = data_process.TrainDataset(TextDataset.train_examples)
        dev_dataset = data_process.TrainDataset(TextDataset.dev_examples)
        test_dataset = data_process.TrainDataset(TextDataset.test_examples)

        train_loader = DataLoader(train_dataset, batch_size=cf.Train_Batch_Size, shuffle=cf.DataLoader_Shuffle)
        dev_loader = DataLoader(dev_dataset, batch_size=cf.DevTest_Batch_Size, shuffle=cf.DataLoader_Shuffle)
        test_loader = DataLoader(test_dataset, batch_size=cf.DevTest_Batch_Size, shuffle=cf.DataLoader_Shuffle)
        # biased training and debiased prediction
        test_acc, test_bacc, test_maf1, test_bmaf1, f_fairness, f_bfairness = tr.Train(train_loader, dev_loader, test_loader, i+100)

        f = open(cf.Base_Model + cf.Dataset_Name + cf.Stereotype + '.txt', 'a')
        f.write(cf.Base_Model)
        f.write('\n')
        f.write(cf.Fusion)
        f.write('\n')
        f.write(cf.Dataset_Name)
        f.write('\n')
        f.write(str(cf.Sigma))
        f.write('\n')
        f.write('Init Acc in {}-Rounds= {}'.format(cf.Round, f_test_bacc))
        f.write('\n')
        f.write('Init F1 in {}-Rounds= {}'.format(cf.Round, f_test_bmaf1))
        f.write('\n')
        f.write('Init Fairness in {}-Rounds= {}'.format(cf.Round, init_factual_keyword_fairness))
        f.write('\n')
        f.write('Final Acc in {}-Rounds= {}'.format(cf.Round, test_acc))
        f.write('\n')
        f.write('Final BaseAcc in {}-Rounds= {}'.format(cf.Round, test_bacc))
        f.write('\n')
        f.write('Final F1 in {}-Rounds= {}'.format(cf.Round, test_maf1))
        f.write('\n')
        f.write('Final BaseF1 in {}-Rounds= {}'.format(cf.Round, test_bmaf1))
        f.write('\n')
        f.write('Final Fair in {}-Rounds= {}'.format(cf.Round, f_fairness))
        f.write('\n')
        f.write('Final BaseFair in {}-Rounds= {}'.format(cf.Round, f_bfairness))
        f.write('\n')
        
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
