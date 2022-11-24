from data_process import *
from model import *
import os
import warnings
import copy
#warnings.filterwarnings('ignore')

def MAIN():


    # setting initialize
    pb.Seed = random.randint(0, pb.INF)
    pb.XMaxLen = 0
    pb.YList = []
    pb.Use_GPU = torch.cuda.is_available()

    # random seed setting
    pb.random_setting(0)

    # output configuration
    print('pb.Dataset_Name={}'.format(pb.Dataset_Name))
    print('pb.Base_Model={}'.format(pb.Base_Model))

    # prepare a specific dataset
    myAllDataset = MyAllDataset(pb.Dataset_Name)
    # build model for intialized training
    if pb.Base_Model == 'TextCNN' or pb.Base_Model == 'TextRCNN':

        [pb.embedding, pb.word2id] = pb.Pickle_Read('./w2v/glove.300d.en.txt.pickle')
        pb.embedding.weight.requires_grad = False

    if pb.Base_Model == 'TextCNN':

        model_x = TextCNN()
        model_s = TextCNN()

    elif pb.Base_Model == 'TextRCNN':

        model_x = TextRCNN()
        model_s = TextRCNN()


    elif pb.Base_Model == 'RoBERTa':
        model_x = RoBERTa()
        model_s = RoBERTa()

    elif pb.Base_Model == 'GPT2':
        model_x = GPT2()
        model_s = GPT2()

    if pb.Use_GPU == True:
        model_x = model_x.cuda()
        model_s = model_s.cuda()


    maf1s = []
    bmaf1s = []
    acc = []
    bacc = []
    fair = []
    bfair = []

    tr = Train(model_x, model_s)

    train_dataset = MyDataset(myAllDataset.train_examples)
    train_loader = DataLoader(train_dataset, batch_size=pb.Train_Batch_Size, shuffle=pb.DataLoader_Shuffle)

    tr.Init_Train(train_loader)
    myAllDataset.Word_Detection(train_loader, tr)

    train_dataset = MyDataset(myAllDataset.train_examples)
    dev_dataset = MyDataset(myAllDataset.dev_examples)
    test_dataset = MyDataset(myAllDataset.test_examples)

    train_loader = DataLoader(train_dataset, batch_size=pb.Train_Batch_Size, shuffle=pb.DataLoader_Shuffle)
    dev_loader = DataLoader(dev_dataset, batch_size=pb.DevTest_Batch_Size, shuffle=pb.DataLoader_Shuffle)
    test_loader = DataLoader(test_dataset, batch_size=pb.DevTest_Batch_Size, shuffle=pb.DataLoader_Shuffle)

    for i in range(pb.Round):

        test_acc, test_bacc, test_maf1, test_bmaf1, f_fairness, f_bfairness = tr.Train(train_loader, dev_loader, test_loader, i)
        acc.append(test_acc)
        bacc.append(test_bacc)
        maf1s.append(test_maf1)
        bmaf1s.append(test_bmaf1)
        fair.append(f_fairness)
        bfair.append(f_bfairness)

        average_maf1 = np.mean(maf1s)
        average_bmaf1 = np.mean(bmaf1s)
        average_acc = np.mean(acc)
        average_bacc = np.mean(bacc)
        average_fair = np.mean(fair)
        average_bfair = np.mean(bfair)
        std_maf1 = np.std(maf1s)
        std_bmaf1 = np.std(bmaf1s)
        std_acc = np.std(acc)
        std_bacc = np.std(bacc)
        std_fair = np.std(fair)
        std_bfair = np.std(bfair)

    print(pb.Base_Model)
    print(pb.Fusion)
    print(pb.Dataset_Name)
    print(pb.Learning_Rate)
    print('************Average Acc in {}-Rounds= {} ******************\n'.format(pb.Round, average_acc))
    print('************Average BaseAcc in {}-Rounds= {} ******************'.format(pb.Round, average_bacc))
    print('************Average F1 in {}-Rounds= {} ******************\n'.format(pb.Round, average_maf1))
    print('************Average BaseF1 in {}-Rounds= {} ******************'.format(pb.Round, average_bmaf1))
    print('************Average Fair in {}-Rounds= {} ******************\n'.format(pb.Round, average_fair))
    print('************Average BaseFair in {}-Rounds= {} ******************'.format(pb.Round, average_bfair))
    print('************Std Acc in {}-Rounds= {} ******************\n'.format(pb.Round, std_acc))
    print('************Std BaseAcc in {}-Rounds= {} ******************'.format(pb.Round, std_bacc))
    print('************Std F1 in {}-Rounds= {} ******************\n'.format(pb.Round, std_maf1))
    print('************Std BaseF1 in {}-Rounds= {} ******************'.format(pb.Round, std_bmaf1))
    print('************Std Fair in {}-Rounds= {} ******************\n'.format(pb.Round, std_fair))
    print('************Std BaseFair in {}-Rounds= {} ******************'.format(pb.Round, std_bfair))

if __name__ == "__main__":

    # set path
    os.chdir("/content/drive/My Drive/PAKDD/")

    # read configuration
    print('sys.argv={}'.format(sys.argv))

    # Read Dataset and Model info
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--Dataset_Name' and i+1<len(sys.argv):
            pb.Dataset_Names = [sys.argv[i+1]]
        if sys.argv[i] == '--Base_Model' and i+1<len(sys.argv):
            pb.Base_Model = sys.argv[i+1]

    # Run model on datasets
    for name in pb.Dataset_Names:
        pb.Dataset_Name = name
        MAIN()
