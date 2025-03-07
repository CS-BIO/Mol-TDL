import torch
import torch.nn as nn
from utils.utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score

from sklearn.metrics import balanced_accuracy_score,r2_score,mean_squared_error,mean_absolute_error
from sklearn import metrics
import pandas as pd
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class polyGraph(torch.nn.Module):
    def __init__(self, n_output=1,dropout=0.1, file=None):
        super(polyGraph, self).__init__()
        self.out = nn.Linear(64, n_output).cuda()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def train(self,model, device, loader_train,loader_train2, optimizer, epoch):
        model.train()
    
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        total_loss = torch.Tensor()
        
        arr_data = []
        arr_data2 = []
        for batch_idx,(data_,data2_) in enumerate(zip(loader_train,loader_train2)):
            arr_data_tmp = []
            arr_data_tmp2 = []
            for idx, (dta_,dta2_) in enumerate(zip(data_,data2_)):
                arr_data_tmp.append(dta_)
                arr_data_tmp2.append(dta2_)
            arr_data.append(arr_data_tmp)
            arr_data2.append(arr_data_tmp2)
        
        for i in range(len(arr_data[0])):
            arr_data_data = [a[i] for a in arr_data]
            arr_data_data2 = [a[i] for a in arr_data2]
        
            optimizer.zero_grad()
            y = arr_data_data[0].y.to(device)
            output = model(arr_data_data,arr_data_data2, device).squeeze()
            #output = self.dropout(output)
            output = self.relu(output)
            output = self.out(output).squeeze()
            
            loss = loss_fn(output, y)
            loss = torch.sum(loss)
    
            loss.backward()
            optimizer.step()
            
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, y.cpu()), 0) 
            total_loss = torch.cat((total_loss, loss.unsqueeze(0).cpu()), 0) 
        return total_labels.detach().numpy().flatten(), total_preds.detach().numpy().flatten(), total_loss.detach().numpy().flatten()


    def predicting(self, model, device, loader_test,loader_test2):
        model.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        with torch.no_grad():
            arr_data = []
            arr_data2 = []
            for batch_idx, (data,data2) in enumerate(zip(loader_test,loader_test2)):
                arr_data_tmp = []
                arr_data_tmp2 = []
                for idx, (data_data,data_data2) in enumerate(zip(data,data2)):
                    arr_data_tmp.append(data_data)
                    arr_data_tmp2.append(data_data2)
                arr_data.append(arr_data_tmp)
                arr_data2.append(arr_data_tmp2)
            for i in range(len(arr_data[0])):
                arr_data_data = [a[i] for a in arr_data]
                arr_data_data2 = [a[i] for a in arr_data2]

                ys = arr_data_data[0].y
                output = model(arr_data_data,arr_data_data2, device).squeeze()
                #output = self.dropout(output)
                output = self.relu(output)
                output = self.out(output).squeeze()
                
                total_preds = torch.cat((total_preds, output.cpu()), 0)
                total_labels = torch.cat((total_labels, ys.cpu()), 0)   
        return total_labels.numpy().flatten(), total_preds.numpy().flatten()

if __name__ == "__main__":

    # CPU or GPU
    polyGraph_model = polyGraph()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    
    
    file_name = "E_eib"
    data_train = TestbedDataset(root='data', dataset= file_name +'_agu_train')
    data_train2 = TestbedDataset(root='data', dataset= file_name +'_agu_train2')
    data_valid = TestbedDataset(root='data', dataset= file_name +'_agu_validation')
    data_valid2 = TestbedDataset(root='data', dataset= file_name +'_agu_validation2')
    data_test = TestbedDataset(root='data', dataset= file_name +'_agu_test')
    data_test2 = TestbedDataset(root='data', dataset= file_name +'_agu_test2')
    
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    LR = 0.001
    NUM_EPOCHS = 4001
    NUM_NET = 5
    
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)
    
    length_train = int(len(data_train)/NUM_NET)
    length_valid = int(len(data_valid)/NUM_NET)
    length_test = int(len(data_test)/NUM_NET)
    
    
    print("length of training set:",length_train)
    print("length of validation set:",length_valid)
    print("length of testing set:",length_test)
    
    train_num = np.linspace(0, length_train-1, length_train)
    valid_num = np.linspace(0, length_valid-1, length_valid)
    test_num = np.linspace(0, length_test-1, length_test)
    
    
    loader_train = []
    loader_train2 = []
    loader_valid = []
    loader_valid2 = []
    loader_test = []
    loader_test2 = []
    for a in range(NUM_NET):
        train_num_tmp = [int(n * NUM_NET + a) for n in train_num]
        valid_num_tmp = [int(n * NUM_NET + a) for n in valid_num]
        test_num_tmp = [int(n * NUM_NET + a) for n in test_num]
        
        data_train_tmp = data_train[train_num_tmp]
        data_train_tmp2 = data_train2[train_num_tmp]
        data_valid_tmp = data_valid[valid_num_tmp]
        data_valid_tmp2 = data_valid2[valid_num_tmp]
        data_test_tmp = data_test[test_num_tmp]
        data_test_tmp2 = data_test2[test_num_tmp]
        
        loader_train.append(DataLoader(data_train_tmp, batch_size=TRAIN_BATCH_SIZE, shuffle=None))
        loader_train2.append(DataLoader(data_train_tmp2, batch_size=TRAIN_BATCH_SIZE, shuffle=None))
        loader_valid.append(DataLoader(data_valid_tmp, batch_size=VALID_BATCH_SIZE, shuffle=None))
        loader_valid2.append(DataLoader(data_valid_tmp2, batch_size=VALID_BATCH_SIZE, shuffle=None))
        loader_test.append(DataLoader(data_test_tmp, batch_size=TEST_BATCH_SIZE, shuffle=None))
        loader_test2.append(DataLoader(data_test_tmp2, batch_size=TEST_BATCH_SIZE, shuffle=None))
        
    for a in range(5):
        model = torch.load('model_pretrain_100k.pt',map_location='cuda:0').to(device)
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        best_rmse = 1000
        best_R2 = 0
        for epoch in range(NUM_EPOCHS):
            T_train, S_train, loss_train = polyGraph_model.train(model, device, loader_train,loader_train2, optimizer, epoch + 1)
            T, S = polyGraph_model.predicting(model, device, loader_test,loader_test2)
            rmse = np.sqrt(mean_squared_error(T, S))
            R2 = r2_score(T, S)
            if R2 > best_R2:
                best_rmse = rmse
                best_R2 = R2
            if epoch % 400 == 0:
                print("=================================================")
                print("epoch:",epoch)
                print("train_loss",np.sum(loss_train))
                print("train_rmse",np.sqrt(mean_squared_error(T_train, S_train)))
                print("train_R2",r2_score(T_train, S_train))
                print("test_rmse",rmse)
                print("test_R2",R2)
                print("--------------")
                print("best_rmse",best_rmse)
                print("best_R2",best_R2)
