import torch
import torch.nn as nn
from copy import deepcopy

from models.gcn import GCNNet
from sklearn.metrics import balanced_accuracy_score,r2_score,mean_squared_error,mean_absolute_error
from sklearn import metrics
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from utils.utils_test import TestbedDataset
from utils.nt_xent import NTXentLoss
from distance import mol_dis_sim
from feature import mol_feature
from rbf_direction import calculate_edge_fea

import pandas as pd
import numpy as np
import os

def train(model, device, dataset_train, optimizer,TRAIN_BATCH_SIZE, nt_xent_criterion):
    model.train()
    train_loader = DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    total_loss = 0
    for ind, (data_,data2_,data_mask,data2_mask) in enumerate(train_loader):
        optimizer.zero_grad()
        print(data_)
        output = model(data_,data2_, device)
        output_mask = model(data_mask,data2_mask, device)
        if output.shape[0] != TRAIN_BATCH_SIZE:
            break
        loss = nt_xent_criterion(output, output_mask)
        
        loss = torch.sum(loss)
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss.unsqueeze(0).cpu())    
#    del train_loader
    return total_loss


def predicting(model, device, dataset_vali, VALID_BATCH_SIZE,nt_xent_criterion):
    model.eval()
    
    vali_loader = DataLoader(dataset_vali, batch_size=VALID_BATCH_SIZE, shuffle=False)
    
    total_loss = 0
    
    for ind, (data_,data2_,data_mask,data2_mask) in enumerate(vali_loader):

        with torch.no_grad():
            output = model(data_,data2_, device)
            output_mask = model(data_mask,data2_mask, device)
            if output.shape[0] != VALID_BATCH_SIZE:
                break
            loss = nt_xent_criterion(output, output_mask)
            loss = torch.sum(loss)
            total_loss += float(loss.unsqueeze(0).cpu())   
#    del vali_loader
    return total_loss

def _mask(data):
    col_num = torch.randperm(data.x.shape[1])
    col_num_mask = col_num[0:int(0.3*len(col_num))]
    data.x[:,col_num_mask] = torch.randn(data.x.shape[0],int(0.3*len(col_num)))
    return data

class myIterableDataset(IterableDataset):

    def __init__(self, file_path, batch):       
        self.file_path = file_path
#        self.error_sample = [44767,57728,59329,69121,77100]
    def __iter__(self):
        i = 0
        while True:
#            if i % 1 == 0:
#                print(i)
#            if i in self.error_sample:
#                i = i + 1
#                continue
#            if i < 77100:
#                i = i + 1
#                continue                
           
            processed_data_file = 'data/processed/' + self.file_path + "_" + str(i) + ".pt"
#            processed_data_file2 = 'data/processed/' + self.file_path + "2_" + str(i) + ".pt"
            if (os.path.exists(processed_data_file)):
                data_ = TestbedDataset(root='data', dataset=self.file_path + "_" + str(i))
                data_2 = TestbedDataset(root='data', dataset=self.file_path + "2_" +str(i))
                list_data = []
                list_data2 = []
                list_data_mask = []
                list_data2_mask = []
                for idx,(data,data2) in enumerate(zip(data_,data_2)):
                    data_mask = deepcopy(data)
                    data2_mask = deepcopy(data2)
                    list_data.append(data)
                    list_data2.append(data2)
                    list_data_mask.append(_mask(data_mask))
                    list_data2_mask.append(_mask(data2_mask))
                yield list_data,list_data2,list_data_mask,list_data2_mask
#                del list_data,list_data_mask
                i = i + 1
            else:
                break          

modeling = GCNNet

# CPU or GPU

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
LR = 0.001
NUM_EPOCHS = 101
NUM_NET = 5
temperature = 0.1 
use_cosine_similarity = True 

file_name = 'pretrain_50k'
file_path_train = file_name + "_agu_train"
file_path_vali = file_name + "_agu_vali"


dataset_train = myIterableDataset(file_path_train,TRAIN_BATCH_SIZE)
dataset_vali = myIterableDataset(file_path_vali,VALID_BATCH_SIZE)

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
    
for a in range(1):
    model = modeling().to(device)
    nt_xent_criterion = NTXentLoss(device, TRAIN_BATCH_SIZE, temperature, use_cosine_similarity)
    
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    best_loss = 10000.0
    best_epoch  = 0
    arr_train_loss = []
    arr_test_loss = []
    for epoch in range(NUM_EPOCHS):
        loss_train = train(model, device, dataset_train, optimizer, TRAIN_BATCH_SIZE, nt_xent_criterion)
        loss_vali = predicting(model, device, dataset_vali, VALID_BATCH_SIZE,nt_xent_criterion)
        arr_train_loss.append(loss_train)
        arr_test_loss.append(loss_vali)
        
        if loss_vali < best_loss:
            torch.save(model,"model_"+file_name+".pt")
            best_loss = loss_vali
            best_epoch = epoch
            
        if epoch % 2 == 0:
            print("=================================================")
            print("epoch:",epoch)
            print("train_loss",loss_train)
            print("vali_loss",loss_vali)
            print("---------------")
            print("best_epoch:",best_epoch)
            print("best_loss:",best_loss)
#        del loss_train,loss_vali
    np.save(file_name + 'arr_train_loss.npy', arr_train_loss)
    np.save(file_name + 'arr_test_loss.npy', arr_test_loss)
    