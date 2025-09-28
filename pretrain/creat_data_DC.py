import pandas as pd
import numpy as np
import random
from distance import mol_dis_sim
from feature import mol_feature
from rbf_direction import calculate_edge_fea
from utils.utils_test import TestbedDataset
from torch_geometric.data import InMemoryDataset, DataLoader

def get_element_index(mat_0_1,mat_dis):
    arr_index = []
    arr_dis = []
    for i in range(len(mat_0_1)):
        _index = []
        for j in range(len(mat_0_1[i])):
            if float(mat_0_1[i][j]) == 1.0:
                _index.append(i)
                _index.append(j)
                arr_index.append(_index)
                arr_dis.append(mat_dis[i][j])
                _index = []
    return arr_index,arr_dis

def get_edge_index(edge_index, edge_dis, arr_coor):
    edge_num = len(edge_index)
    ed_arr_index = []
    ed_arr_index_num = []
    ed_index = []
    ed_index_num = []
    
    arr_edge_representation = []
    
    for i in range(edge_num):
        rbf_rep = calculate_edge_fea.rbf_vector(edge_dis[i])
        direction_rep = calculate_edge_fea.direction_vector(arr_coor,edge_index[i])
        edge_representation = rbf_rep + direction_rep 
        
        for j in range(edge_num):
            if i == j:
                ed_index.append(edge_index[i])
                ed_index.append(edge_index[j])
                ed_index_num.append(i)
                ed_index_num.append(j)
                ed_arr_index.append(ed_index)
                ed_arr_index_num.append(ed_index_num)
                ed_index = []
                ed_index_num = []
            elif len(list(set(edge_index[i]) & set(edge_index[j]))) == 1: #不能等于2，因为至少有一个点是自环
                ed_index.append(edge_index[i])
                ed_index.append(edge_index[j])
                ed_index_num.append(i)
                ed_index_num.append(j)
                ed_arr_index.append(ed_index)
                ed_arr_index_num.append(ed_index_num)
                ed_index = []
                ed_index_num = []
            
        arr_edge_representation.append(edge_representation)         
                
    return ed_arr_index,ed_arr_index_num,arr_edge_representation

def get_tri_index(edge_index, edge_edge_index):
    node_edge_num = len(edge_index)
    edge_edge_num = len(edge_edge_index)
    print(node_edge_num)
    print(edge_edge_num)
    tri_arr_index = []
    tri_index = []
    for i in range(node_edge_num):
        for j in range(edge_edge_num):
#            print("=================================")
#            print(edge_index[i])
#            print(edge_edge_index[j][0])
#            print(edge_edge_index[j][1])
            if edge_index[i] ==  edge_edge_index[j][0] and edge_index[i] ==  edge_edge_index[j][1]:
                tri_index.append(edge_index[i])
                tri_index.append(edge_edge_index[j][0])
                tri_index.append(edge_edge_index[j][1])
                tri_arr_index.append(tri_index)
                tri_index = []
            elif edge_index[i] in edge_edge_index[j]:
                pass
            elif len(list((set(edge_index[i]) & set(edge_edge_index[j][0])) | (set(edge_index[i]) & set(edge_edge_index[j][1])) | (set(edge_edge_index[j][0]) & set(edge_edge_index[j][1])))) == 3:
                tri_index.append(edge_index[i])
                tri_index.append(edge_edge_index[j][0])
                tri_index.append(edge_edge_index[j][1])
                tri_arr_index.append(tri_index)
                tri_index = []
    print(tri_arr_index[0])
    print(len(tri_arr_index))

def get_new_coor(coor,Net_type):
    arr_Element = Net_type.split("-")
    new_coor = []
    for i in range(len(coor)):
        if coor[i].split("\t")[0] in arr_Element:
            new_coor.append(coor[i])
    return new_coor
    
def smile_to_graph3(arr_coor,List_cutoff):
    c_size = []
    features = []
    edge_indexs = []
    
    edge_size = []
    arr_edge_edge_feat = []
    arr_edge_edge_num = []
    
    for i in range(len(List_cutoff)):
#        print(i)
        arr_cutoff = List_cutoff[i].split("-")

        drug_dis, drug_dis_real, drug_atoms = mol_dis_sim.Calculate_distance(arr_coor,arr_cutoff)         
        drug_feat = mol_feature.Calculate_feature(drug_dis_real,drug_atoms)
        edge_index, edge_dis = get_element_index(drug_dis,drug_dis_real)
        
        edge_edge_index,edge_edge_num,edge_edge_feat = get_edge_index(edge_index, edge_dis, arr_coor)
        
        c_size.append(len(arr_coor))
        features.append(drug_feat)
        edge_indexs.append(edge_index)
        
        edge_size.append(len(edge_index))
        arr_edge_edge_feat.append(edge_edge_feat)
        arr_edge_edge_num.append(edge_edge_num)
    return c_size, features, edge_indexs,edge_size,arr_edge_edge_feat,arr_edge_edge_num

#def get_default_sider_task_names():
    """Get that default sider task names and return the side results for the drug"""
#    return ['value']

def creat_data(datafile):    
    fr_3d_train = open('3D_coor_'+datafile+'_train.txt','r')
    fr_3d_vali = open('3D_coor_'+datafile+'_vali.txt','r')
#    List_cutoff = ['0-3']
    List_cutoff = ['0-2','0-2.5','0-3','0-3.5','0-4']
#    List_cutoff = ['0-2','2-4','4-6','6-8','8-1000']
        
    
    print("Start the computation of the training set data..")
    arr_coor_train = []
    i = 0
    for line in fr_3d_train:
        arrLin = line.strip().split("\t")
        if len(arrLin) == 4:
            arr_coor_train.append(line.strip())
        elif len(arrLin) == 1:           
            g = smile_to_graph3(arr_coor_train,List_cutoff)
            
            mask_coor = arr_coor_train
            atom_num = len(arr_coor_train)
            givenIndices = random.sample(range(0, atom_num), int(atom_num*0.2))
            indicesList = sorted(givenIndices, reverse=True)
            for a in range(len(indicesList)):
                mask_coor.pop(indicesList[a])
                
            g_mask = smile_to_graph3(mask_coor,List_cutoff)           
            
            TestbedDataset(root='data', dataset = datafile + "_agu_train"+str(i), smile_graph = g)
            TestbedDataset(root='data', dataset = datafile + "_agu_train"+str(i)+"_mask", smile_graph = g_mask)
           
            arr_coor_train = []
            i = i + 1

            
    print("Start the computation of the validation set data..")
    arr_coor_vali = []
    i = 0    
    for line in fr_3d_vali:
        
        arrLin = line.strip().split("\t")
        if len(arrLin) == 4:
            arr_coor_vali.append(line.strip())
        elif len(arrLin) == 1:            
            g = smile_to_graph3(arr_coor_vali,List_cutoff)
            
            mask_coor = arr_coor_vali
            atom_num = len(arr_coor_vali)
            givenIndices = random.sample(range(0, atom_num), int(atom_num*0.2))
            indicesList = sorted(givenIndices, reverse=True)            
            for a in range(len(indicesList)):
                mask_coor.pop(indicesList[a])
                
            g_mask = smile_to_graph3(mask_coor,List_cutoff) 

            TestbedDataset(root='data', dataset = datafile + "_agu_vali"+str(i), smile_graph = g)
            TestbedDataset(root='data', dataset = datafile + "_agu_vali"+str(i)+"_mask", smile_graph = g_mask)

            arr_coor_vali = []  
            i = i + 1   

if __name__ == "__main__":
    da = ['pretrain_50k']
    for datafile in da:
        creat_data(datafile)
