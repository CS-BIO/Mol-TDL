# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 19:39:28 2023

@author: cshen
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import KFold
from Augmentation import DataAugmentation
import pandas as pd
import numpy as np
import random

file_name = "pretrain_500k"

#data = pd.read_csv("data/"+file_name+".csv")
#
#random_num = random.sample(range(0, len(data)), len(data))
#
#train_num = random_num[0:int(0.9*len(data))]
#valication_num = random_num[int(0.9*len(data)):]
#
#data_train = data.loc[train_num,:]
#data_valication = data.loc[valication_num,:]
#
#data_train.to_csv("data/"+file_name+"_train.csv", index=False, encoding='utf-8')
#data_valication.to_csv("data/"+file_name+"_vali.csv", index=False, encoding='utf-8')




#data_train = pd.read_csv("data/"+file_name+"_train.csv")
#data_valication = pd.read_csv("data/"+file_name+"_vali.csv")

for i in range(len(data_train)):
    if i % 100 == 0:
        print(i)
    try:
        mol = Chem.MolFromSmiles(data_train["smiles"][train_num[i]])
        hmol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(hmol) 
    except:
        mol = Chem.MolFromSmiles(data_train["smiles"][train_num[i]].replace("(*)","").replace("*",""))
        try:
            hmol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(hmol)
        except:
            pass
            
    writer = Chem.SDWriter("data/sdf_"+file_name+"/train/"+str(i)+'.sdf')
    writer.write(hmol)

for i in range(len(data_valication)):
    if i % 100 == 0:
        print(i)
    try:
        mol = Chem.MolFromSmiles(data_valication["smiles"][valication_num[i]])
        hmol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(hmol) 
    except:
        mol = Chem.MolFromSmiles(data_valication["smiles"][valication_num[i]].replace("(*)","").replace("*",""))
        try:
            hmol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(hmol)
        except:
            pass
            
    writer = Chem.SDWriter("data/sdf_"+file_name+"/valication/"+str(i)+'.sdf')
    writer.write(hmol)
        






#def calculate_coor(lin):
#    arrLin = lin.strip().split(" ")
#    flag = 1
##    print(arrLin)
#    for j in range(len(arrLin)):
#        # print("================================")
#        # print(j)
#        # print(arrLin[j] != "")
#        # print(flag)
#        if arrLin[j] != "" and flag == 1:
#            x = arrLin[j]
#            flag = flag + 1
#            # print("x有值了：",x)
#        elif arrLin[j] != "" and flag == 2:
#            y = arrLin[j]
#            flag = flag + 1
#            # print("y有值了：",y)
#        elif arrLin[j] != "" and flag == 3:
#            z = arrLin[j]
#            flag = flag + 1
#        elif arrLin[j] != "" and flag == 4:
#            atom = arrLin[j]
#            flag = 0
#            # print("z有值了：",z)
##    print(x + "\t" + y + "\t" + z + "\t" + atom)
#    return x,y,z,atom
#
#fw_train = open("3D_coor_"+file_name+"_train.txt","w")
#fw_vali = open("3D_coor_"+file_name+"_vali.txt","w")
#
#data_train = pd.read_csv("data/"+file_name+"_train.csv")
#data_vali = pd.read_csv("data/"+file_name+"_vali.csv")
#
#for i in range(len(data_train)):
#    print(i)
#    f_sdf1 = open("data/sdf_"+file_name+"/train/"+str(i)+".sdf")
#    num_atoms1 = 0
#    for lin in f_sdf1:
#        if len(lin.strip().split(" ")) > 20 and "CHG" not in lin.strip() and "ISO" not in lin.strip() and "RAD" not in lin.strip():
#            arrlin = lin.strip().split(" ")
#            x,y,z,atom = calculate_coor(lin)
#            fw_train.writelines(atom+"\t"+ x + "\t" + y + "\t" + z)
#            fw_train.writelines("\n")
#            num_atoms1 = num_atoms1 + 1
#    fw_train.writelines(data_train["smiles"][i])
#    fw_train.writelines("\n")
#
#
#for i in range(len(data_vali)):
#    print(i)
#    f_sdf1 = open("data/sdf_"+file_name+"/valication/"+str(i)+".sdf")
#    num_atoms1 = 0
#    for lin in f_sdf1:
#        if len(lin.strip().split(" ")) > 20 and "CHG" not in lin.strip() and "ISO" not in lin.strip() and "RAD" not in lin.strip():
#            arrlin = lin.strip().split(" ")
#            x,y,z,atom = calculate_coor(lin)
#            fw_vali.writelines(atom+"\t"+ x + "\t" + y + "\t" + z)
#            fw_vali.writelines("\n")
#            num_atoms1 = num_atoms1 + 1
#    fw_vali.writelines(data_vali["smiles"][i])
#    fw_vali.writelines("\n")



























