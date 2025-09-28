# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 19:44:32 2021

@author: 申聪
"""
import numpy as np

class mol_dis_sim(object):
    def Calculate_distance(Coor,arr_cutoff):
#        print(Coor)
        Num_atoms = len(Coor)
    
        all_atoms = []
        Distance_matrix_real = np.ones((Num_atoms,Num_atoms),dtype=float) 
        Distance_matrix = np.ones((Num_atoms,Num_atoms),dtype=float)
        for i in range(Num_atoms):
            all_atoms.append(Coor[i].split("\t")[0])
            for j in range(i+1,Num_atoms):
                # print("这是j的值：",j)
                # print("这是Coordinate_value[j]：",Coordinate_value[j])
                x_i = float(Coor[i].split("\t")[1])
                y_i = float(Coor[i].split("\t")[2])
                z_i = float(Coor[i].split("\t")[3])
                
                x_j = float(Coor[j].split("\t")[1])
                y_j = float(Coor[j].split("\t")[2])
                z_j = float(Coor[j].split("\t")[3])
    
                dis = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2 + (z_i - z_j) ** 2)
    
    #                eta = args.eta[a]
    #                exp_k = 4
    #                Similarity_matrix[i][j] = 1/(1+np.power(abs(dis)/eta,exp_k))
    #                Similarity_matrix[j][i] = 1/(1+np.power(abs(dis)/eta,exp_k))
#                Distance_matrix_real[i][j] = dis
#                Distance_matrix_real[j][i] = dis
                if dis <= float(arr_cutoff[0]) or dis >= float(arr_cutoff[1]):
                    Distance_matrix[i][j] = 0.0
                    Distance_matrix[j][i] = 0.0
                    Distance_matrix_real[i][j] = 0.0
                    Distance_matrix_real[j][i] = 0.0                   
                else:
                    Distance_matrix[i][j] = 1.0
                    Distance_matrix[j][i] = 1.0
                    Distance_matrix_real[i][j] = dis
                    Distance_matrix_real[j][i] = dis

        return Distance_matrix,Distance_matrix_real,all_atoms


