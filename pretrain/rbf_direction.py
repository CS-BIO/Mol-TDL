# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 19:41:46 2021

@author: 申聪
"""
import numpy as np
from rdkit import Chem
from numpy import random
from typing import List, Tuple, Union


class calculate_edge_fea(object):
           
    def rbf_vector(dis):
        p = 5
        num_radial = 7
        cutoff = 5
        inv_cutoff = 1 / cutoff
        norm_const = (2 * inv_cutoff)
        d_scaled = dis * inv_cutoff
        
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = -p * (p + 1) / 2
        
        env_val = (
            1
            + a * d_scaled ** p
            + b * d_scaled ** (p + 1)
            + c * d_scaled ** (p + 2)
        )
        env = np.where(d_scaled < 1, env_val, np.zeros_like(d_scaled)) #满足条件输出env_val，否则输出0
        frequencies = np.pi * np.arange(1, num_radial + 1, dtype=np.float32)
        
        return list(env * norm_const * np.sin(frequencies * d_scaled) / dis)
    
    def direction_vector(coor,edge):
        if edge[0] == edge[1]:
            edge_dir = [0.0,0.0,0.0]
        else:
            source = [float(coor[edge[0]].split("\t")[1]),float(coor[edge[0]].split("\t")[2]),float(coor[edge[0]].split("\t")[3])]
            target = [float(coor[edge[1]].split("\t")[1]),float(coor[edge[1]].split("\t")[2]),float(coor[edge[1]].split("\t")[3])]
            edge_direc = np.array(target) - np.array(source)
            dis = np.sqrt(np.sum(edge_direc**2))
            edge_dir = list(edge_direc / dis)
        
        return edge_dir

