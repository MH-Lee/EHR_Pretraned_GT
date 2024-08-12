######################################################################################################################
###################################################### Imports #######################################################
######################################################################################################################
import numpy as np
import pandas as pd
import csv
import os
import pickle as pkl
from torch_geometric.data import Data
import torch
#import torch.nn as nn
#import torch.nn.functional as F
from tqdm import tqdm
import itertools

def node_edge_gen(seq, dic_global, liste_node, liste_type):
    indices_d = list(set(seq['diagnoses']))
    indices_p = list(set(seq['procedures']))
    indices_m = list(set(seq['drugs']))

    for diag in indices_d:
        node = int(dic_global[diag]['identifiant']) + 1
        type_node = int(dic_global[diag]['type_code'])
        liste_node.append(node)
        liste_type.append(type_node)
    
    for proc in indices_p:
        node = int(dic_global[proc]['identifiant']) + 1
        type_node = int(dic_global[proc]['type_code'])
        liste_node.append(node)
        liste_type.append(type_node)
    
    for med in indices_m:
        node = int(dic_global[med]['identifiant']) + 1
        type_node = int(dic_global[med]['type_code'])
        liste_node.append(node)
        liste_type.append(type_node)
        
    return liste_node, liste_type
  
def creation_edge_index(x, liste_type, dic_type):
    # Edges (graphe complet)
    edge_attr = []
    edge_index = []
    all_edges = []

    for i in range(len(x)):
        for j in range(i+1,len(x)):
            all_edges.append((i, j))
            edge_attr.append(dic_type[(liste_type[i], liste_type[j])])
    source, target = zip(*all_edges)

    edge_index = torch.tensor([source, target], dtype=torch.int64)
    edge_attr = torch.tensor(edge_attr, dtype=torch.int64)

    return edge_index , edge_attr


if __name__ == '__main__':
    
    # creation du dictionnaire stay_id -> label
    path_labels = './data/dict/data_y_bin.pkl'
    with open(path_labels, 'rb') as fp:
        dic_label = pkl.load(fp)

    with open('./data/dict/adm_info.pkl', 'rb') as fp:
        demo_dict = pkl.load(fp)

    # import dictionnaire pickle dic_global
    with open('./data/dic_global.pkl', 'rb') as fp:
        dic_global = pkl.load(fp)
        
    with open('./data/dict/data_raw.pkl', 'rb') as fp:
        data_x = pkl.load(fp)

    # nombre de noeuds dont vst
    num_nodes = len(dic_global) + 1

    # dictionnaire type correspondance
    dic_type = {(0,0): 0,
                (0,1): 0,
                (1,0): 0, 
                (0,2): 0,
                (2,0): 0,
                (0,3): 0,
                (3,0): 0,
                (1,1): 1,
                (1,2): 2,
                (2,1): 2,
                (1,3): 3,
                (3,1): 3,
                (2,2): 4,
                (2,3): 5,
                (3,2): 5,
                (3,3): 6}
        
    liste_dataset = []
    for subj_id, vitsits_demo in tqdm(demo_dict.items()):
        # print(subj_id, vitsits_demo)
        liste_patient = []
        label = torch.tensor(dic_label[subj_id], dtype=torch.int64)
        for hadm_id, info in vitsits_demo.items():
            visite_objet = Data()
            visite_objet.subject_id = torch.tensor([subj_id], dtype=torch.int64)
            visite_objet.hadm_id = torch.tensor([hadm_id], dtype=torch.int64)
            visite_objet.age = torch.tensor([info['age']], dtype=torch.int64)
            visite_objet.time = torch.tensor([info['admittime']], dtype=torch.int64)
            visite_objet.rang = torch.tensor([info['rang']], dtype=torch.int64)
            visite_objet.type = torch.tensor([info['adm_type']], dtype=torch.int64)
            
            liste_node ,liste_type = [], []
            liste_node, liste_type = node_edge_gen(data_x[subj_id][hadm_id], dic_global,liste_node,liste_type)
            visite_objet.x = torch.cat([torch.tensor([0]), torch.tensor(liste_node, dtype=torch.int64)]).reshape(1,-1)
            liste_node = [0] + liste_node
            liste_type = [0] + liste_type
            
            edge_index, edge_attr = creation_edge_index(liste_node, liste_type, dic_type) 
            visite_objet.edge_index = edge_index
            visite_objet.edge_attr = edge_attr
            liste_patient.append(visite_objet)
            
        liste_patient.append(label)
        liste_dataset.append(liste_patient)


with open('./data/dic_type_correspondance.pkl', 'wb') as fp:
    pkl.dump(dic_type, fp)

with open('./data/data.pkl', 'wb') as fp:
    pkl.dump(liste_dataset, fp)