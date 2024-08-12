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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import itertools

######################################################################################################################
##################################################### Fonctions ######################################################
######################################################################################################################
def create_graph_pad(patient):
    visite = Data(
        x = torch.tensor([0,1], dtype=torch.int64),
        edge_index = torch.tensor([[0],[1]], dtype=torch.int64),
        edge_attr = torch.tensor([7], dtype=torch.int64),  # on met des valeurs qui n'existent pas pour les visites de padding
        # label = patient.label,
        age = torch.tensor([130], dtype=torch.int64),
        time = torch.tensor([367], dtype=torch.int64),
        rang = torch.tensor([51], dtype=torch.int64),
        type = torch.tensor([10], dtype=torch.int64),
        subject_id = patient.subject_id,
        hadm_id = torch.tensor([1], dtype=torch.int64),
        mask_v = torch.tensor([0], dtype=torch.int64),
        )
    return visite
######################################################################################################################
##################################################### Main ###########################################################
######################################################################################################################

if __name__ == '__main__':
    print(os.getcwd())
    with open('./data/data_v3.pkl', 'rb') as f:
        dataset = pkl.load(f)
    print('dataset loaded')

    # nombre minimal de visite que l'on souhaite prendre en compte
    num_min_visites = 2
    #nombre maximal de visite que l'on souhaite prendre en compte
    num_padding = 50


    dataset_pad = []
    
    for i, patient in tqdm(enumerate(dataset)):
        patient_pad = []
        # import pdb; pdb.set_trace()
        patient_x = patient[:-1]
        patient_y = patient[-1]
        patient_x.sort(key=lambda x: x.rang)
        nb_visites = len(patient_x)

        if nb_visites >= num_min_visites:
            if nb_visites < num_padding:
                nb_padding = num_padding - nb_visites
                patient_liste = []
                patient_pad = [create_graph_pad(patient_x[0]) for _ in range(nb_padding)]
                rang = 0
                for visite in patient_x:
                    visite.x = visite.x.squeeze()
                    # decaler de +5 tous les chiffres sauf 0
                    visite.x = torch.where(visite.x != 0, visite.x + 5, visite.x)
                    visite.mask_v = torch.tensor([1], dtype=torch.int64)
                    # visite.time = visite.date
                    # del visite.date
                    visite.rang = torch.tensor([rang], dtype=torch.int64)
                    rang += 1
                    patient_liste.append(visite)
                # patient_liste.append(patient_y)
                patient_pad_liste = patient_pad + patient_liste

            elif nb_visites > num_padding:
                patient_pad = patient_x[-num_padding:]
                patient_pad_liste = []

                rang = 0
                for visite in patient_pad:
                    visite.x = visite.x.squeeze()
                    visite.x = torch.where(visite.x != 0, visite.x + 5, visite.x)
                    visite.mask_v = torch.tensor([1], dtype=torch.int64)
                    # visite.time = visite.date
                    # del visite.date
                    visite.rang = torch.tensor([rang], dtype=torch.int64)
                    rang += 1
                    patient_pad_liste.append(visite)
                # patient_pad_liste.append(patient_y)
                
            elif nb_visites == num_padding:
                patient_pad = patient_x
                patient_pad_liste = []
                rang = 0
                for visite in patient_pad:
                    visite.x = visite.x.squeeze()
                    visite.x = torch.where(visite.x != 0, visite.x + 5, visite.x)
                    visite.mask_v = torch.tensor([1], dtype=torch.int64)
                    # visite.time = visite.date
                    # del visite.date
                    visite.rang = torch.tensor([rang], dtype=torch.int64)
                    rang += 1
                    patient_pad_liste.append(visite)
                # patient_pad_liste.append(patient_y)

            dataset_pad.append((patient_pad_liste, patient_y))
            # if i == 1000:
            #     break

print('dataset_pad created')

with open('./data/pretrained_data_pad_v3.pkl', 'wb') as f:
    pkl.dump(dataset_pad, f)