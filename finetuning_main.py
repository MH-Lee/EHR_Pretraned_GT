import os
import pickle
import torch
import random
import itertools
import math
from time import time
from torch_geometric.loader import DataListLoader as GraphLoader
from torch_geometric.data import Batch
from src.models.transformer import BertForNDP
from src.models.graphtransformer import BertConfig
from sklearn.model_selection import ShuffleSplit
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def run_epoch(model, optim_behrt, trainload, device, **global_params):
    tr_loss = 0
    start = time.time()
    model.train()
    for step, data in enumerate(trainload):
        optim_behrt.zero_grad()

        batched_data = Batch()
        graph_batch = batched_data.from_data_list(list(itertools.chain.from_iterable(data)))
        graph_batch = graph_batch.to(device)
        nodes = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_index_readout = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        batch = graph_batch.batch
        age_ids = torch.reshape(graph_batch.age, [graph_batch.age.shape[0] // 50, 50])
        time_ids = torch.reshape(graph_batch.time, [graph_batch.time.shape[0] // 50, 50])
        delta_ids = torch.reshape(graph_batch.delta, [graph_batch.delta.shape[0] // 50, 50])
        type_ids = torch.reshape(graph_batch.adm_type, [graph_batch.adm_type.shape[0] // 50, 50])
        posi_ids = torch.reshape(graph_batch.posi_ids, [graph_batch.posi_ids.shape[0] // 50, 50])
        attMask = torch.reshape(graph_batch.mask_v, [graph_batch.mask_v.shape[0] // 50, 50])
        attMask = torch.cat((torch.ones((attMask.shape[0], 1)).to(device), attMask), dim=1)
        los = torch.reshape(graph_batch.los, [graph_batch.los.shape[0] // 50, 50])

        labels = torch.reshape(graph_batch.label, [graph_batch.label.shape[0] // 50, 50])[:, 0].float()
        masks = torch.reshape(graph_batch.mask, [graph_batch.mask.shape[0] // 50, 50])[:, 0]
        loss, logits = model(nodes, edge_index, edge_index_readout, edge_attr, batch, age_ids,\
                             time_ids,delta_ids,type_ids,posi_ids,attMask, labels, masks, los)

        if global_params['gradient_accumulation_steps'] >1:
            loss = loss/global_params['gradient_accumulation_steps']
        loss.backward()
        tr_loss += loss.item()
        if step%500 == 0:
            print(loss.item())
        optim_behrt.step()
        #sched.step()
        del loss
    cost = time.time() - start
    return tr_loss, cost


def train(model, trainload, valload, device, **train_params):
    best_val = math.inf
    for e in range(train_params["epochs"]):
        print("Epoch n" + str(e))
        train_loss, train_time_cost = run_epoch(model, trainload, device)
        val_loss, val_time_cost,pred, label, mask = eval(valload, False, device)
        train_loss = (train_loss * train_params['batch_size']) / len(trainload)
        val_loss = (val_loss * train_params['batch_size']) / len(valload)
        print('TRAIN {}\t{} secs\n'.format(train_loss, train_time_cost))
        print("Epoch n" + str(e) + '\n TRAIN {}\t{} secs\n'.format(train_loss, train_time_cost))
        print('EVAL {}\t{} secs\n'.format(val_loss, val_time_cost) + '\n\n\n')
        print('EVAL {}\t{} secs\n'.format(val_loss, val_time_cost))
        if val_loss < best_val:
            print("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = model.module if hasattr(model, 'module') else model
            save_model(model_to_save.state_dict(), './results/finetune/gt_behrt.pth')
            best_val = val_loss
    return train_loss, val_loss

@torch.no_grad()
def eval(model, _valload, device):
    tr_loss = 0
    start = time.time()
    model.eval()

    for step, data in enumerate(_valload):
        batched_data = Batch()
        graph_batch = batched_data.from_data_list(list(itertools.chain.from_iterable(data)))
        graph_batch = graph_batch.to(device)
        nodes = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_index_readout = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        batch = graph_batch.batch
        age_ids = torch.reshape(graph_batch.age, [graph_batch.age.shape[0] // 50, 50])
        time_ids = torch.reshape(graph_batch.time, [graph_batch.time.shape[0] // 50, 50])
        type_ids = torch.reshape(graph_batch.adm_type, [graph_batch.adm_type.shape[0] // 50, 50])
        posi_ids = torch.reshape(graph_batch.posi_ids, [graph_batch.posi_ids.shape[0] // 50, 50])
        attMask = torch.reshape(graph_batch.mask_v, [graph_batch.mask_v.shape[0] // 50, 50])
        attMask = torch.cat((torch.ones((attMask.shape[0], 1)).to(device), attMask), dim=1)
        los = torch.reshape(graph_batch.los, [graph_batch.los.shape[0] // 50, 50])

        labels = torch.reshape(graph_batch.label, [graph_batch.label.shape[0] // 50, 50])[:, 0].float()
        masks = torch.reshape(graph_batch.mask, [graph_batch.mask.shape[0] // 50, 50])[:, 0]
        loss, logits = model(nodes, edge_index, edge_index_readout, edge_attr, batch, age_ids, time_ids,delta_ids,type_ids,posi_ids,attMask, labels, masks, los)
        tr_loss += loss.item()
        del loss

    print("TOTAL LOSS", (tr_loss * train_params['batch_size']) / len(_valload))

    cost = time.time() - start
    return tr_loss, cost, logits, labels, masks

def save_model(_model_dict, file_name):
    torch.save(_model_dict, file_name)
    
    
if __name__ == '__main__': 