import os
import pickle
import torch
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt
import random
import pandas as pd
from tqdm import tqdm
import sklearn.metrics as skm
import itertools
from torch_geometric.loader import DataListLoader as GraphLoader
from torch_geometric.data import Batch
from src.models.graphtransformer import BertConfig, PreTrainModel
from src.utils.datasets import GDSet, split_dataset

def train(model, optim, trainload, device, epoch):
    tr_loss = 0
    start = time.time()
    model.train()
    
    CE_loss = torch.nn.CrossEntropyLoss(ignore_index=3)
    
    for step, data, _ in tqdm(enumerate(trainload)):
        optim.zero_grad()

        batched_data = Batch()
        graph_batch = batched_data.from_data_list(list(itertools.chain.from_iterable(data)))
        graph_batch = graph_batch.to(device)
        nodes = graph_batch.x
                
        list_index = [i for i in range(nodes.shape[0])]
        random.shuffle(list_index)
        index_nodes_to_mask = list_index[:int((nodes.shape[0]) * pourcentage_nodes_to_mask)]
        index_nodes_not_masked = list(set(list_index) - set(index_nodes_to_mask))
        labels_nodes = nodes
        ytrue = nodes

        labels_nodes[index_nodes_not_masked] = 3
        nodes[index_nodes_to_mask] = mask_node_embeddings
        
        edge_index = graph_batch.edge_index
        edge_index_readout = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        batch = graph_batch.batch

        pred = model(nodes, edge_index, edge_index_readout, edge_attr, batch)
        loss = CE_loss(pred, labels_nodes)        
        loss.backward()
        tr_loss += loss.item()
        optim.step()

        del loss

    print(f"{epoch} TOTAL TRAIN LOSS", (tr_loss * train_params['batch_size']) / len(trainload))
    cost = time.time() - start
    return tr_loss, cost, pred, ytrue

def eval(model, _valload, device, epoch):
    val_loss = 0
    start = time.time()
    model.eval()
    CE_loss = torch.nn.CrossEntropyLoss(ignore_index=3)
    
    with torch.no_grad():
        for step, data, _ in tqdm(enumerate(_valload)):
            batched_data = Batch()
            graph_batch = batched_data.from_data_list(list(itertools.chain.from_iterable(data)))
            graph_batch = graph_batch.to(device)
            nodes = graph_batch.x
                    
            list_index = [i for i in range(nodes.shape[0])]
            random.shuffle(list_index)
            index_nodes_to_mask = list_index[:int((nodes.shape[0]) * pourcentage_nodes_to_mask)]
            index_nodes_not_masked = list(set(list_index) - set(index_nodes_to_mask))
            labels_nodes = nodes
            ytrue = nodes
            
            labels_nodes[index_nodes_not_masked] = 3
            nodes[index_nodes_to_mask] = mask_node_embeddings
            
            edge_index = graph_batch.edge_index
            edge_index_readout = graph_batch.edge_index
            edge_attr = graph_batch.edge_attr
            batch = graph_batch.batch

            pred = model(nodes, edge_index, edge_index_readout, edge_attr, batch)

            loss = CE_loss(pred, labels_nodes)            
            val_loss += loss.item()
            
            del loss

    print(f"{epoch} TOTAL EVAL LOSS", (val_loss * train_params['batch_size']) / len(_valload))    
    cost = time.time() - start
    return val_loss, cost, pred, ytrue

def run_epoch(model, optim_model, trainload, valload, device, exp, path_results):
    best_val = math.inf
    loss_train_liste = []
    loss_val_liste = []
    
    for e in tqdm(range(train_params["epochs"])):
        print("Epoch n" + str(e))
        train_loss, train_time_cost, pred_train, ytrue_train = train(model, optim_model, trainload, device, e)
        val_loss, val_time_cost, pred_eval, ytrue_eval = eval(model, valload, device, e)
        accuracy_train = skm.accuracy_score(ytrue_train.cpu().detach().numpy(), pred_train.cpu().detach().numpy().argmax(axis=1))
        accuracy_eval = skm.accuracy_score(ytrue_eval.cpu().detach().numpy(), pred_eval.cpu().detach().numpy().argmax(axis=1))

        train_loss = (train_loss * train_params['batch_size']) / len(trainload)
        val_loss = (val_loss * train_params['batch_size']) / len(valload)
        loss_train_liste.append(train_loss)
        loss_val_liste.append(val_loss)
        print('TRAIN \t{} secs'.format(train_time_cost))
        print(f'TRAIN accuracy : {accuracy_train}')
        
        with open(path_results + 'losses_and_times/' + "pre_training_log_train_" + f'{exp}' + ".txt", 'a') as f:
            f.write("Epoch n" + str(e) + '\n TRAIN {}\t{} secs\n'.format(train_loss, train_time_cost))
            f.write('EVAL {}\t{} secs\n'.format(val_loss, val_time_cost) + '\n\n\n')
        print('EVAL \t{} secs'.format(val_time_cost))
        print('EVAL accuracy : {}\n\n'.format(accuracy_eval))

        if val_loss < best_val:
            print("** ** * Saving fine - tuned model ** ** * ")
            torch.save(model.gnn.state_dict(), path_results + 'weights/' + 'GraphTransformer_pretrain_num_' + f'{exp}' + '.pch')
            best_val = val_loss
            
    epoch = [i for i in range(train_params["epochs"])]
    plt.plot(epoch, loss_train_liste)
    plt.legend(['train'])
    plt.plot(epoch,loss_val_liste)
    plt.legend(['val'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(path_results + 'plots/' + 'Pre_training.png')
    plt.show()
    return train_loss, val_loss, accuracy_train, accuracy_eval, train_time_cost, val_time_cost

def experiment(model_config, num_experiments=5, path_results='./results/'):
    conf = BertConfig(model_config)
    df = pd.DataFrame(columns=['Experiment', 'Model', 'Metric', 'Score'])

    for exp in tqdm(range(num_experiments)):
        model = PreTrainModel(conf).to(train_params['device'])
        transformer_vars = [i for i in model.parameters()]
        optim_model = torch.optim.AdamW(transformer_vars, lr=train_params['lr'], weight_decay=train_params['weight_decay'])

        print(f"\n Experiment {exp + 1}")
        trainDSet, valDSet, testDSet = split_dataset(dataset, train_params, random_seed=exp)
        trainload = GraphLoader(GDSet(trainDSet), batch_size=train_params['batch_size'], shuffle=False)
        valload = GraphLoader(GDSet(valDSet), batch_size=train_params['batch_size'], shuffle=False)

        train_loss, val_loss, accuracy_train, accuracy_eval, train_time_cost, val_time_cost = run_epoch(
            model, optim_model, trainload, valload, train_params['device'], exp, path_results
        )

        df.loc[len(df)] = [exp + 1, 'GT_BERT', 'Val Accuracy', accuracy_eval]
        df.loc[len(df)] = [exp + 1, 'GT_BERT', 'Train Loss', train_loss]
        df.loc[len(df)] = [exp + 1, 'GT_BERT', 'Val Loss', val_loss]
        df.loc[len(df)] = [exp + 1, 'GT_BERT', 'Train Time', train_time_cost]
        df.loc[len(df)] = [exp + 1, 'GT_BERT', 'Val Time', val_time_cost]

    df.to_csv(path_results + 'dataframes/' + 'GT_behrt_results_pretraining_1.csv')
    return df


if __name__ == '__main__': 
    path = './data/'
    path_results = './results/'
    with open(os.path.join(path, 'pretrained_data_pad.pkl'), 'rb') as handle:
        dataset = pickle.load(handle)
    
    pourcentage_nodes_to_mask = 0.15
    labels_masked_nodes = []
    mask_node_embeddings = 2
    
    train_params = {
        'batch_size': 5,
        'max_len_seq': 50,
        'device': "cuda" if torch.cuda.is_available() else "cpu", 
        'data_len' : len(dataset),
        'val_split' : 0.1,
        'test_split' : 0.2,
        'train_data_len' : int(0.9*0.8*len(dataset)),   # the train dataset is 90% of 80% of the whole dataset
        'val_data_len' : int(0.1*0.8*len(dataset)),   # the validation dataset is 10% of 80% of the whole dataset
        'test_data_len' : int(0.2*len(dataset)),   # the test dataset is 20% of the whole dataset
        'epochs' : 15,
        'lr': 0.0001,
        'weight_decay': 0.0001
    }

    model_config = {
        'vocab_size': 15322, # number of disease + symbols for word embedding (avec vst) + 1 for mask
        'edge_relationship_size': 8, # number of vocab for edge_attr
        'hidden_size': 50*5, # word embedding and seg embedding hidden size
        'seg_vocab_size': 2, # number of vocab for seg embedding
        'age_vocab_size': 151, # number of vocab for age embedding
        'time_vocab_size': 380, # number of vocab for time embedding
        'type_vocab_size': 11+1, # number of vocab for type embedding + 1 for mask
        # 'node_attr_size': 8, # number of vocab for node_attr embedding
        'max_position_embedding': 50, # maximum number of tokens
        'hidden_dropout_prob': 0.2, # dropout rate
        'graph_dropout_prob': 0.2, # dropout rate
        'num_hidden_layers': 6, # number of multi-head attention layers required
        'num_attention_heads': 2, # number of attention heads
        'attention_probs_dropout_prob': 0.2, # multi-head attention dropout rate
        'intermediate_size': 512, # the size of the "intermediate" layer in the transformer encoder
        'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
        'initializer_range': 0.02, # parameter weight initializer range
        'n_layers' : 3 - 1,
        'alpha' : 0.1
    } 
    
    df = experiment(model_config=model_config, num_experiments=5, path_results=path_results)
    # Group by Model and Metric and calculate average and standard deviation
    result_df = df.groupby(['Model', 'Metric']).agg({'Score': ['mean', 'std']}).reset_index()

    # Rename columns for clarity
    result_df.columns = ['Model', 'Metric', 'Average Score', 'Standard Deviation']

    result_df['Average Score'] = result_df['Average Score'].round(2)
    result_df['Standard Deviation'] = result_df['Standard Deviation'].round(2)

    # save the result
    result_df.to_csv(path_results + 'dataframes/' + 'GT_behrt_results_pretraining_global.csv')