import random
from random import choices
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import copy
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree
from models import SSP, Split_model
from server import Server
from client import Client_GC
from utils import get_stats, split_data, get_numGraphLabels, init_structure_encoding

def _randChunk(graphs, num_client, overlap, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum/num_client))
    graphs_chunks = []
    if not overlap:
        for i in range(num_client):
            graphs_chunks.append(graphs[i*minSize:(i+1)*minSize])
        for g in graphs[num_client*minSize:]:
            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_client)
        for s in sizes:
            graphs_chunks.append(choices(graphs, k=s))
    return graphs_chunks

def prepareData_multiDS(args, datapath, group='chem', batchSize=128, seed=None):
    assert group in ['chem', "biochem", 'biochemsn', "biosncv", "chemsn", "chemsncv", "chemcv"]

    if group == 'chem':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]
    elif group == 'biochem':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1", "Peking_1", "OHSU", "PROTEINS"]
    elif group == 'biochemsn':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1", "Peking_1", "OHSU", "PROTEINS", "IMDB-MULTI", "IMDB-BINARY"]
    elif group == 'biosncv':
        datasets = ["Peking_1", "OHSU", "PROTEINS", "IMDB-MULTI", "IMDB-BINARY", "Letter-high", "Letter-med", "Letter-low"]
    elif group == 'chemsn':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1", "IMDB-MULTI", "IMDB-BINARY"]
    elif group == 'chemsncv':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1", "IMDB-MULTI", "IMDB-BINARY", "Letter-high", "Letter-med", "Letter-low"]
    elif group == 'chemcv':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1", "Letter-high", "Letter-med", "Letter-low"]


    splitedData = {}
    df = pd.DataFrame()
    for data in datasets:
        if data == "IMDB-BINARY":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
        elif data == "IMDB-MULTI":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
        elif "Letter" in data:
            tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=True)
        else:
            tudataset = TUDataset(f"{datapath}/TUDataset", data)

        graphs = [x for x in tudataset]
        print("  **", data, len(graphs))

        graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
        graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)

        graphs_train = init_structure_encoding(args, gs=graphs_train, type_init=args.type_init)
        graphs_val = init_structure_encoding(args, gs=graphs_val, type_init=args.type_init)
        graphs_test = init_structure_encoding(args, gs=graphs_test, type_init=args.type_init)

        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)

        num_node_features = graphs[0].num_node_features
        num_graph_labels = get_numGraphLabels(graphs_train)

        splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                             num_node_features, num_graph_labels, len(graphs_train))

        df = get_stats(df, data, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)

    return splitedData, df

    splitedData = {}
    df = pd.DataFrame()
    for data in datasets:
        if data == "IMDB-BINARY":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
        elif data == "IMDB-MULTI":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
        elif "Letter" in data:
            tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=True)
        else:
            tudataset = TUDataset(f"{datapath}/TUDataset", data)

        graphs = [x for x in tudataset]
        print("  **", data, len(graphs))
        num_node_features = graphs[0].num_node_features
        graphs_chunks = _randChunk(graphs, nc_per_ds, overlap=False, seed=seed)
        for idx, chunks in enumerate(graphs_chunks):
            ds = f'{idx}-{data}'
            ds_tvt = chunks
            graphs_train, graphs_valtest = split_data(ds_tvt, train=0.8, test=0.2, shuffle=True, seed=seed)
            graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)

            graphs_train = init_structure_encoding(args, gs=graphs_train, type_init=args.type_init)
            graphs_val = init_structure_encoding(args, gs=graphs_val, type_init=args.type_init)
            graphs_test = init_structure_encoding(args, gs=graphs_test, type_init=args.type_init)

            dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
            dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
            dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)
            num_graph_labels = get_numGraphLabels(graphs_train)
            splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                               num_node_features, num_graph_labels, len(graphs_train))
            df = get_stats(df, ds, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)

    return splitedData, df


def setup_devices_SSP(splitedData, args):
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
        first_batch = next(iter(dataloaders['train']))

        if first_batch.x is not None:
            node_feature_dim = first_batch.x.size(1)
            node_feature_dim = [node_feature_dim]

        edge_feature_dim = 0

        if args.alg == "fedSSP":
            former= SSP(num_graph_labels, args.nlayer, node_feature_dim, edge_feature_dim, node_feature_dim[0], args.head, args.hidden)
            head = former.fc
            former.fc = nn.Identity()
            basicModel = Split_model(former, head)
            cmodel_gc = copy.deepcopy(basicModel)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
        clients.append(Client_GC(copy.deepcopy(cmodel_gc), idx, ds, train_size, dataloaders, optimizer, args))
    smodel = copy.deepcopy(cmodel_gc)

    server = Server(smodel, args.device)
    return clients, server, idx_clients
