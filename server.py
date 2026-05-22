import torch
import numpy as np
import random
import networkx as nx
from dtaidistance import dtw
from client import clientAvgSSP
import copy

class Server():
    def __init__(self, model, device):
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.model_cache = []
        self.customized_params = {}

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_encoders = []
        self.uploaded_filters = []
        self.uploaded_feature_extractor = []

        self.clients = []
        self.selected_clients = []

        self.uploaded_model_gs = []

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.Budget = []


    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def receive_models_SSP(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters_SSP(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters_SSP(w, client_model)

    def add_parameters_SSP(self, w, client_model):
        w = 1 / len(self.selected_clients)
        for (server_name, server_param), (client_name, client_param) in zip(self.global_model.named_parameters(), client_model.named_parameters()):
            if 'encoder' in server_name and 'atom' not in server_name:
                server_param.data += client_param.data.clone() * w
    def send_models_SSP(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters_SSP(self.global_model)


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


