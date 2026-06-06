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

        # --- cosine similarity aggregation state ---
        # last round's models that were shared by clients, keyed by client id
        self.distributed = {}
        # per-client models
        self.client_models = {}

        self.sim_matrix = None  # [N, N] cosine similarity between updates
        self.agg_weights = None  # [N, N] row-normalized aggregation weights


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

    # ------------------------------------------------------------------ #
    # Cosine-similarity based aggregation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _shared_named(model):
        """Named parameters that are actually shared/aggregated (the spectral
        encoders): name contains 'encoder' but not 'atom'. Returns a list so the
        ordering is deterministic and identical across clients."""
        return [(n, p) for n, p in model.named_parameters()
                if 'encoder' in n and 'atom' not in n]

    def _flatten_shared(self, model):
        """Flatten the shared parameters of a model into a single 1D vector."""
        return torch.cat([p.data.flatten() for _, p in self._shared_named(model)]).detach().clone()

    def snapshot_reference(self, clients):
        """Record the initial shared weights of every client. This becomes the
        reference point for the first 'update' delta. Call once before training
        starts (while all clients still hold the common initialization)."""
        self.distributed = {}
        for client in clients:
            self.distributed[client.id] = self._flatten_shared(client.model.base)

    def aggregate_cos_sim_based_SSP(self, temperature=0.5, sim_source='delta'):
        """Build one personalized model per client.

        For each pair (i, j) we measure how similar client i's update is to
        client j's update via cosine similarity, normalize those similarities
        across j (softmax with `temperature`), and form client i's model as the
        similarity-weighted sum of all clients' shared (spectral-encoder)
        weights. Non-shared parameters are left untouched (they stay local).

        sim_source:
            'delta'   -> similarity computed on this round's update
                         (current shared weights - last distributed weights).
                         More discriminative; matches the paper's Δθ framing.
            'weights' -> similarity computed on the raw shared weights.
        """
        assert (len(self.uploaded_models) > 0)

        ids = self.uploaded_ids
        models = self.uploaded_models
        N = len(ids)

        # 1) current shared-weight vector for each client (aligned with ids/models)
        cur_vecs = [self._flatten_shared(m) for m in models]

        # 2) form the vectors we compute similarity on
        use_delta = (sim_source == 'delta') and all(cid in self.distributed for cid in ids)
        if use_delta:
            updates = [cur - self.distributed[cid] for cid, cur in zip(ids, cur_vecs)]
        else:
            updates = cur_vecs

        M = torch.stack(updates)  # [N, P]

        # 3) pairwise cosine similarity (eps guards against zero-norm updates)
        Mn = F.normalize(M, dim=1, eps=1e-12)  # row-wise L2 normalize
        S = Mn @ Mn.t()  # [N, N], entries in [-1, 1]
        self.sim_matrix = S.detach().clone()

        # 4) normalize similarities into per-client aggregation weights
        #    softmax keeps everything positive, summing to 1, and handles
        #    negative cosine similarities gracefully.
        Wn = torch.softmax(S / temperature, dim=1)  # [N, N], rows sum to 1
        self.agg_weights = Wn.detach().clone()

        # 5) build a personalized model for every client
        shared_dicts = [dict(self._shared_named(m)) for m in models]
        names = [n for n, _ in self._shared_named(models[0])]

        self.client_models = {}
        for i, cid in enumerate(ids):
            pmodel = copy.deepcopy(models[i])  # non-shared params kept (harmless)
            pshared = dict(self._shared_named(pmodel))
            for name in names:
                acc = torch.zeros_like(pshared[name].data)
                for j in range(N):
                    acc += Wn[i, j] * shared_dicts[j][name].data
                pshared[name].data.copy_(acc)
            self.client_models[cid] = pmodel

    def send_cos_sim_aggreagted_SSP(self):
        """Send each client its own personalized model and update the reference
        used for next round's update delta."""
        assert (len(self.clients) > 0)
        for client in self.clients:
            pmodel = self.client_models[client.id]
            client.set_parameters_SSP(pmodel)
            self.distributed[client.id] = self._flatten_shared(pmodel)


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


