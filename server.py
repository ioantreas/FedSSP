import torch
import numpy as np
import random
import networkx as nx
from dtaidistance import dtw
from client import clientAvgSSP
import copy
import torch.nn.functional as F

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

    def aggregate_cos_sim_based_SSP(self, temperature=0.5, sim_source='weights'):
        """Build one client model per client.

        For each pair of clients we measure how similar their spectral-encoder
        vectors are via cosine similarity, normalize those similarities across the
        row (softmax with `temperature`), and form each client's model as the
        similarity-weighted sum of every client's shared (spectral-encoder)
        weights. Non-shared parameters are left untouched (they stay local).

        sim_source:
            'delta'   -> similarity computed on this round's update
                         (current shared weights - last distributed weights).
                         More discriminative; matches the paper's Δθ framing.
            'weights' -> similarity computed on the raw shared weights.
        """
        assert (len(self.uploaded_models) > 0)

        client_ids = self.uploaded_ids  # client id for each uploaded model
        client_bases = self.uploaded_models  # the uploaded SSP base modules
        num_clients = len(client_ids)

        # 1) Flatten each client's SHARED spectral-encoder weights into one vector.
        current_encoder_vectors = [self._flatten_shared(base) for base in client_bases]

        # 2) Decide what vector to compare: this round's update (delta) or the
        #    raw weights. Delta = current shared weights - what we last sent them.
        have_reference_for_all = all(cid in self.distributed for cid in client_ids)
        use_delta = (sim_source == 'delta') and have_reference_for_all
        if use_delta:
            similarity_vectors = [
                current_vec - self.distributed[cid]
                for cid, current_vec in zip(client_ids, current_encoder_vectors)
            ]
        else:
            similarity_vectors = current_encoder_vectors

        # 3) Pairwise cosine similarity between clients.
        stacked_vectors = torch.stack(similarity_vectors)  # [num_clients, num_params]
        unit_vectors = F.normalize(stacked_vectors, dim=1, eps=1e-12)
        similarity_matrix = unit_vectors @ unit_vectors.t()  # [num_clients, num_clients], in [-1, 1]
        self.sim_matrix = similarity_matrix.detach().clone()

        # 4) Turn similarities into aggregation weights. Softmax per row keeps
        #    weights positive, summing to 1, and handles negative similarities.
        #    aggregation_weights[i, j] = weight of client j in client i's model.
        aggregation_weights = torch.softmax(similarity_matrix / temperature, dim=1)
        self.agg_weights = aggregation_weights.detach().clone()

        # 5) Build the client model for each client.
        shared_params_per_client = [dict(self._shared_named(base)) for base in client_bases]
        shared_param_names = [name for name, _ in self._shared_named(client_bases[0])]

        self.client_models = {}
        for target_idx, target_id in enumerate(client_ids):
            # Copy this client's base so non-shared params stay its own (harmless).
            client_model = copy.deepcopy(client_bases[target_idx])
            client_model_shared_params = dict(self._shared_named(client_model))

            for name in shared_param_names:
                weighted_sum = torch.zeros_like(client_model_shared_params[name].data)
                for source_idx in range(num_clients):
                    weight = aggregation_weights[target_idx, source_idx]
                    weighted_sum += weight * shared_params_per_client[source_idx][name].data
                client_model_shared_params[name].data.copy_(weighted_sum)

            self.client_models[target_id] = client_model
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


