import pandas as pd
from client import collate_pyg_to_dgl
import torch
import numpy as np
def proscess_loader(loader, device):
    preprocessed_batches = []
    for batch in loader:
        batch.to(device)
        e, u, g, length, valid_indices = collate_pyg_to_dgl(batch)
        valid_labels = batch.y[valid_indices].to(device)
        preprocessed_batches.append((e.to(device), u.to(device), g.to(device), length.to(device), valid_labels, len(valid_indices)))
    return preprocessed_batches

def run_fedSSP(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0, summary_writer=None):
    device = torch.device('cuda:0')
    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    for client in clients:
        dataloaders = client.dataLoader
        train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
        client.train_preprocessed_batches = proscess_loader(train_loader, device)
        client.test_preprocessed_batches = proscess_loader(test_loader, device)
        client.val_preprocessed_batches = proscess_loader(val_loader, device)
        server.clients = clients
        server.selected_clients = clients
        client.train_samples = len(train_loader)


    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)
            server.selected_clients = selected_clients#新增
            server.clients = clients

        for client in selected_clients:
            client.local_train(local_epoch)

        if c_round != 1:
            for i, w in enumerate(server.uploaded_weights):
                w = 1 / len(server.selected_clients)
                server.uploaded_weights[i] = w
            global_consensus = 0
            for cid, w in zip(server.uploaded_ids, server.uploaded_weights):
                global_consensus += server.clients[cid].current_mean * w
            for client in server.selected_clients:
                client.global_consensus = global_consensus.data.clone()
            server.receive_models_SSP()
            server.aggregate_parameters_SSP()
            server.send_models_SSP()

        else:
            tot_samples = 0
            for client in server.selected_clients:
                tot_samples += client.train_samples
                server.uploaded_ids.append(client.id)
                server.uploaded_weights.append(client.train_samples)
            for i, w in enumerate(server.uploaded_weights):
                w = w / tot_samples
                server.uploaded_weights[i] = w
            global_consensus = 0
            for cid, w in zip(server.uploaded_ids, server.uploaded_weights):
                w = 1 / len(server.selected_clients)
                global_consensus += server.clients[cid].current_mean * w
            for client in server.selected_clients:
                client.global_consensus = global_consensus.data.clone()
        if c_round % 1 == 0:
            accs = []
            losses = []
            for idx in range(len(clients)):
                loss, acc = clients[idx].evaluate()
                accs.append(acc)
                losses.append(loss)

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)

            summary_writer.add_scalar(f'Test/Acc/Mean_{args.alg}', mean_acc, c_round)
            summary_writer.add_scalar(f'Test/Acc/Std_{args.alg}', std_acc, c_round)
    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame





