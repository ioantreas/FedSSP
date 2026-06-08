import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import numpy as np

def run_selftrain_GC(args, clients, server, local_epoch):
    # all clients are initialized with the same weights
    for client in clients:
        client.download_from_server(args, server)

    allAccs = {}
    for client in clients:
        client.local_train(local_epoch)

        loss, acc = client.evaluate()
        allAccs[client.name] = [client.train_stats['trainingAccs'][-1], client.train_stats['valAccs'][-1], acc]
        print("  > {} done.".format(client.name))

    return allAccs


def run_fedavg(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0, summary_writer=None):
    for client in clients:
        client.download_from_server(args, server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    start_time = time.time()
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        round_start = time.time()
        if (c_round) % 1 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            # only get weights of graphconv layers
            client.local_train(local_epoch)

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(args, server)

        round_elapsed = time.time() - round_start
        total_elapsed = time.time() - start_time
        # write to log files
        if c_round % 1 == 0:

            accs = []
            losses = []

            for idx in range(len(clients)):
                loss, acc = clients[idx].evaluate()

                accs.append(acc)
                losses.append(loss)

                summary_writer.add_scalar(
                    'Test/Acc/user' + str(idx + 1),
                    acc,
                    c_round
                )

                summary_writer.add_scalar(
                    'Test/Loss/user' + str(idx + 1),
                    loss,
                    c_round
                )

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)

            csv_path = os.path.join(
                args.outbase,
                "raw",
                args.data_group,
                f"{args.repeat}_metrics_{args.alg}.csv"
            )

            if not os.path.exists(csv_path):
                with open(csv_path, "w") as f:
                    f.write(
                        "round,mean_acc,std_acc,round_time,total_time\n"
                    )

            with open(csv_path, "a") as f:
                f.write(
                    f"{c_round},"
                    f"{mean_acc},"
                    f"{std_acc},"
                    f"{round_elapsed},"
                    f"{total_elapsed}\n"
                )

            summary_writer.add_scalar(
                f'Test/Acc/Mean_{args.alg}',
                mean_acc,
                c_round
            )

            summary_writer.add_scalar(
                f'Test/Acc/Std_{args.alg}',
                std_acc,
                c_round
            )

            summary_writer.add_scalar(
                'Time/Round_Seconds',
                round_elapsed,
                c_round
            )

            summary_writer.add_scalar(
                'Time/Total_Seconds',
                total_elapsed,
                c_round
            )

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


def run_fedstar(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0, summary_writer=None):
    for client in clients:
        client.download_from_server(args, server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    start_time = time.time()
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        round_start = time.time()
        if (c_round) % 1 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            # only get weights of graphconv layers
            client.local_train(local_epoch)

        server.aggregate_weights_se(selected_clients)
        for client in selected_clients:
            client.download_from_server(args, server)

        round_elapsed = time.time() - round_start
        total_elapsed = time.time() - start_time
        # write to log files
        if c_round % 1 == 0:

            accs = []
            losses = []

            for idx in range(len(clients)):
                loss, acc = clients[idx].evaluate()

                accs.append(acc)
                losses.append(loss)

                summary_writer.add_scalar(
                    'Test/Acc/user' + str(idx + 1),
                    acc,
                    c_round
                )

                summary_writer.add_scalar(
                    'Test/Loss/user' + str(idx + 1),
                    loss,
                    c_round
                )

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)

            csv_path = os.path.join(
                args.outbase,
                "raw",
                args.data_group,
                f"{args.repeat}_metrics_{args.alg}.csv"
            )

            if not os.path.exists(csv_path):
                with open(csv_path, "w") as f:
                    f.write(
                        "round,mean_acc,std_acc,round_time,total_time\n"
                    )

            with open(csv_path, "a") as f:
                f.write(
                    f"{c_round},"
                    f"{mean_acc},"
                    f"{std_acc},"
                    f"{round_elapsed},"
                    f"{total_elapsed}\n"
                )

            summary_writer.add_scalar(
                f'Test/Acc/Mean_{args.alg}',
                mean_acc,
                c_round
            )

            summary_writer.add_scalar(
                f'Test/Acc/Std_{args.alg}',
                std_acc,
                c_round
            )

            summary_writer.add_scalar(
                'Time/Round_Seconds',
                round_elapsed,
                c_round
            )

            summary_writer.add_scalar(
                'Time/Total_Seconds',
                total_elapsed,
                c_round
            )

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


