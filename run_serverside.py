"""Sec. IV-B: Comparison of FL aggregation methods (no client errors).

Compares FedAvg, FedAdp, FedLWS, FedHyper-G, FedLAW, and FedHAW (proposed)
on MNIST, Stanford Dogs, or CIFAR-10 under non-IID data partitioning.

Data directory convention:
    ./data/{dataset}/alpha01/   (for Dirichlet alpha=0.1)
    ./data/{dataset}/alpha1/    (for Dirichlet alpha=1.0)

Override with --data_dir if needed.

Usage examples:
    # MNIST, alpha=0.1
    python run_serverside.py --dataset mnist --dir_alpha 0.1

    # Stanford Dogs with server-side computation time measurement
    python run_serverside.py --dataset dogs --dir_alpha 0.1 --measure_time

    # CIFAR-10
    python run_serverside.py --dataset cifar10 --dir_alpha 0.1 \\
        --T 50 --E 10 --mu 0.1 --opt SGDdecay --opt_proxy Adam
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector

from functions.training import (
    train_fedavg, train_fedhaw, train_fedlaw,
    train_fedadp, train_fedhyp, train_fedlws, softmax_vec
)
from functions.networks import (
    Net, Net_dogs, Resnet18Cifar,
    ProxyNet_MNIST, ProxyNet_dogs, ProxyNet_CIFAR10_Resnet18
)
from functions.utils import (
    test, softmax_vec_adp,
    figure_gamma, figure_weights, figure_thetast, figure_gamma_layer
)
from functions.datasets import set_loaders, set_local_data


def alpha_str(alpha):
    """Convert a float alpha to the directory name suffix (e.g. 0.1->'alpha01', 1.0->'alpha1')."""
    if alpha == int(alpha):
        return 'alpha' + str(int(alpha))
    else:
        return 'alpha' + str(alpha).replace('0.', '').replace('.', '')


def get_args():
    parser = argparse.ArgumentParser(description='FedHAW - Sec. IV-B: server-side comparison')
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'dogs', 'cifar10'],
                        help='Dataset to use')
    parser.add_argument('--dir_alpha', type=float, default=0.1,
                        help='Dirichlet alpha used when partitioning local data (determines data directory)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override for local training data directory. '
                             'Defaults to ./data/{dataset}/{alpha_str}/')
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='Directory to save results and figures')

    # FL hyperparameters
    parser.add_argument('--K', type=int, default=10, help='Number of clients')
    parser.add_argument('--T', type=int, default=200, help='Number of FL rounds')
    parser.add_argument('--E', type=int, default=1, help='Number of local epochs per round')
    parser.add_argument('--batch_size', type=int, default=64, help='Local mini-batch size')
    parser.add_argument('--mu', type=float, default=0.001, help='Local learning rate')
    parser.add_argument('--opt', type=str, default='SGD',
                        choices=['SGD', 'SGDmomentum', 'SGDdecay', 'Adam'],
                        help='Local optimizer')

    # FedHAW hyperparameters
    parser.add_argument('--lr_gamma', type=float, default=0.001,
                        help='Hypergradient step size for gamma (FedHAW)')
    parser.add_argument('--lr_lambda', type=float, default=0.01,
                        help='Hypergradient step size for lambda (FedHAW)')

    # FedLAW hyperparameters
    parser.add_argument('--Eproxy', type=int, default=100,
                        help='Number of proxy optimization epochs (FedLAW)')
    parser.add_argument('--muproxy', type=float, default=0.01,
                        help='Proxy optimizer learning rate (FedLAW)')
    parser.add_argument('--opt_proxy', type=str, default='SGD',
                        choices=['SGD', 'SGDmomentum', 'Adam'],
                        help='Proxy optimizer (FedLAW)')

    # FedAdp hyperparameter
    parser.add_argument('--alpha_adp', type=float, default=5.0,
                        help='Temperature parameter alpha (FedAdp)')

    # FedHyper-G hyperparameter
    parser.add_argument('--init_thetast', type=float, default=1.0,
                        help='Initial step-size multiplier theta* (FedHyper-G)')

    # FedLWS hyperparameter
    parser.add_argument('--lws_beta', type=float, default=0.1,
                        help='Heterogeneity sensitivity parameter beta (FedLWS)')

    # Model hyperparameters
    parser.add_argument('--num_feature', type=int, default=128,
                        help='Hidden layer width (MNIST/Dogs MLP)')
    parser.add_argument('--dogs_inputsize', type=int, default=768,
                        help='Feature dimension of Dogs dataset')
    parser.add_argument('--dogs_outputsize', type=int, default=120,
                        help='Number of classes in Dogs dataset')

    # Timing experiment
    parser.add_argument('--measure_time', action='store_true',
                        help='Measure and report server-side aggregation time per round')

    return parser.parse_args()


def make_model(args):
    if args.dataset == 'mnist':
        return Net(num_feature=args.num_feature)
    elif args.dataset == 'dogs':
        return Net_dogs(inputsize=args.dogs_inputsize,
                        num_feature=args.num_feature,
                        outputsize=args.dogs_outputsize)
    elif args.dataset == 'cifar10':
        return Resnet18Cifar(num_classes=10)


def make_proxy_model(args, init_gammat, init_lambdat):
    if args.dataset == 'mnist':
        return ProxyNet_MNIST(init_gammat=init_gammat, init_lambdat=init_lambdat)
    elif args.dataset == 'dogs':
        return ProxyNet_dogs(init_gammat=init_gammat, init_lambdat=init_lambdat)
    elif args.dataset == 'cifar10':
        return ProxyNet_CIFAR10_Resnet18(init_gammat=init_gammat, init_lambdat=init_lambdat)


def main():
    args = get_args()

    # Resolve data directories
    astr = alpha_str(args.dir_alpha)
    data_dir = args.data_dir or os.path.join('./data', args.dataset, astr)
    # Dogs test data (.pt files) are stored directly under ./data/dogs/
    dogs_test_dir = os.path.join('./data', 'dogs') if args.dataset == 'dogs' else None

    os.makedirs(args.result_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    print(f'Data directory: {data_dir}', flush=True)

    criterion = nn.NLLLoss() if args.dataset in ['mnist', 'dogs'] else nn.CrossEntropyLoss()

    test_loader, proxy_loader = set_loaders(args.dataset, args.batch_size,
                                             dogs_data_dir=dogs_test_dir)
    print('Test/proxy data loaded.', flush=True)

    train_datasets = set_local_data(args.K, data_dir)
    Nk = [len(train_datasets[k]) for k in range(args.K)]
    N = sum(Nk)
    init_lambdat = [np.log(Nk[k] / N) for k in range(args.K)]
    init_gammat = np.log(1.0)
    print('Local data loaded.', flush=True)

    global_model = make_model(args).to(device)
    init_params = parameters_to_vector(global_model.parameters()).detach().clone()

    results = {}

    # ------------------------------------------------------------------
    # FedAvg
    # ------------------------------------------------------------------
    print('\n=== FedAvg ===', flush=True)
    loss_list, acc_list, time_list = [], [], []
    params = init_params.clone()
    for t in range(args.T):
        out = train_fedavg(args.dataset, train_datasets, params, args.batch_size,
                           args.E, args.mu, device, args.K, make_model(args),
                           criterion, args.opt, measure_time=args.measure_time)
        params, loss = out[0], out[1]
        if args.measure_time:
            time_list.append(out[2])
        acc = test(params, test_loader, device, make_model(args), args.dataset)
        loss_list.append(loss)
        acc_list.append(acc)
        print(f'Round {t}: loss={loss:.4f}, acc={acc:.4f}', flush=True)
    results['FedAvg'] = {'loss': loss_list, 'acc': acc_list}
    if args.measure_time:
        print(f'FedAvg avg agg time: {np.mean(time_list):.6f} s', flush=True)

    # ------------------------------------------------------------------
    # FedAdp
    # ------------------------------------------------------------------
    print('\n=== FedAdp ===', flush=True)
    loss_list, acc_list, time_list = [], [], []
    params = init_params.clone()
    lambdat = None
    for t in range(args.T):
        out = train_fedadp(args.dataset, train_datasets, params, args.batch_size,
                           args.E, args.mu, device, args.K, make_model(args),
                           criterion, args.opt, loop=t, alpha=args.alpha_adp,
                           lambdat_pre=lambdat, measure_time=args.measure_time)
        params, loss, lambdat = out[0], out[1], out[2]
        if args.measure_time:
            time_list.append(out[3])
        acc = test(params, test_loader, device, make_model(args), args.dataset)
        loss_list.append(loss)
        acc_list.append(acc)
        print(f'Round {t}: loss={loss:.4f}, acc={acc:.4f}', flush=True)
    results['FedAdp'] = {'loss': loss_list, 'acc': acc_list}
    if args.measure_time:
        print(f'FedAdp avg agg time: {np.mean(time_list):.6f} s', flush=True)

    # ------------------------------------------------------------------
    # FedLWS
    # ------------------------------------------------------------------
    print('\n=== FedLWS ===', flush=True)
    loss_list, acc_list, time_list = [], [], []
    gamma_layer_list = []
    params = init_params.clone()
    for t in range(args.T):
        out = train_fedlws(args.dataset, train_datasets, params, args.batch_size,
                           args.E, args.mu, device, args.K, make_model(args),
                           criterion, args.opt, lws_beta=args.lws_beta,
                           measure_time=args.measure_time)
        params, loss, gamma_layers = out[0], out[1], out[2]
        if args.measure_time:
            time_list.append(out[3])
        acc = test(params, test_loader, device, make_model(args), args.dataset)
        loss_list.append(loss)
        acc_list.append(acc)
        if t == 0:
            gamma_layer_list = [[] for _ in range(len(gamma_layers))]
        for li, g in enumerate(gamma_layers):
            gamma_layer_list[li].append(g)
        print(f'Round {t}: loss={loss:.4f}, acc={acc:.4f}', flush=True)
    results['FedLWS'] = {'loss': loss_list, 'acc': acc_list}
    figure_gamma_layer(gamma_layer_list, os.path.join(args.result_dir, 'gamma_layer_fedlws.png'))
    if args.measure_time:
        print(f'FedLWS avg agg time: {np.mean(time_list):.6f} s', flush=True)

    # ------------------------------------------------------------------
    # FedHyper-G
    # ------------------------------------------------------------------
    print('\n=== FedHyper-G ===', flush=True)
    loss_list, acc_list, time_list = [], [], []
    thetast_list = []
    params = init_params.clone()
    thetast = args.init_thetast
    dvec = None
    for t in range(args.T):
        out = train_fedhyp(args.dataset, train_datasets, params, args.batch_size,
                           args.E, args.mu, device, args.K, make_model(args),
                           criterion, args.opt, loop=t, dpre=dvec, thetast=thetast,
                           measure_time=args.measure_time)
        params, loss, thetast, dvec = out[0], out[1], out[2], out[3]
        if args.measure_time:
            time_list.append(out[4])
        acc = test(params, test_loader, device, make_model(args), args.dataset)
        loss_list.append(loss)
        acc_list.append(acc)
        thetast_list.append(thetast)
        print(f'Round {t}: loss={loss:.4f}, acc={acc:.4f}', flush=True)
    results['FedHyper-G'] = {'loss': loss_list, 'acc': acc_list}
    figure_thetast(thetast_list, os.path.join(args.result_dir, 'thetast_fedhyperg.png'))
    if args.measure_time:
        print(f'FedHyper-G avg agg time: {np.mean(time_list):.6f} s', flush=True)

    # ------------------------------------------------------------------
    # FedLAW
    # ------------------------------------------------------------------
    print('\n=== FedLAW ===', flush=True)
    loss_list, acc_list, time_list = [], [], []
    gammat_list = []
    sm_lambdat_list = [[] for _ in range(args.K)]
    params = init_params.clone()
    for t in range(args.T):
        proxy_model = make_proxy_model(args, init_gammat, init_lambdat)
        out = train_fedlaw(args.dataset, train_datasets, params, args.batch_size,
                           args.E, args.mu, device, args.K, make_model(args),
                           criterion, args.opt, model_proxy=proxy_model,
                           Eproxy=args.Eproxy, muproxy=args.muproxy,
                           proxy_loader=proxy_loader, opt_proxy=args.opt_proxy,
                           measure_time=args.measure_time)
        params, loss, gammat, lambdat = out[0], out[1], out[2], out[3]
        if args.measure_time:
            time_list.append(out[5])
        acc = test(params, test_loader, device, make_model(args), args.dataset)
        loss_list.append(loss)
        acc_list.append(acc)
        gammat_list.append(np.exp(gammat))
        sm_lambdat = softmax_vec(lambdat)
        for k in range(args.K):
            sm_lambdat_list[k].append(sm_lambdat[k])
        print(f'Round {t}: loss={loss:.4f}, acc={acc:.4f}, exp(gamma)={np.exp(gammat):.4f}', flush=True)
    results['FedLAW'] = {'loss': loss_list, 'acc': acc_list}
    figure_gamma(gammat_list, os.path.join(args.result_dir, 'gamma_fedlaw.png'))
    figure_weights(sm_lambdat_list, os.path.join(args.result_dir, 'weights_fedlaw.png'))
    if args.measure_time:
        print(f'FedLAW avg agg time: {np.mean(time_list):.6f} s', flush=True)

    # ------------------------------------------------------------------
    # FedHAW (proposed)
    # ------------------------------------------------------------------
    print('\n=== FedHAW (proposed) ===', flush=True)
    loss_list, acc_list, time_list = [], [], []
    gammat_list = []
    sm_lambdat_list = [[] for _ in range(args.K)]
    params = init_params.clone()
    gammat = init_gammat
    lambdat = init_lambdat.copy()
    params_candidates_pre = None
    for t in range(args.T):
        out = train_fedhaw(args.dataset, train_datasets, params, args.batch_size,
                           args.E, args.mu, device, args.K, make_model(args),
                           criterion, args.opt, loop=t,
                           params_candidates_pre=params_candidates_pre,
                           gammat=gammat, lambdat=lambdat,
                           lr_gamma=args.lr_gamma, lr_lambda=args.lr_lambda,
                           measure_time=args.measure_time)
        params, loss, gammat, lambdat, params_candidates_pre = out[0], out[1], out[2], out[3], out[4]
        if args.measure_time:
            time_list.append(out[5])
        acc = test(params, test_loader, device, make_model(args), args.dataset)
        loss_list.append(loss)
        acc_list.append(acc)
        gammat_list.append(np.exp(gammat))
        sm_lambdat = softmax_vec(lambdat)
        for k in range(args.K):
            sm_lambdat_list[k].append(sm_lambdat[k])
        print(f'Round {t}: loss={loss:.4f}, acc={acc:.4f}, exp(gamma)={np.exp(gammat):.4f}', flush=True)
    results['FedHAW'] = {'loss': loss_list, 'acc': acc_list}
    figure_gamma(gammat_list, os.path.join(args.result_dir, 'gamma_fedhaw.png'))
    figure_weights(sm_lambdat_list, os.path.join(args.result_dir, 'weights_fedhaw.png'))
    if args.measure_time:
        print(f'FedHAW avg agg time: {np.mean(time_list):.6f} s', flush=True)

    # ------------------------------------------------------------------
    # Plot accuracy and loss curves
    # ------------------------------------------------------------------
    styles = {
        'FedAvg':     dict(linestyle=':', linewidth=2),
        'FedAdp':     dict(linestyle='--', marker='o', markevery=max(1, args.T // 20), linewidth=2),
        'FedLWS':     dict(linestyle='-.', marker='^', markevery=max(1, args.T // 20), linewidth=2),
        'FedHyper-G': dict(linestyle='-.', marker='d', markevery=max(1, args.T // 20), linewidth=2),
        'FedLAW':     dict(linestyle='--', linewidth=2),
        'FedHAW':     dict(linestyle='-', linewidth=2),
    }
    for metric in ['acc', 'loss']:
        plt.figure()
        for name, data in results.items():
            label = 'FedHAW (proposed)' if name == 'FedHAW' else name
            plt.plot(data[metric], label=label, **styles[name])
        plt.xlabel('round $t$', fontsize=16)
        plt.ylabel('accuracy' if metric == 'acc' else 'loss', fontsize=16)
        plt.legend(fontsize=14)
        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(args.result_dir, f'{metric}.pdf'))
        plt.close()

    print(f'\nResults saved to {args.result_dir}', flush=True)


if __name__ == '__main__':
    main()
