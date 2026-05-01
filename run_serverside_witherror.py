"""Sec. IV-C: Comparison of FL aggregation methods under stochastic client errors.

Same algorithms as Sec. IV-B, but each client has a randomly assigned per-round
error probability. A client with an error sends its current model unchanged,
simulating packet loss or a compute failure.

Data directory convention:
    ./data/{dataset}/alpha01/   (for Dirichlet alpha=0.1)
    ./data/{dataset}/alpha1/    (for Dirichlet alpha=1.0)

Usage examples:
    # MNIST
    python run_serverside_witherror.py --dataset mnist --dir_alpha 0.1 \\
        --max_error_rate 0.2 --randseed 23

    # CIFAR-10
    python run_serverside_witherror.py --dataset cifar10 --dir_alpha 0.1 \\
        --T 50 --E 10 --mu 0.1 --opt SGDdecay --opt_proxy Adam --max_error_rate 0.2
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector

from functions.training_witherror import (
    train_fedavg_witherror, train_fedhaw_witherror, train_fedlaw_witherror,
    train_fedadp_witherror, train_fedhyp_witherror, train_fedlws_witherror
)
from functions.training import softmax_vec
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
    if alpha == int(alpha):
        return 'alpha' + str(int(alpha))
    else:
        return 'alpha' + str(alpha).replace('0.', '').replace('.', '')


def get_args():
    parser = argparse.ArgumentParser(
        description='FedHAW - Sec. IV-C: comparison under stochastic client errors')
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'dogs', 'cifar10'])
    parser.add_argument('--dir_alpha', type=float, default=0.1,
                        help='Dirichlet alpha used when partitioning local data')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override for local training data directory. '
                             'Defaults to ./data/{dataset}/{alpha_str}/')
    parser.add_argument('--result_dir', type=str, default='./results')

    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--T', type=int, default=200)
    parser.add_argument('--E', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mu', type=float, default=0.001)
    parser.add_argument('--opt', type=str, default='SGD',
                        choices=['SGD', 'SGDmomentum', 'SGDdecay', 'Adam'])

    # Error model
    parser.add_argument('--max_error_rate', type=float, default=0.2,
                        help='Each client error rate is drawn uniformly from [0, max_error_rate]')
    parser.add_argument('--randseed', type=int, default=23,
                        help='Random seed for drawing per-client error rates')

    # FedHAW
    parser.add_argument('--lr_gamma', type=float, default=0.0001)
    parser.add_argument('--lr_lambda', type=float, default=0.01)

    # FedLAW
    parser.add_argument('--Eproxy', type=int, default=100)
    parser.add_argument('--muproxy', type=float, default=0.01)
    parser.add_argument('--opt_proxy', type=str, default='SGD',
                        choices=['SGD', 'SGDmomentum', 'Adam'])

    # FedAdp
    parser.add_argument('--alpha_adp', type=float, default=5.0)

    # FedHyper-G
    parser.add_argument('--init_thetast', type=float, default=1.0)

    # FedLWS
    parser.add_argument('--lws_beta', type=float, default=0.1)

    # Model
    parser.add_argument('--num_feature', type=int, default=128)
    parser.add_argument('--dogs_inputsize', type=int, default=768)
    parser.add_argument('--dogs_outputsize', type=int, default=120)

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

    astr = alpha_str(args.dir_alpha)
    data_dir = args.data_dir or os.path.join('./data', args.dataset, astr)
    dogs_test_dir = os.path.join('./data', 'dogs') if args.dataset == 'dogs' else None

    os.makedirs(args.result_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    print(f'Data directory: {data_dir}', flush=True)

    criterion = nn.NLLLoss() if args.dataset in ['mnist', 'dogs'] else nn.CrossEntropyLoss()

    test_loader, proxy_loader = set_loaders(args.dataset, args.batch_size,
                                             dogs_data_dir=dogs_test_dir)
    train_datasets = set_local_data(args.K, data_dir)
    Nk = [len(train_datasets[k]) for k in range(args.K)]
    N = sum(Nk)
    init_lambdat = [np.log(Nk[k] / N) for k in range(args.K)]
    init_gammat = np.log(1.0)

    global_model = make_model(args).to(device)
    init_params = parameters_to_vector(global_model.parameters()).detach().clone()

    # Generate stochastic error flags for all rounds and clients.
    # Each client k has a fixed per-round error probability drawn once
    # from Uniform[0, max_error_rate]. error_flag[t, k]=1 means client k
    # experiences an error (sends stale model) in round t.
    np.random.seed(args.randseed)
    clients_error_rate = np.random.rand(args.K) * args.max_error_rate
    error_flag = (np.random.rand(args.T, args.K) < clients_error_rate).astype(int)
    print(f'Client error rates: {clients_error_rate}', flush=True)

    results = {}

    # ------------------------------------------------------------------
    # FedAvg
    # ------------------------------------------------------------------
    print('\n=== FedAvg ===', flush=True)
    loss_list, acc_list = [], []
    params = init_params.clone()
    for t in range(args.T):
        params, loss = train_fedavg_witherror(
            args.dataset, train_datasets, params, args.batch_size,
            args.E, args.mu, device, args.K, make_model(args),
            criterion, args.opt, error_flag[t]
        )
        acc = test(params, test_loader, device, make_model(args), args.dataset)
        loss_list.append(loss)
        acc_list.append(acc)
        print(f'Round {t}: loss={loss:.4f}, acc={acc:.4f}', flush=True)
    results['FedAvg'] = {'loss': loss_list, 'acc': acc_list}

    # ------------------------------------------------------------------
    # FedAdp
    # ------------------------------------------------------------------
    print('\n=== FedAdp ===', flush=True)
    loss_list, acc_list = [], []
    params = init_params.clone()
    lambdat = None
    for t in range(args.T):
        params, loss, lambdat = train_fedadp_witherror(
            args.dataset, train_datasets, params, args.batch_size,
            args.E, args.mu, device, args.K, make_model(args),
            criterion, args.opt, loop=t, alpha=args.alpha_adp,
            lambdat_pre=lambdat, error_flag=error_flag[t]
        )
        acc = test(params, test_loader, device, make_model(args), args.dataset)
        loss_list.append(loss)
        acc_list.append(acc)
        print(f'Round {t}: loss={loss:.4f}, acc={acc:.4f}', flush=True)
    results['FedAdp'] = {'loss': loss_list, 'acc': acc_list}

    # ------------------------------------------------------------------
    # FedLWS
    # ------------------------------------------------------------------
    print('\n=== FedLWS ===', flush=True)
    loss_list, acc_list = [], []
    gamma_layer_list = []
    params = init_params.clone()
    for t in range(args.T):
        params, loss, gamma_layers = train_fedlws_witherror(
            args.dataset, train_datasets, params, args.batch_size,
            args.E, args.mu, device, args.K, make_model(args),
            criterion, args.opt, lws_beta=args.lws_beta, error_flag=error_flag[t]
        )
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

    # ------------------------------------------------------------------
    # FedHyper-G
    # ------------------------------------------------------------------
    print('\n=== FedHyper-G ===', flush=True)
    loss_list, acc_list = [], []
    thetast_list = []
    params = init_params.clone()
    thetast = args.init_thetast
    dvec = None
    for t in range(args.T):
        params, loss, thetast, dvec = train_fedhyp_witherror(
            args.dataset, train_datasets, params, args.batch_size,
            args.E, args.mu, device, args.K, make_model(args),
            criterion, args.opt, loop=t, dpre=dvec, thetast=thetast,
            error_flag=error_flag[t]
        )
        acc = test(params, test_loader, device, make_model(args), args.dataset)
        loss_list.append(loss)
        acc_list.append(acc)
        thetast_list.append(thetast)
        print(f'Round {t}: loss={loss:.4f}, acc={acc:.4f}', flush=True)
    results['FedHyper-G'] = {'loss': loss_list, 'acc': acc_list}
    figure_thetast(thetast_list, os.path.join(args.result_dir, 'thetast_fedhyperg.png'))

    # ------------------------------------------------------------------
    # FedLAW
    # ------------------------------------------------------------------
    print('\n=== FedLAW ===', flush=True)
    loss_list, acc_list = [], []
    gammat_list = []
    sm_lambdat_list = [[] for _ in range(args.K)]
    params = init_params.clone()
    for t in range(args.T):
        proxy_model = make_proxy_model(args, init_gammat, init_lambdat)
        params, loss, gammat, lambdat, _ = train_fedlaw_witherror(
            args.dataset, train_datasets, params, args.batch_size,
            args.E, args.mu, device, args.K, make_model(args),
            criterion, args.opt, model_proxy=proxy_model,
            Eproxy=args.Eproxy, muproxy=args.muproxy,
            proxy_loader=proxy_loader, opt_proxy=args.opt_proxy,
            error_flag=error_flag[t]
        )
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

    # ------------------------------------------------------------------
    # FedHAW (proposed)
    # ------------------------------------------------------------------
    print('\n=== FedHAW (proposed) ===', flush=True)
    loss_list, acc_list = [], []
    gammat_list = []
    sm_lambdat_list = [[] for _ in range(args.K)]
    params = init_params.clone()
    gammat = init_gammat
    lambdat = init_lambdat.copy()
    params_candidates_pre = None
    for t in range(args.T):
        params, loss, gammat, lambdat, params_candidates_pre = train_fedhaw_witherror(
            args.dataset, train_datasets, params, args.batch_size,
            args.E, args.mu, device, args.K, make_model(args),
            criterion, args.opt, loop=t,
            params_candidates_pre=params_candidates_pre,
            gammat=gammat, lambdat=lambdat,
            lr_gamma=args.lr_gamma, lr_lambda=args.lr_lambda,
            error_flag=error_flag[t]
        )
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

    # ------------------------------------------------------------------
    # Plots
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
