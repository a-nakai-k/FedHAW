"""Training functions for federated learning algorithms (no client errors).

Each function performs one round of federated learning:
  1. Local training on each client.
  2. Server-side aggregation.

All functions accept `measure_time=False`. When True they additionally return
the wall-clock time (seconds) of the aggregation step, which is used to
compare server-side computational costs across methods (Sec. IV-B).
"""

import time
import torch
import torch.optim as optim
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .utils import softmax_vec_adp, vector_to_layer_params_grouped


def softmax_vec(x):
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp)


def _make_optimizer(model, opt, mu):
    """Create a local optimizer for a client model."""
    if opt == 'SGD':
        return optim.SGD(model.parameters(), lr=mu), None
    elif opt == 'SGDmomentum':
        optimizer = optim.SGD(model.parameters(), lr=mu, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)
        return optimizer, scheduler
    elif opt == 'SGDdecay':
        return optim.SGD(model.parameters(), lr=mu, weight_decay=1e-4), None
    elif opt == 'Adam':
        return optim.Adam(model.parameters(), lr=mu), None
    else:
        raise ValueError(f"Unknown optimizer: {opt}")


def _local_train(dataname, dataset, params_vector_pre, batch_size, E, mu, device, model, criterion, opt):
    """Run local SGD on one client and return the updated parameter vector and total loss."""
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    if dataname == 'dogs':
        inputsize = model.l1.weight.shape[1]

    local_model = model.to(device)
    vector_to_parameters(params_vector_pre.detach().clone(), local_model.parameters())
    optimizer, scheduler = _make_optimizer(local_model, opt, mu)

    running_loss = 0.0
    for _ in range(E):
        for inputs, targets in train_loader:
            if dataname == 'mnist':
                inputs = inputs.view(-1, 28 * 28)
            elif dataname == 'dogs':
                inputs = inputs.view(-1, inputsize)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if scheduler is not None:
            scheduler.step()

    return parameters_to_vector(local_model.parameters()).detach().clone(), running_loss


def train_fedavg(dataname, datasets, params_vector_pre, batch_size, E, mu,
                 device, K, model, criterion, opt, measure_time=False):
    """FedAvg: weighted average of client models (weights proportional to local data size)."""
    params_candidates, pk, total_loss = [], [], 0.0

    for node in range(K):
        pk.append(len(datasets[node]))
        p, loss = _local_train(dataname, datasets[node], params_vector_pre, batch_size, E, mu, device, model, criterion, opt)
        params_candidates.append(p)
        total_loss += loss

    t0 = time.time()
    pk = np.array(pk) / sum(pk)
    with torch.no_grad():
        params_vector_global = sum(pk[node] * params_candidates[node] for node in range(K))
    agg_time = time.time() - t0

    result = (params_vector_global.detach().clone(), total_loss / K)
    return result + (agg_time,) if measure_time else result


def train_fedhaw(dataname, datasets, params_vector_pre, batch_size, E, mu,
                 device, K, model, criterion, opt,
                 loop, params_candidates_pre, gammat, lambdat,
                 lr_gamma, lr_lambda, measure_time=False):
    """FedHAW (proposed): hypergradient-based adaptive aggregation weight (gamma, lambda).

    gamma controls the global step size; lambda_k controls per-client weights.
    Both are updated via hypergradient descent on the server.
    """
    params_candidates, total_loss = [], 0.0

    for node in range(K):
        p, loss = _local_train(dataname, datasets[node], params_vector_pre, batch_size, E, mu, device, model, criterion, opt)
        params_candidates.append(p)
        total_loss += loss

    t0 = time.time()
    with torch.no_grad():
        if loop > 0:
            sm_lambdatpre = softmax_vec(lambdat)
            params_tmpsum = sum(sm_lambdatpre[node] * params_candidates[node] for node in range(K))
            tmpvec = params_vector_pre - params_tmpsum
            # Hypergradient update for gamma (global step-size parameter)
            gammat = gammat - lr_gamma * np.exp(gammat) / mu * torch.sum(tmpvec * params_vector_pre).item()
            # Hypergradient update for lambda_k (per-client weight parameters)
            lambdat = [
                lambdat[node] - lr_lambda * np.exp(2 * gammat) * sm_lambdatpre[node] * (1 - sm_lambdatpre[node]) / mu
                * torch.sum(tmpvec * params_candidates_pre[node]).item()
                for node in range(K)
            ]
        sm_lambdat = softmax_vec(lambdat)
        params_vector_global = sum(sm_lambdat[node] * params_candidates[node] for node in range(K))
        params_vector_global = params_vector_global * np.exp(gammat)
    agg_time = time.time() - t0

    result = (params_vector_global.detach().clone(), total_loss / K, gammat, lambdat, params_candidates)
    return result + (agg_time,) if measure_time else result


def train_fedlaw(dataname, datasets, params_vector_pre, batch_size, E, mu,
                 device, K, model, criterion, opt,
                 model_proxy, Eproxy, muproxy, proxy_loader, opt_proxy,
                 measure_time=False):
    """FedLAW: learns aggregation weights (gamma, lambda) via proxy dataset gradient descent."""
    params_candidates, total_loss = [], 0.0
    if dataname == 'mnist':
        num_feature = model.l1.weight.shape[0]
    elif dataname == 'dogs':
        inputsize = model.l1.weight.shape[1]
        num_feature = model.l1.weight.shape[0]
        outputsize = model.l3.weight.shape[0]

    for node in range(K):
        p, loss = _local_train(dataname, datasets[node], params_vector_pre, batch_size, E, mu, device, model, criterion, opt)
        params_candidates.append(p)
        total_loss += loss

    t0 = time.time()
    # Optimize proxy model parameters on the proxy dataset
    modelproxy = model_proxy.to(device)
    if opt_proxy == 'SGD':
        proxy_optimizer = optim.SGD(modelproxy.parameters(), lr=muproxy)
        proxy_scheduler = None
    elif opt_proxy == 'SGDmomentum':
        proxy_optimizer = optim.SGD(modelproxy.parameters(), lr=muproxy, momentum=0.9)
        proxy_scheduler = None
    elif opt_proxy == 'Adam':
        proxy_optimizer = optim.Adam(modelproxy.parameters(), lr=muproxy, betas=(0.5, 0.999))
        proxy_scheduler = optim.lr_scheduler.StepLR(proxy_optimizer, step_size=20, gamma=0.5)
    else:
        raise ValueError(f"Unknown proxy optimizer: {opt_proxy}")

    for _ in range(Eproxy):
        for inputs, targets in proxy_loader:
            if dataname == 'mnist':
                inputs = inputs.view(-1, 28 * 28)
            elif dataname == 'dogs':
                inputs = inputs.view(-1, inputsize)
            inputs, targets = inputs.to(device), targets.to(device)
            proxy_optimizer.zero_grad()
            if dataname == 'mnist':
                output = modelproxy(params_candidates, inputs, num_feature=num_feature)
            elif dataname == 'dogs':
                output = modelproxy(params_candidates, inputs, inputsize=inputsize,
                                    num_feature=num_feature, outputsize=outputsize)
            else:
                output = modelproxy(params_candidates, inputs)
            loss = criterion(output, targets)
            loss.backward()
            proxy_optimizer.step()
        if proxy_scheduler is not None:
            proxy_scheduler.step()

    gammat = modelproxy.gamma.item()
    lambdat = modelproxy.lambdas.detach().clone().cpu().numpy()

    with torch.no_grad():
        sm_lambdat = softmax_vec(lambdat)
        params_vector_global = sum(sm_lambdat[node] * params_candidates[node] for node in range(K))
        params_vector_global = params_vector_global * np.exp(gammat)
    agg_time = time.time() - t0

    result = (params_vector_global.detach().clone(), total_loss / K, gammat, lambdat, params_candidates)
    return result + (agg_time,) if measure_time else result


def train_fedadp(dataname, datasets, params_vector_pre, batch_size, E, mu,
                 device, K, model, criterion, opt,
                 loop, alpha, lambdat_pre, measure_time=False):
    """FedAdp: angle-based adaptive aggregation.

    Clients whose gradient direction is closer to the global gradient direction
    receive higher weight. The angle lambda_k is tracked with a running average.
    """
    params_candidates, Nk, total_loss = [], [], 0.0

    for node in range(K):
        Nk.append(len(datasets[node]))
        p, loss = _local_train(dataname, datasets[node], params_vector_pre, batch_size, E, mu, device, model, criterion, opt)
        params_candidates.append(p)
        total_loss += loss

    t0 = time.time()
    pk = np.array(Nk) / sum(Nk)
    with torch.no_grad():
        grad_vec, grad_vec_global = [], np.zeros(params_vector_pre.shape, dtype=np.float64)
        for node in range(K):
            gk = (params_vector_pre - params_candidates[node]).cpu().numpy() / mu
            grad_vec.append(gk)
            grad_vec_global += pk[node] * gk
        tmplambdas = np.zeros(K)
        for node in range(K):
            tmplambdas[node] = np.arccos(
                np.clip(
                    np.dot(grad_vec[node], grad_vec_global)
                    / (np.linalg.norm(grad_vec[node]) * np.linalg.norm(grad_vec_global) + 1e-12),
                    -1.0, 1.0
                )
            )
        lambdat = tmplambdas if loop == 0 else (loop * lambdat_pre + tmplambdas) / (loop + 1)
        sm_lambdat = softmax_vec_adp(alpha, lambdat, Nk)
        params_vector_global = sum(sm_lambdat[node] * params_candidates[node] for node in range(K))
    agg_time = time.time() - t0

    result = (params_vector_global.detach().clone(), total_loss / K, lambdat)
    return result + (agg_time,) if measure_time else result


def train_fedhyp(dataname, datasets, params_vector_pre, batch_size, E, mu,
                 device, K, model, criterion, opt,
                 loop, dpre, thetast, measure_time=False):
    """FedHyper-G: hypergradient-based global step-size adaptation.

    The scalar theta* scales the average parameter update direction.
    It is updated based on the inner product between consecutive update directions.
    """
    params_candidates, total_loss = [], 0.0

    for node in range(K):
        p, loss = _local_train(dataname, datasets[node], params_vector_pre, batch_size, E, mu, device, model, criterion, opt)
        params_candidates.append(p)
        total_loss += loss

    t0 = time.time()
    with torch.no_grad():
        dvec = sum((params_candidates[node] - params_vector_pre) for node in range(K)) / K
        if loop > 0:
            alpha = (thetast - 1.0) + torch.sum(dvec * dpre).item()
            alpha = max(-0.8, min(4.0, alpha))
            thetast = 1.0 + alpha
        params_vector_global = params_vector_pre + thetast * dvec
    agg_time = time.time() - t0

    result = (params_vector_global.detach().clone(), total_loss / K, thetast, dvec)
    return result + (agg_time,) if measure_time else result


def train_fedlws(dataname, datasets, params_vector_pre, batch_size, E, mu,
                 device, K, model, criterion, opt, lws_beta, measure_time=False):
    """FedLWS: layer-wise step-size adaptation.

    Each layer receives a scaling factor gamma_l that shrinks the update when
    the layer's gradient variance across clients is high.
    """
    params_candidates, Nk, total_loss = [], [], 0.0

    for node in range(K):
        Nk.append(len(datasets[node]))
        p, loss = _local_train(dataname, datasets[node], params_vector_pre, batch_size, E, mu, device, model, criterion, opt)
        params_candidates.append(p)
        total_loss += loss

    t0 = time.time()
    pk = np.array(Nk) / sum(Nk)
    with torch.no_grad():
        # Compute the weighted average of client parameters (w_hat)
        params_vector_global_hat = sum(pk[node] * params_candidates[node].cpu().numpy() for node in range(K))

        # Per-client gradient vectors (difference from global model)
        grad_vec = [(params_candidates[node] - params_vector_pre).cpu().numpy() for node in range(K)]
        grad_vec_global = sum(pk[node] * grad_vec[node] for node in range(K))  # weighted mean
        grad_vec_mean = sum(grad_vec) / K                                         # unweighted mean

        layer_params_hat = vector_to_layer_params_grouped(params_vector_global_hat, model, device)
        layer_params_pre = vector_to_layer_params_grouped(params_vector_pre, model, device)
        layer_grad_global = vector_to_layer_params_grouped(grad_vec_global, model, device)
        layer_grad_mean = vector_to_layer_params_grouped(grad_vec_mean, model, device)
        layer_grads_k = [vector_to_layer_params_grouped(grad_vec[node], model, device) for node in range(K)]

        gamma_layer_list = []
        for layer in range(len(layer_grad_global)):
            # Layer-wise gradient variance across clients
            tau_layer = sum(
                torch.norm(layer_grads_k[node][layer] - layer_grad_mean[layer], p=2).item()
                for node in range(K)
            ) / K
            norm_pre = torch.norm(layer_params_pre[layer], p=2).item()
            norm_grad = torch.norm(layer_grad_global[layer], p=2).item()
            # Larger variance -> smaller gamma -> more conservative update
            gamma_layer = norm_pre / (lws_beta * tau_layer * norm_grad + norm_pre)
            gamma_layer_list.append(gamma_layer)
            layer_params_hat[layer] = gamma_layer * layer_params_hat[layer]

        params_vector_global = torch.cat(layer_params_hat)
    agg_time = time.time() - t0

    result = (params_vector_global.detach().clone(), total_loss / K, gamma_layer_list)
    return result + (agg_time,) if measure_time else result


def train_fedprox(dataname, datasets, params_vector_pre, batch_size, E, mu,
                  device, K, model, criterion, opt, eta_prox):
    """FedProx: adds a proximal regularization term to each client's local objective."""
    params_candidates, pk, total_loss = [], [], 0.0
    if dataname == 'dogs':
        inputsize = model.l1.weight.shape[1]

    for node in range(K):
        train_loader = torch.utils.data.DataLoader(datasets[node], batch_size=batch_size, shuffle=True)
        pk.append(len(train_loader))
        local_model = model.to(device)
        vector_to_parameters(params_vector_pre.detach().clone(), local_model.parameters())
        optimizer, scheduler = _make_optimizer(local_model, opt, mu)
        for _ in range(E):
            for inputs, targets in train_loader:
                if dataname == 'mnist':
                    inputs = inputs.view(-1, 28 * 28)
                elif dataname == 'dogs':
                    inputs = inputs.view(-1, inputsize)
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                prox = 0.5 * eta_prox * torch.norm(
                    parameters_to_vector(local_model.parameters()) - params_vector_pre.detach().clone(), p=2
                ) ** 2
                loss = criterion(local_model(inputs), targets) + prox
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if scheduler is not None:
                scheduler.step()
        params_candidates.append(parameters_to_vector(local_model.parameters()).detach().clone())

    pk = np.array(pk) / sum(pk)
    with torch.no_grad():
        params_vector_global = sum(pk[node] * params_candidates[node] for node in range(K))
    return params_vector_global.detach().clone(), total_loss / K
