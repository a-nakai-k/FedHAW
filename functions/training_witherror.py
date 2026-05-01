"""Training functions for federated learning under stochastic client errors.

Each client has a randomly assigned error rate. In each round, client k
transmits an error (i.e., returns its model unchanged from the previous round)
with probability equal to its error rate. The error_flag array (shape [K])
encodes which clients experience an error in a given round (1 = error, 0 = ok).
"""

import torch
import torch.optim as optim
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .utils import softmax_vec_adp, vector_to_layer_params_grouped
from .training import softmax_vec, _make_optimizer


def _local_train_witherror(dataname, dataset, params_vector_pre, batch_size, E, mu,
                            device, model, criterion, opt, error):
    """Run local training on one client; skip training if the client has an error.

    When error=1, the client sends back its current (unchanged) model, i.e.,
    it contributes params_vector_pre to the aggregation.

    Returns:
        Updated parameter vector (or params_vector_pre if error=1), running loss.
    """
    if dataname == 'dogs':
        inputsize = model.l1.weight.shape[1]

    local_model = model.to(device)
    vector_to_parameters(params_vector_pre.detach().clone(), local_model.parameters())

    if error == 0:
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
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
    else:
        running_loss = 0.0

    return parameters_to_vector(local_model.parameters()).detach().clone(), running_loss


def train_fedavg_witherror(dataname, datasets, params_vector_pre, batch_size, E, mu,
                            device, K, model, criterion, opt, error_flag):
    """FedAvg under stochastic client errors."""
    params_candidates, pk, total_loss = [], [], 0.0

    for node in range(K):
        pk.append(len(datasets[node]))
        p, loss = _local_train_witherror(dataname, datasets[node], params_vector_pre, batch_size,
                                          E, mu, device, model, criterion, opt, error_flag[node])
        params_candidates.append(p)
        total_loss += loss

    pk = np.array(pk) / sum(pk)
    with torch.no_grad():
        params_vector_global = sum(pk[node] * params_candidates[node] for node in range(K))
    return params_vector_global.detach().clone(), total_loss / K


def train_fedhaw_witherror(dataname, datasets, params_vector_pre, batch_size, E, mu,
                            device, K, model, criterion, opt,
                            loop, params_candidates_pre, gammat, lambdat,
                            lr_gamma, lr_lambda, error_flag):
    """FedHAW under stochastic client errors (Sec. IV-C)."""
    params_candidates, total_loss = [], 0.0

    for node in range(K):
        p, loss = _local_train_witherror(dataname, datasets[node], params_vector_pre, batch_size,
                                          E, mu, device, model, criterion, opt, error_flag[node])
        params_candidates.append(p)
        total_loss += loss

    with torch.no_grad():
        if loop > 0:
            sm_lambdatpre = softmax_vec(lambdat)
            params_tmpsum = sum(sm_lambdatpre[node] * params_candidates[node] for node in range(K))
            tmpvec = params_vector_pre - params_tmpsum
            gammat = gammat - lr_gamma * np.exp(gammat) / mu * torch.sum(tmpvec * params_vector_pre).item()
            lambdat = [
                lambdat[node] - lr_lambda * np.exp(2 * gammat) * sm_lambdatpre[node] * (1 - sm_lambdatpre[node]) / mu
                * torch.sum(tmpvec * params_candidates_pre[node]).item()
                for node in range(K)
            ]
        sm_lambdat = softmax_vec(lambdat)
        params_vector_global = sum(sm_lambdat[node] * params_candidates[node] for node in range(K))
        params_vector_global = params_vector_global * np.exp(gammat)

    return params_vector_global.detach().clone(), total_loss / K, gammat, lambdat, params_candidates


def train_fedlaw_witherror(dataname, datasets, params_vector_pre, batch_size, E, mu,
                            device, K, model, criterion, opt,
                            model_proxy, Eproxy, muproxy, proxy_loader, opt_proxy, error_flag):
    """FedLAW under stochastic client errors."""
    params_candidates, total_loss = [], 0.0
    if dataname == 'mnist':
        num_feature = model.l1.weight.shape[0]
    elif dataname == 'dogs':
        inputsize = model.l1.weight.shape[1]
        num_feature = model.l1.weight.shape[0]
        outputsize = model.l3.weight.shape[0]

    for node in range(K):
        p, loss = _local_train_witherror(dataname, datasets[node], params_vector_pre, batch_size,
                                          E, mu, device, model, criterion, opt, error_flag[node])
        params_candidates.append(p)
        total_loss += loss

    modelproxy = model_proxy.to(device)
    if opt_proxy == 'SGD':
        proxy_optimizer = optim.SGD(modelproxy.parameters(), lr=muproxy)
        proxy_scheduler = None
    elif opt_proxy == 'Adam':
        proxy_optimizer = optim.Adam(modelproxy.parameters(), lr=muproxy, betas=(0.5, 0.999))
        proxy_scheduler = optim.lr_scheduler.StepLR(proxy_optimizer, step_size=20, gamma=0.5)
    else:
        proxy_optimizer = optim.SGD(modelproxy.parameters(), lr=muproxy)
        proxy_scheduler = None

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

    return params_vector_global.detach().clone(), total_loss / K, gammat, lambdat, params_candidates


def train_fedadp_witherror(dataname, datasets, params_vector_pre, batch_size, E, mu,
                            device, K, model, criterion, opt,
                            loop, alpha, lambdat_pre, error_flag):
    """FedAdp under stochastic client errors."""
    params_candidates, Nk, total_loss = [], [], 0.0

    for node in range(K):
        Nk.append(len(datasets[node]))
        p, loss = _local_train_witherror(dataname, datasets[node], params_vector_pre, batch_size,
                                          E, mu, device, model, criterion, opt, error_flag[node])
        params_candidates.append(p)
        total_loss += loss

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

    return params_vector_global.detach().clone(), total_loss / K, lambdat


def train_fedhyp_witherror(dataname, datasets, params_vector_pre, batch_size, E, mu,
                            device, K, model, criterion, opt,
                            loop, dpre, thetast, error_flag):
    """FedHyper-G under stochastic client errors."""
    params_candidates, total_loss = [], 0.0

    for node in range(K):
        p, loss = _local_train_witherror(dataname, datasets[node], params_vector_pre, batch_size,
                                          E, mu, device, model, criterion, opt, error_flag[node])
        params_candidates.append(p)
        total_loss += loss

    with torch.no_grad():
        dvec = sum((params_candidates[node] - params_vector_pre) for node in range(K)) / K
        if loop > 0:
            alpha_val = (thetast - 1.0) + torch.sum(dvec * dpre).item()
            alpha_val = max(-0.8, min(4.0, alpha_val))
            thetast = 1.0 + alpha_val
        params_vector_global = params_vector_pre + thetast * dvec

    return params_vector_global.detach().clone(), total_loss / K, thetast, dvec


def train_fedlws_witherror(dataname, datasets, params_vector_pre, batch_size, E, mu,
                            device, K, model, criterion, opt, lws_beta, error_flag):
    """FedLWS under stochastic client errors."""
    params_candidates, Nk, total_loss = [], [], 0.0

    for node in range(K):
        Nk.append(len(datasets[node]))
        p, loss = _local_train_witherror(dataname, datasets[node], params_vector_pre, batch_size,
                                          E, mu, device, model, criterion, opt, error_flag[node])
        params_candidates.append(p)
        total_loss += loss

    pk = np.array(Nk) / sum(Nk)
    with torch.no_grad():
        params_vector_global_hat = sum(pk[node] * params_candidates[node].cpu().numpy() for node in range(K))
        grad_vec = [(params_candidates[node] - params_vector_pre).cpu().numpy() for node in range(K)]
        grad_vec_global = sum(pk[node] * grad_vec[node] for node in range(K))
        grad_vec_mean = sum(grad_vec) / K

        layer_params_hat = vector_to_layer_params_grouped(params_vector_global_hat, model, device)
        layer_params_pre = vector_to_layer_params_grouped(params_vector_pre, model, device)
        layer_grad_global = vector_to_layer_params_grouped(grad_vec_global, model, device)
        layer_grad_mean = vector_to_layer_params_grouped(grad_vec_mean, model, device)
        layer_grads_k = [vector_to_layer_params_grouped(grad_vec[node], model, device) for node in range(K)]

        gamma_layer_list = []
        for layer in range(len(layer_grad_global)):
            tau_layer = sum(
                torch.norm(layer_grads_k[node][layer] - layer_grad_mean[layer], p=2).item()
                for node in range(K)
            ) / K
            norm_pre = torch.norm(layer_params_pre[layer], p=2).item()
            norm_grad = torch.norm(layer_grad_global[layer], p=2).item()
            gamma_layer = norm_pre / (lws_beta * tau_layer * norm_grad + norm_pre)
            gamma_layer_list.append(gamma_layer)
            layer_params_hat[layer] = gamma_layer * layer_params_hat[layer]

        params_vector_global = torch.cat(layer_params_hat)

    return params_vector_global.detach().clone(), total_loss / K, gamma_layer_list


def train_fedprox_witherror(dataname, datasets, params_vector_pre, batch_size, E, mu,
                             device, K, model, criterion, opt, eta_prox, error_flag):
    """FedProx under stochastic client errors."""
    params_candidates, pk, total_loss = [], [], 0.0
    if dataname == 'dogs':
        inputsize = model.l1.weight.shape[1]

    for node in range(K):
        train_loader = torch.utils.data.DataLoader(datasets[node], batch_size=batch_size, shuffle=True)
        pk.append(len(train_loader))
        local_model = model.to(device)
        vector_to_parameters(params_vector_pre.detach().clone(), local_model.parameters())
        if error_flag[node] == 0:
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


def train_fedprox_haw_witherror(dataname, datasets, params_vector_pre, batch_size, E, mu,
                                  device, K, model, criterion, opt, eta_prox,
                                  loop, params_candidates_pre, gammat, lambdat,
                                  lr_gamma, lr_lambda, error_flag):
    """FedProx+HAW: FedProx local objective combined with the HAW aggregation scheme (Sec. IV-D).

    Clients use the proximal regularization during local training, while the
    server applies FedHAW's hypergradient updates for gamma and lambda.
    """
    params_candidates, total_loss = [], 0.0
    if dataname == 'dogs':
        inputsize = model.l1.weight.shape[1]

    for node in range(K):
        train_loader = torch.utils.data.DataLoader(datasets[node], batch_size=batch_size, shuffle=True)
        local_model = model.to(device)
        vector_to_parameters(params_vector_pre.detach().clone(), local_model.parameters())
        if error_flag[node] == 0:
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

    with torch.no_grad():
        if loop > 0:
            sm_lambdatpre = softmax_vec(lambdat)
            params_tmpsum = sum(sm_lambdatpre[node] * params_candidates[node] for node in range(K))
            tmpvec = params_vector_pre - params_tmpsum
            gammat = gammat - lr_gamma * np.exp(gammat) / mu * torch.sum(tmpvec * params_vector_pre).item()
            lambdat = [
                lambdat[node] - lr_lambda * np.exp(2 * gammat) * sm_lambdatpre[node] * (1 - sm_lambdatpre[node]) / mu
                * torch.sum(tmpvec * params_candidates_pre[node]).item()
                for node in range(K)
            ]
        sm_lambdat = softmax_vec(lambdat)
        params_vector_global = sum(sm_lambdat[node] * params_candidates[node] for node in range(K))
        params_vector_global = params_vector_global * np.exp(gammat)

    return params_vector_global.detach().clone(), total_loss / K, gammat, lambdat, params_candidates
