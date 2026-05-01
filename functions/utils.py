import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def softmax_vec(x):
    """Numerically stable softmax over a list or numpy array."""
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp)


def softmax_vec_adp(alpha, lambdat, Nk):
    """Adaptive softmax used by FedAdp.

    Weights are proportional to N_k * exp(alpha * (1 - exp(-exp(-alpha*(lambda_k - 1))))).
    Clients with smaller angle lambda_k (more aligned with global gradient) get higher weight.
    """
    f = alpha * (1 - np.exp(-np.exp(-alpha * (np.array(lambdat) - 1))))
    x_exp = np.array(Nk) * np.exp(f)
    return x_exp / np.sum(x_exp)


def test(params_vector_global, test_loader, device, model, dataname):
    """Evaluate the global model on the test set and return accuracy."""
    correct = 0
    count = 0
    if dataname == 'dogs':
        inputsize = model.l1.weight.shape[1]
    with torch.no_grad():
        model = model.to(device)
        vector_to_parameters(params_vector_global.detach().clone(), model.parameters())
        for (inputs, targets) in test_loader:
            if dataname == 'mnist':
                inputs = inputs.view(-1, 28 * 28)
            elif dataname == 'dogs':
                inputs = inputs.view(-1, inputsize)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum()
            count += inputs.size(0)
    return float(correct) / float(count)


def vector_to_layer_params_grouped(vector, model, device=None):
    """Split a flattened parameter vector into per-module tensors.

    Parameters of each module (weight + bias concatenated) are grouped into a
    single 1D tensor, so that FedLWS can apply layer-wise scaling factors.

    Returns:
        list[Tensor]: one 1D tensor per module that has parameters.
                      torch.cat(result) reconstructs the original flat vector.
    """
    if isinstance(vector, np.ndarray):
        vector = torch.from_numpy(vector).float()
        if device is not None:
            vector = vector.to(device)
        elif next(model.parameters(), None) is not None:
            vector = vector.to(next(model.parameters()).device)
    result = []
    offset = 0
    for _, module in model.named_modules():
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        chunks = []
        for param in params:
            numel = param.numel()
            chunks.append(vector[offset:offset + numel].view(-1))
            offset += numel
        result.append(torch.cat(chunks))
    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def figure_gamma(gammat_list, save_path):
    """Plot the evolution of exp(gamma) over FL rounds."""
    fig = plt.figure()
    plt.plot(range(len(gammat_list)), gammat_list, linewidth=2)
    plt.xlabel(r"round $t$", fontsize=16)
    plt.ylabel(r"$\exp(\gamma_t)$", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def figure_weights(sm_lambdat_list, save_path):
    """Plot the evolution of per-client aggregation weights over FL rounds."""
    fig = plt.figure()
    K = len(sm_lambdat_list)
    T = len(sm_lambdat_list[0])
    for node in range(K):
        plt.plot(range(T), sm_lambdat_list[node], label=f'client {node}')
    plt.legend(fontsize=14)
    plt.xlabel(r"round $t$", fontsize=16)
    plt.ylabel("aggregation weight", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def figure_thetast(thetast_list, save_path):
    """Plot the evolution of the FedHyper-G step-size multiplier theta*."""
    fig = plt.figure()
    plt.plot(range(len(thetast_list)), thetast_list, linewidth=2)
    plt.xlabel(r"round $t$", fontsize=16)
    plt.ylabel(r"$\theta^*_t$", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def figure_gamma_layer(gamma_layer_list, save_path):
    """Plot the per-layer scaling factors of FedLWS over FL rounds.

    Args:
        gamma_layer_list: list of lists indexed as [layer_idx][round].
    """
    fig = plt.figure()
    num_layers = len(gamma_layer_list)
    T = len(gamma_layer_list[0]) if num_layers > 0 else 0
    for layer_idx in range(num_layers):
        plt.plot(range(T), gamma_layer_list[layer_idx], label=f'layer {layer_idx}')
    plt.legend(fontsize=14)
    plt.xlabel(r"round $t$", fontsize=16)
    plt.ylabel(r"$\gamma_\ell$", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
