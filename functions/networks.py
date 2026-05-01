import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MLP for MNIST (input: 784-dim flattened, output: 10 classes)
# ---------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, num_feature):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, num_feature)
        self.l2 = nn.Linear(num_feature, num_feature)
        self.l3 = nn.Linear(num_feature, 10)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# MLP for Stanford Dogs (input: pre-extracted feature vectors)
# ---------------------------------------------------------------------------
class Net_dogs(nn.Module):
    def __init__(self, inputsize, num_feature, outputsize):
        super(Net_dogs, self).__init__()
        self.l1 = nn.Linear(inputsize, num_feature)
        self.l2 = nn.Linear(num_feature, num_feature)
        self.l3 = nn.Linear(num_feature, outputsize)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# Proxy network for FedLAW on MNIST.
# Holds learnable scalar gamma and per-client lambda parameters.
# The forward pass computes the aggregated model output differentiably.
# ---------------------------------------------------------------------------
class ProxyNet_MNIST(nn.Module):
    def __init__(self, init_gammat, init_lambdat):
        super(ProxyNet_MNIST, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(init_gammat, dtype=torch.float32))
        self.lambdas = nn.Parameter(torch.tensor(init_lambdat.copy(), dtype=torch.float32))

    def softmax_tensor(self, x):
        x_exp = torch.exp(x)
        return x_exp / torch.sum(x_exp)

    def forward(self, params_candidates, x, num_feature):
        K = len(params_candidates)
        sm_lambdat = self.softmax_tensor(self.lambdas)
        params_vector_global = sm_lambdat[0] * params_candidates[0]
        for node in range(1, K):
            params_vector_global += sm_lambdat[node] * params_candidates[node]
        params_vector_global *= torch.exp(self.gamma)

        # Split the flattened parameter vector into layer-wise weights and biases.
        param_idx = 0
        l1_weight_size = num_feature * 784
        l1_weight = params_vector_global[param_idx:param_idx + l1_weight_size].view(num_feature, 784)
        param_idx += l1_weight_size
        l1_bias = params_vector_global[param_idx:param_idx + num_feature]
        param_idx += num_feature

        l2_weight_size = num_feature * num_feature
        l2_weight = params_vector_global[param_idx:param_idx + l2_weight_size].view(num_feature, num_feature)
        param_idx += l2_weight_size
        l2_bias = params_vector_global[param_idx:param_idx + num_feature]
        param_idx += num_feature

        l3_weight = params_vector_global[param_idx:param_idx + 10 * num_feature].view(10, num_feature)
        param_idx += 10 * num_feature
        l3_bias = params_vector_global[param_idx:param_idx + 10]

        x = x.view(x.size(0), -1)
        x = F.relu(F.linear(x, l1_weight, l1_bias))
        x = F.relu(F.linear(x, l2_weight, l2_bias))
        x = F.log_softmax(F.linear(x, l3_weight, l3_bias), dim=1)
        return x


# ---------------------------------------------------------------------------
# Proxy network for FedLAW on Stanford Dogs.
# ---------------------------------------------------------------------------
class ProxyNet_dogs(nn.Module):
    def __init__(self, init_gammat, init_lambdat):
        super(ProxyNet_dogs, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(init_gammat, dtype=torch.float32))
        self.lambdas = nn.Parameter(torch.tensor(init_lambdat.copy(), dtype=torch.float32))

    def softmax_tensor(self, x):
        x_exp = torch.exp(x)
        return x_exp / torch.sum(x_exp)

    def forward(self, params_candidates, x, inputsize, num_feature, outputsize):
        K = len(params_candidates)
        sm_lambdat = self.softmax_tensor(self.lambdas)
        params_vector_global = sm_lambdat[0] * params_candidates[0]
        for node in range(1, K):
            params_vector_global += sm_lambdat[node] * params_candidates[node]
        params_vector_global *= torch.exp(self.gamma)

        param_idx = 0
        l1_weight = params_vector_global[param_idx:param_idx + num_feature * inputsize].view(num_feature, inputsize)
        param_idx += num_feature * inputsize
        l1_bias = params_vector_global[param_idx:param_idx + num_feature]
        param_idx += num_feature

        l2_weight = params_vector_global[param_idx:param_idx + num_feature * num_feature].view(num_feature, num_feature)
        param_idx += num_feature * num_feature
        l2_bias = params_vector_global[param_idx:param_idx + num_feature]
        param_idx += num_feature

        l3_weight = params_vector_global[param_idx:param_idx + outputsize * num_feature].view(outputsize, num_feature)
        param_idx += outputsize * num_feature
        l3_bias = params_vector_global[param_idx:param_idx + outputsize]

        x = x.view(x.size(0), -1)
        x = F.relu(F.linear(x, l1_weight, l1_bias))
        x = F.relu(F.linear(x, l2_weight, l2_bias))
        x = F.log_softmax(F.linear(x, l3_weight, l3_bias), dim=1)
        return x


# ---------------------------------------------------------------------------
# ResNet-18 adapted for CIFAR-10 (32x32 images).
# Uses 3x3 conv at the first layer and stride=1 (no large downsampling).
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Resnet18Cifar(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet18Cifar, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)


# ---------------------------------------------------------------------------
# Proxy network for FedLAW on CIFAR-10 with ResNet-18.
# The forward pass manually reconstructs the ResNet-18 computation from
# the aggregated (flattened) parameter vector so that gradients flow through
# gamma and lambda.
# ---------------------------------------------------------------------------
class ProxyNet_CIFAR10_Resnet18(nn.Module):
    def __init__(self, init_gammat, init_lambdat):
        super(ProxyNet_CIFAR10_Resnet18, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(init_gammat, dtype=torch.float32))
        self.lambdas = nn.Parameter(torch.tensor(init_lambdat.copy(), dtype=torch.float32))

    def softmax_tensor(self, x):
        x_exp = torch.exp(x)
        return x_exp / torch.sum(x_exp)

    def forward(self, params_candidates, x):
        K = len(params_candidates)
        sm_lambdat = self.softmax_tensor(self.lambdas)
        params_vector_global = sm_lambdat[0] * params_candidates[0]
        for node in range(1, K):
            params_vector_global += sm_lambdat[node] * params_candidates[node]
        params_vector_global *= torch.exp(self.gamma)

        idx = 0

        def _take(n):
            nonlocal idx
            v = params_vector_global[idx:idx + n]
            idx += n
            return v

        # conv1: (64, 3, 3, 3), no bias
        conv1_w = _take(64 * 3 * 3 * 3).view(64, 3, 3, 3)
        # bn1: weight + bias
        bn1_w = _take(64)
        bn1_b = _take(64)

        # Helper to apply a basic block (no shortcut)
        def basic_block_no_sc(x, c1w, bn1w, bn1b, c2w, bn2w, bn2b, ch):
            identity = x
            dev = x.device
            out = F.conv2d(x, c1w, None, padding=1)
            out = F.batch_norm(out, torch.zeros(ch, device=dev), torch.ones(ch, device=dev), bn1w, bn1b, training=True)
            out = F.relu(out)
            out = F.conv2d(out, c2w, None, padding=1)
            out = F.batch_norm(out, torch.zeros(ch, device=dev), torch.ones(ch, device=dev), bn2w, bn2b, training=True)
            out += identity
            return F.relu(out)

        # Helper to apply a basic block with a 1x1 shortcut convolution
        def basic_block_with_sc(x, stride, c1w, bn1w, bn1b, c2w, bn2w, bn2b, sc_cw, sc_bnw, sc_bnb, in_ch, out_ch):
            identity = x
            dev = x.device
            out = F.conv2d(x, c1w, None, stride=stride, padding=1)
            out = F.batch_norm(out, torch.zeros(out_ch, device=dev), torch.ones(out_ch, device=dev), bn1w, bn1b, training=True)
            out = F.relu(out)
            out = F.conv2d(out, c2w, None, padding=1)
            out = F.batch_norm(out, torch.zeros(out_ch, device=dev), torch.ones(out_ch, device=dev), bn2w, bn2b, training=True)
            identity = F.conv2d(identity, sc_cw, None, stride=stride)
            identity = F.batch_norm(identity, torch.zeros(out_ch, device=dev), torch.ones(out_ch, device=dev), sc_bnw, sc_bnb, training=True)
            out += identity
            return F.relu(out)

        # layer1: 2 blocks, 64->64 (no shortcut needed)
        l1b0_c1 = _take(64 * 64 * 3 * 3).view(64, 64, 3, 3)
        l1b0_bn1w, l1b0_bn1b = _take(64), _take(64)
        l1b0_c2 = _take(64 * 64 * 3 * 3).view(64, 64, 3, 3)
        l1b0_bn2w, l1b0_bn2b = _take(64), _take(64)
        l1b1_c1 = _take(64 * 64 * 3 * 3).view(64, 64, 3, 3)
        l1b1_bn1w, l1b1_bn1b = _take(64), _take(64)
        l1b1_c2 = _take(64 * 64 * 3 * 3).view(64, 64, 3, 3)
        l1b1_bn2w, l1b1_bn2b = _take(64), _take(64)

        # layer2: 2 blocks, 64->128, stride=2 (block0 has shortcut)
        l2b0_c1 = _take(128 * 64 * 3 * 3).view(128, 64, 3, 3)
        l2b0_bn1w, l2b0_bn1b = _take(128), _take(128)
        l2b0_c2 = _take(128 * 128 * 3 * 3).view(128, 128, 3, 3)
        l2b0_bn2w, l2b0_bn2b = _take(128), _take(128)
        l2b0_sc_c = _take(128 * 64).view(128, 64, 1, 1)
        l2b0_sc_bnw, l2b0_sc_bnb = _take(128), _take(128)
        l2b1_c1 = _take(128 * 128 * 3 * 3).view(128, 128, 3, 3)
        l2b1_bn1w, l2b1_bn1b = _take(128), _take(128)
        l2b1_c2 = _take(128 * 128 * 3 * 3).view(128, 128, 3, 3)
        l2b1_bn2w, l2b1_bn2b = _take(128), _take(128)

        # layer3: 2 blocks, 128->256, stride=2 (block0 has shortcut)
        l3b0_c1 = _take(256 * 128 * 3 * 3).view(256, 128, 3, 3)
        l3b0_bn1w, l3b0_bn1b = _take(256), _take(256)
        l3b0_c2 = _take(256 * 256 * 3 * 3).view(256, 256, 3, 3)
        l3b0_bn2w, l3b0_bn2b = _take(256), _take(256)
        l3b0_sc_c = _take(256 * 128).view(256, 128, 1, 1)
        l3b0_sc_bnw, l3b0_sc_bnb = _take(256), _take(256)
        l3b1_c1 = _take(256 * 256 * 3 * 3).view(256, 256, 3, 3)
        l3b1_bn1w, l3b1_bn1b = _take(256), _take(256)
        l3b1_c2 = _take(256 * 256 * 3 * 3).view(256, 256, 3, 3)
        l3b1_bn2w, l3b1_bn2b = _take(256), _take(256)

        # layer4: 2 blocks, 256->512, stride=2 (block0 has shortcut)
        l4b0_c1 = _take(512 * 256 * 3 * 3).view(512, 256, 3, 3)
        l4b0_bn1w, l4b0_bn1b = _take(512), _take(512)
        l4b0_c2 = _take(512 * 512 * 3 * 3).view(512, 512, 3, 3)
        l4b0_bn2w, l4b0_bn2b = _take(512), _take(512)
        l4b0_sc_c = _take(512 * 256).view(512, 256, 1, 1)
        l4b0_sc_bnw, l4b0_sc_bnb = _take(512), _take(512)
        l4b1_c1 = _take(512 * 512 * 3 * 3).view(512, 512, 3, 3)
        l4b1_bn1w, l4b1_bn1b = _take(512), _take(512)
        l4b1_c2 = _take(512 * 512 * 3 * 3).view(512, 512, 3, 3)
        l4b1_bn2w, l4b1_bn2b = _take(512), _take(512)

        # linear: 512->10
        lin_w = _take(10 * 512).view(10, 512)
        lin_b = _take(10)

        # Forward pass
        dev = x.device
        x = F.conv2d(x, conv1_w, None, padding=1)
        x = F.batch_norm(x, torch.zeros(64, device=dev), torch.ones(64, device=dev), bn1_w, bn1_b, training=True)
        x = F.relu(x)

        x = basic_block_no_sc(x, l1b0_c1, l1b0_bn1w, l1b0_bn1b, l1b0_c2, l1b0_bn2w, l1b0_bn2b, 64)
        x = basic_block_no_sc(x, l1b1_c1, l1b1_bn1w, l1b1_bn1b, l1b1_c2, l1b1_bn2w, l1b1_bn2b, 64)

        x = basic_block_with_sc(x, 2, l2b0_c1, l2b0_bn1w, l2b0_bn1b, l2b0_c2, l2b0_bn2w, l2b0_bn2b, l2b0_sc_c, l2b0_sc_bnw, l2b0_sc_bnb, 64, 128)
        x = basic_block_no_sc(x, l2b1_c1, l2b1_bn1w, l2b1_bn1b, l2b1_c2, l2b1_bn2w, l2b1_bn2b, 128)

        x = basic_block_with_sc(x, 2, l3b0_c1, l3b0_bn1w, l3b0_bn1b, l3b0_c2, l3b0_bn2w, l3b0_bn2b, l3b0_sc_c, l3b0_sc_bnw, l3b0_sc_bnb, 128, 256)
        x = basic_block_no_sc(x, l3b1_c1, l3b1_bn1w, l3b1_bn1b, l3b1_c2, l3b1_bn2w, l3b1_bn2b, 256)

        x = basic_block_with_sc(x, 2, l4b0_c1, l4b0_bn1w, l4b0_bn1b, l4b0_c2, l4b0_bn2w, l4b0_bn2b, l4b0_sc_c, l4b0_sc_bnw, l4b0_sc_bnb, 256, 512)
        x = basic_block_no_sc(x, l4b1_c1, l4b1_bn1w, l4b1_bn1b, l4b1_c2, l4b1_bn2w, l4b1_bn2b, 512)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        return F.linear(x, lin_w, lin_b)
