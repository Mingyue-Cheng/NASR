import torch
import torch.nn as nn
import torch.nn.functional as F


class CLNeck(nn.Module):
    """
    for a user x's representation vector X in shape of L * D,
    use CLNeck to project X to a vector in shape D
    """

    def __init__(self, seq_len, embed_size, num_layer=1):
        super(CLNeck, self).__init__()

        self.input_dim = seq_len * embed_size
        self.fc_names = []
        self.bn_names = []
        self.relu = nn.ReLU()

        for i in range(num_layer):
            hidden_dim = self.input_dim if i < num_layer - 1 else embed_size
            self.add_module("fc{}".format(i), nn.Linear(self.input_dim, hidden_dim))
            self.fc_names.append("fc{}".format(i))

            self.add_module("bn{}".format(i), nn.BatchNorm1d(hidden_dim))
            self.bn_names.append("bn{}".format(i))

    def forward(self, x):
        # x in shape of N * L * D
        x = x.view(x.shape[0], -1)
        assert x.shape[-1] == self.input_dim
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            x = self.relu(x)
            x = fc(x)
            bn = getattr(self, bn_name)
            x = bn(x)
        return x


class CLHead(nn.Module):
    """
    compute the L2-distance between student net and teacher net
    """

    def __init__(self, embed_size):
        super(CLHead, self).__init__()
        self.predictor = CLNeck(1, embed_size, num_layer=2)

    def forward(self, input, target):
        # input&target in shape of N * D
        pred = self.predictor(input)
        # pred = input
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        target_norm = target_norm.detach()
        loss = -2 * (pred_norm * target_norm).sum()
        loss /= input.shape[0]
        return loss
