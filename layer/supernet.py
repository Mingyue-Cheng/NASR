import torch
import torch.nn as nn
from .NextItNet import NextItNet
from .TRM import TransformerBlock


class Supernet(nn.Module):
    """
    A supernet is made up of blocks
    """
    def __init__(self, num_block, layers_per_block, d_model, setting):
        super(Supernet, self).__init__()

        self.num_block = num_block
        self.layers = layers_per_block if isinstance(layers_per_block, list) else [layers_per_block] * num_block
        self.d_model = d_model
        self.setting = setting

        self.stem = nn.Identity()
        self._make_block()

    def _make_block(self):
        self._blocks = nn.ModuleList()
        for index in range(self.num_block):
            layers = self.layers[index]
            self._blocks.append(Block(layers, self.d_model, self.setting))

    def forward(self, x, start_block, forward_op):
        if start_block == 0:
            x = self.stem(x)
        for i, block in enumerate(self._blocks):
            if i < start_block:
                continue
            x = block(x, forward_op)
            break
        return x


class Block(nn.Module):
    """
    a block is made up of MixOps
    """
    def __init__(self, layers, d_model, setting):
        super(Block, self).__init__()

        self._block_layers = nn.ModuleList()

        for layer in range(layers):
            self._block_layers.append(MixOp(d_model, setting))

    def forward(self, x, index_list=None):
        assert len(index_list) == len(self._block_layers)
        for i, layer in enumerate(self._block_layers):
            x = layer(x, index_list[i])
        return x


class MixOp(nn.Module):
    """
    a MixOp is one layer in the network, containing all the candidate operations
    """
    def __init__(self, d_model, setting):
        super(MixOp, self).__init__()

        attn_heads, dilations, d_layer = setting
        self._mix_ops = nn.ModuleList()

        for head in attn_heads:
            for _d_layer in d_layer:
                self._mix_ops.append(TransformerBlock(_d_layer, d_model, head))

        for dilation in dilations:
            for _d_layer in d_layer:
                self._mix_ops.append(NextItNet(_d_layer, d_model, dilation))

    def forward(self, x, index):

        return self._mix_ops[index](x)
