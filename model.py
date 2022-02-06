import torch
import torch.nn as nn
import numpy as np
from layer.supernet import Supernet
from layer.cl import CLNeck, CLHead
from layer.TRM import TransformerBlock, PositionalEmbedding
from layer.NextItNet import NextItNet
from torch.nn.init import xavier_normal_, uniform_, constant_


class SiameseSupernet(nn.Module):
    def __init__(self, args):
        super(SiameseSupernet, self).__init__()

        self.start_block = 0
        self.num_block = args.num_block
        self.layers_per_block = args.layers_per_block if isinstance(args.layers_per_block, list) \
            else [args.layers_per_block] * args.num_block
        self.embed_size = args.d_model
        self.num_item = args.num_item + 2  # mask token = num_item + 1
        self.seq_len = args.max_len
        self.paths_per_step = args.paths_per_step

        self.embedding = nn.Embedding(self.num_item, self.embed_size)
        self.dropout = nn.Dropout(0.1)

        supernet_setting = [args.heads, args.dilations, args.d_layer]
        self.num_op = (len(args.heads) + len(args.dilations)) * len(args.d_layer)
        self.backbone = Supernet(self.num_block, self.layers_per_block, self.embed_size, supernet_setting)
        self.necks = nn.ModuleList()
        self.heads = nn.ModuleList()

        for block in range(self.num_block):
            self.necks.append(CLNeck(self.seq_len, self.embed_size, num_layer=2))
            self.heads.append(CLHead(self.embed_size))

        self.best_paths = []

    def forward_train(self, ol_seq, tgt_seq):
        layers = self.layers_per_block[self.start_block]
        path_sampled = [np.random.randint(0, self.num_op, size=layers) for _ in range(self.paths_per_step)]
        with torch.no_grad():
            tgt_rep = 0
            tgt_emb = self.dropout(self.embedding(tgt_seq))
            tgt_emb = tgt_emb.view(tgt_emb.shape[0] * tgt_emb.shape[1], tgt_emb.shape[2], tgt_emb.shape[3])
            if self.start_block > 0:
                for i, best_path in enumerate(self.best_paths):
                    tgt_emb = self.backbone(tgt_emb, i, best_path)
            tgt_emb = tgt_emb.view(-1, self.paths_per_step, self.seq_len, self.embed_size)

            for index, path in enumerate(path_sampled):
                temp_rep = self.necks[self.start_block](self.backbone(
                    tgt_emb[:, index, ...].contiguous(), self.start_block, path
                ))
                tgt_rep += nn.functional.normalize(temp_rep, dim=1)

            tgt_rep /= self.paths_per_step

        ol_rep = []
        ol_emb = self.dropout(self.embedding(ol_seq))
        ol_emb = ol_emb.view(ol_emb.shape[0] * ol_emb.shape[1], ol_emb.shape[2], ol_emb.shape[3])
        if self.start_block > 0:
            for i, best_path in enumerate(self.best_paths):
                ol_emb = self.backbone(ol_emb, i, best_path)
        ol_emb = ol_emb.view(-1, self.paths_per_step, self.seq_len, self.embed_size)

        for index, path in enumerate(path_sampled):
            temp_rep = self.necks[self.start_block](self.backbone(
                ol_emb[:, index, ...].contiguous(), self.start_block, path
            ))
            ol_rep.append(temp_rep)

        loss = 0
        for rep in ol_rep:
            loss += self.heads[self.start_block](rep, tgt_rep)
        return loss

    @torch.no_grad()
    def forward_val(self, ol_seq, tgt_seq):
        layers = self.layers_per_block[self.start_block]
        path_all = self.build_all_path(self.num_op, layers)

        tgt_rep = 0
        tgt_emb = self.dropout(self.embedding(tgt_seq))
        if self.start_block > 0:
            for i, best_path in enumerate(self.best_paths):
                tgt_emb = self.backbone(tgt_emb, i, best_path)

        for path in path_all:
            temp_rep = self.necks[self.start_block](self.backbone(
                tgt_emb, self.start_block, path
            ))
            tgt_rep += nn.functional.normalize(temp_rep, dim=1)

        tgt_rep /= len(path_all)

        ol_rep = []
        ol_emb = self.dropout(self.embedding(ol_seq))
        if self.start_block > 0:
            for i, best_path in enumerate(self.best_paths):
                ol_emb = self.backbone(ol_emb, i, best_path)

        for path in path_all:
            temp_rep = self.necks[self.start_block](self.backbone(
                tgt_emb, self.start_block, path
            ))
            ol_rep.append(temp_rep)

        loss = {}
        for path, rep in zip(path_all, ol_rep):
            path_encoding = ''
            for i in path:
                path_encoding += str(i)
            loss[path_encoding] = self.heads[self.start_block](rep, tgt_rep)
        return loss

    def forward(self, ol_seq, tgt_seq, mode='train'):
        if mode == 'train':
            return self.forward_train(ol_seq, tgt_seq)
        elif mode == 'val':
            return self.forward_val(ol_seq, tgt_seq)
        else:
            raise NotImplementedError

    @staticmethod
    def build_all_path(num_op, layers):
        assert 1 <= layers <= 8
        paths = [[], [], [], [], [], [], [], []]
        for i0 in range(num_op):
            paths[0].append([i0])
            for i1 in range(num_op):
                paths[1].append([i0, i1])
                for i2 in range(num_op):
                    paths[2].append([i0, i1, i2])
                    for i3 in range(num_op):
                        paths[3].append([i0, i1, i2, i3])
                        for i4 in range(num_op):
                            paths[4].append([i0, i1, i2, i3, i4])
                            for i5 in range(num_op):
                                paths[5].append([i0, i1, i2, i3, i4, i5])
                                for i6 in range(num_op):
                                    paths[6].append([i0, i1, i2, i3, i4, i5, i6])
                                    for i7 in range(num_op):
                                        paths[7].append([i0, i1, i2, i3, i4, i5, i6, i7])
        return paths[layers - 1]


class NASModel(nn.Module):
    def __init__(self, args, encoding, draw = False):
        super(NASModel, self).__init__()
        self.args = args
        self.draw = draw
        self.device = args.device
        self.encoding = encoding
        self.embed_size = args.d_model
        self.num_item = args.num_item + 2  # mask token = num_item + 1
        self.max_len = args.max_len
        self.attention_mask = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.bool)).to(self.device)

        self.embedding = nn.Embedding(self.num_item, self.embed_size)
        self.position = PositionalEmbedding(self.max_len, self.embed_size)
        self.final_layer = nn.Linear(self.embed_size, self.num_item - 1)
        self._build_layers()

        self.apply(self._init_weights)

    def _build_layers(self):
        self._layers = nn.ModuleList()

        heads = self.args.heads
        dilations = self.args.dilations
        d_layer = self.args.d_layer

        for block_code in self.encoding:
            for layer_code in block_code:
                if layer_code + 1 > len(heads) * len(d_layer):
                    layer_code -= len(heads) * len(d_layer)
                    self._layers.append(
                        NextItNet(d_layer[layer_code % len(d_layer)], self.embed_size, dilations[layer_code // len(d_layer)], dropout=None))
                else:
                    self._layers.append(
                        TransformerBlock(d_layer[layer_code % len(d_layer)], self.embed_size, heads[layer_code // len(d_layer)]))

        print(self._layers)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / self.num_item)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def forward(self, x):
        mask = self.attention_mask
        x = self.embedding(x) + self.position(x)

        for layer in self._layers:
            x = layer(x, mask)

        if self.draw:
            return x

        pred = self.final_layer(x)
        return pred
