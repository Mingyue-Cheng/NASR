import math
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from joblib import Parallel, delayed


class CE:
    def __init__(self, model):
        self.model = model
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    def compute(self, batch):
        seqs, labels = batch

        outputs = self.model(seqs)  # B * L * N
        outputs = outputs.view(-1, outputs.shape[-1])  # (B*L) * N
        labels = labels.view(-1)

        loss = self.ce(outputs, labels)
        return loss


class BCE:
    def __init__(self, model, neg_samples, num_item, device):
        self.model = model
        self.neg_samples = neg_samples
        self.num_item = num_item
        self.device = device
        self.bce = nn.BCEWithLogitsLoss()

    def generate_item_neg_items_random(self, seq, mask):
        neg_items = []
        items = np.arange(0, self.num_item + 1)
        for i in range(len(seq)):
            _mask = mask[i]
            _mask = (_mask[_mask > 0]).tolist()
            seen = list(set(seq[i].tolist()))
            if self.num_item + 1 in seen:
                seen.remove(self.num_item + 1)
            _items = np.delete(items, seen)
            for j in range(len(_mask)):
                neg_items.append([_mask[j]] + _items[np.random.randint(0, len(_items), self.neg_samples)].tolist())
        return torch.LongTensor(neg_items)

    def compute(self, batch):
        seqs, mask = batch

        neg_items = self.generate_item_neg_items_random(seqs, mask).to(self.device)

        outputs = self.model(seqs)  # B * L * N
        outputs = outputs.view(-1, outputs.shape[-1])  # (B*L) * N
        mask = mask.view(-1)

        preds = outputs[mask > 0]  # num_of_mask * N
        values = preds.gather(1, neg_items)
        labels = torch.Tensor([[1] + [0] * self.neg_samples for i in range(values.shape[0])]).to(self.device)

        loss = self.bce(values, labels)
        return loss


class BPR:
    def __init__(self, model, neg_samples, num_item, device):
        self.model = model
        self.neg_samples = neg_samples
        self.num_item = num_item
        self.device = device
        self.bpr = self.bpr_loss

    def generate_item_pairs_raw_random(self, seq, mask):
        item_pairs = []
        items = np.arange(0, self.num_item + 1)
        for i in range(len(seq)):
            _mask = mask[i]
            _mask = (_mask[_mask > 0]).tolist()
            seen = list(set(seq[i].tolist()))
            if self.num_item + 1 in seen:
                seen.remove(self.num_item + 1)
            _items = np.delete(items, seen)
            for j in range(len(_mask)):
                item_pairs.append([_mask[j]] + _items[np.random.randint(0, len(_items), self.neg_samples)].tolist())
        return torch.LongTensor(item_pairs)


    def compute(self, batch):
        seqs, mask = batch

        item_pairs = self.generate_item_pairs_raw_random(seqs, mask).to(self.device)
        outputs = self.model(seqs)  # B * L * N
        outputs = outputs.view(-1, outputs.shape[-1])  # (B*L) * N
        mask = mask.view(-1)

        preds = outputs[mask > 0]  # num_of_mask * N
        scores = preds.gather(1, item_pairs)
        loss = self.bpr(scores[:, 1:].reshape(-1),
                        scores[:, 0].repeat(self.neg_samples, 1).transpose(0, 1).contiguous().view(-1))

        return loss

    @staticmethod
    def bpr_loss(neg_scores, pos_scores):
        return torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
