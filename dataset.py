import copy
import torch
import random
import pandas as pd
import numpy as np
import torch.utils.data as Data


# use 0 to padding


class TrainDataset(Data.Dataset):
    def __init__(self, mask_prob, max_len, path, device):
        self.data = pd.read_csv(path, header=None).values
        self.num_item = self.data.max()
        self.num_user = self.data.shape[0]
        self.mask_token = self.num_item + 1
        self.mask_prob = mask_prob
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = self.data[index, -self.max_len - 3:-3].tolist()
        pos = self.data[index, -self.max_len - 2:-2].tolist()
        pos = pos[-len(seq):]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        padding_len = self.max_len - len(pos)
        pos = [0] * padding_len + pos

        return torch.LongTensor(seq).to(self.device), torch.LongTensor(pos).to(self.device)


class EvalDataset(Data.Dataset):
    def __init__(self, max_len, sample_size, mode, enable_sample, path, device):
        self.data = pd.read_csv(path, header=None).values
        self.num_item = self.data.max()
        self.num_user = self.data.shape[0]
        self.mask_token = self.num_item + 1
        self.max_len = max_len
        self.sample_size = sample_size
        self.mode = mode
        self.enable_sample = enable_sample
        self.device = device

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = self.data[index, :-2] if self.mode == 'val' else self.data[index, :-1]
        pos = self.data[index, -2] if self.mode == 'val' else self.data[index, -1]
        negs = []

        if self.enable_sample:
            seen = set(seq)
            seen.update([pos])
            while len(negs) < self.sample_size:
                candidate = np.random.randint(0, self.num_item) + 1
                while candidate in seen or candidate in negs:
                    candidate = np.random.randint(0, self.num_item) + 1
                negs.append(candidate)

            answers = [pos] + negs
            labels = [1] + [0] * len(negs)

            seq = list(seq)
            seq = seq + [self.mask_token]

            seq = seq[-self.max_len:]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            return torch.LongTensor(seq).to(self.device), torch.LongTensor(answers).to(self.device), torch.LongTensor(
                labels).to(self.device)

        else:
            seq = list(seq)
            seq = seq[-self.max_len:]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            answers = [pos]
            return torch.LongTensor(seq).to(self.device), torch.LongTensor(answers).to(self.device)


class NASDataset(Data.Dataset):
    def __init__(self, data, mask_prob, max_len, num_item, device, mode, times, method):
        self.data = data
        self.num_item = num_item
        self.num_user = self.data.shape[0]
        self.mask_token = self.num_item + 1
        self.mask_prob = mask_prob
        self.max_len = max_len
        self.device = device
        self.mode = mode
        self.times = times
        self.method = method

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = self.data[index, :-2]

        if self.mode == 'val':
            ol_seq = self._random_mask(seq, 1, self.method[0])[0]
            tgt_seq = self._random_mask(seq, 1, self.method[1])[0]

        elif self.mode == 'train':
            ol_seq = self._random_mask(seq, self.times, self.method[0])
            tgt_seq = self._random_mask(seq, self.times, self.method[1])

        else:
            raise NotImplementedError

        return torch.LongTensor(ol_seq).to(self.device), torch.LongTensor(tgt_seq).to(self.device)

    def _random_mask(self, seq, times, method):
        seqs = []
        for i in range(times):
            _seq = []

            if method == 'mask':
                for s in seq:
                    if s != 0:
                        prob = random.random()
                        if prob < self.mask_prob:
                            _seq.append(self.mask_token)
                        else:
                            _seq.append(s)
                    else:
                        _seq.append(s)

            elif method == 'clip':
                _seq = copy.deepcopy(seq)
                index = random.randint(0, self.max_len - 1)
                len_ = min(int(self.mask_prob * self.max_len), self.max_len - index)
                _seq[index:index + len_] = self.mask_token
                _seq = _seq.tolist()

            elif method == 'permute':
                _seq = copy.deepcopy(seq)
                index = random.randint(0, self.max_len - 1)
                len_ = min(int(self.mask_prob * self.max_len), self.max_len - index)
                tmp = _seq[index:index + len_]
                random.shuffle(tmp)
                _seq[index:index + len_] = tmp
                _seq = _seq.tolist()

            _seq = _seq[-self.max_len:]
            mask_len = self.max_len - len(_seq)
            _seq = [0] * mask_len + _seq
            seqs.append(_seq)
        return seqs
