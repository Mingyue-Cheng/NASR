import torch
import time
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from loss import CE, BCE, BPR


class NASTrainer():
    def __init__(self, args, nas_model, train_loader, val_loader):
        self.args = args
        self.device = args.device
        print(self.device)
        self.model = nas_model.to(torch.device(self.device))

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epoch_per_stage = args.epoch_per_stage
        self.lr = args.nas_lr
        self.num_block = args.num_block

    def search(self):
        for i in range(self.num_block):
            self.train_a_stage()
            best_path = self.val_stage()
            print(best_path)
            self.model.best_paths.append(best_path)
            self.model.start_block += 1

        return self.model.best_paths

    def train_a_stage(self):
        self.optim = torch.optim.AdamW(self.model.parameters(), self.lr)
        for epoch in range(self.epoch_per_stage):
            tqdm_loader = tqdm(self.train_loader)
            loss_sum = 0
            loss_list = []
            tmp = []
            for index, batch in enumerate(tqdm_loader):
                ol_seq, tgt_seq = batch
                loss = self.model(ol_seq, tgt_seq, 'train')
                loss_sum += loss.item()
                tmp.append(loss.item())
                if (index + 1) % 50 == 0:
                    loss_list.append(np.mean(tmp))
                    tmp = []
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
            print('stage{},epoch{},loss{}'.format(self.model.start_block, epoch, loss_sum))
        return 0

    def val_stage(self):
        tqdm_loader = tqdm(self.val_loader)
        loss = {}
        for _, batch in enumerate(tqdm_loader):
            ol_seq, tgt_seq = batch
            loss_batch = self.model(ol_seq, tgt_seq, 'val')
            for k, v in loss_batch.items():
                if loss.__contains__(k):
                    loss[k] += v
                else:
                    loss[k] = v
        loss = sorted(loss.items(), key=lambda x: x[1])
        best_path_encoding = loss[0][0]
        best_path = [int(i) for i in best_path_encoding]
        return best_path


class ModelTrainer():
    def __init__(self, args, model, train_loader, val_loader, test_loader):
        self.args = args
        self.device = args.device
        print(self.device)
        self.model = model.to(torch.device(self.device))

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps


        self.loss = args.loss_type
        self.neg_samples = args.neg_samples
        self.enable_sample = args.enable_sample
        if self.loss == 'ce':
            self.cr = CE(self.model)
        elif self.loss == 'bce':
            self.cr = BCE(self.model, self.neg_samples, self.args.num_item, self.device)
        elif self.loss == 'bpr':
            self.cr = BPR(self.model, self.neg_samples, self.args.num_item, self.device)


        self.num_epoch = args.num_epoch
        self.metric_ks = args.metric_ks
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        if self.num_epoch:
            self.result_file = open(self.save_path + '/result.txt', 'w')
            self.result_file.close()

        self.step = 0
        self.metric = args.best_metric
        self.best_metric = -1e9

        self.labels = torch.zeros(512, self.args.num_item + 1).to(self.device)

    def train(self):
        # BERT training
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=True)
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch(pred=True)
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            print('Searched Model train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))
            print('Searched Model train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost),
                  file=self.result_file)
            self.result_file.close()

    def _train_one_epoch(self, pred):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)

        loss_sum = 0
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()
            loss = self.cr.compute(batch)

            loss_sum += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            self.step += 1
            if self.step % self.lr_decay_steps == 0:
                self.scheduler.step()
            if self.step % self.eval_per_steps == 0:
                self.sample_time = 0
                metric = {}
                for mode in ['val', 'test']:
                    metric[mode] = self.eval_model(mode)
                print(metric)
                self.result_file = open(self.save_path + '/result.txt', 'a+')
                print('step{0}'.format(self.step), file=self.result_file)
                print(metric, file=self.result_file)
                self.result_file.close()
                if metric['test'][self.metric] > self.best_metric:
                    torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
                    self.result_file = open(self.save_path + '/result.txt', 'a+')
                    print('saving model of step{0}'.format(self.step), file=self.result_file)
                    self.result_file.close()
                    self.best_metric = metric['test'][self.metric]
                self.model.train()

        return loss_sum / idx, time.perf_counter() - t0

    def eval_model(self, mode):
        self.model.eval()
        tqdm_data_loader = tqdm(self.val_loader) if mode == 'val' else tqdm(self.test_loader)
        metrics = {}

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_data_loader):
                batch = [x.to(self.device) for x in batch]
                metrics_batch = self.compute_metrics(batch)

                for k, v in metrics_batch.items():
                    if not metrics.__contains__(k):
                        metrics[k] = v
                    else:
                        metrics[k] += v

        for k, v in metrics.items():
            metrics[k] = v / (idx + 1)
        return metrics

    def compute_metrics(self, batch):
        if self.enable_sample:
            seqs, answers, labels = batch
            scores = self.model(seqs)
            scores = scores[:, -1, :]  # the prediction score of the last token in seq, B * N
            scores = scores.gather(1, answers)  # only consider positive and negative items' score

        else:
            seqs, answers = batch
            scores = self.model(seqs)
            scores = scores[:, -1, :]
            row = []
            col = []
            seqs = seqs.tolist()
            answers = answers.tolist()
            for i in range(len(answers)):
                seq = list(set(seqs[i] + answers[i]))
                seq.remove(answers[i][0])
                if self.args.num_item + 1 in seq:
                    seq.remove(self.args.num_item + 1)
                row += [i] * len(seq)
                col += seq
                self.labels[i][answers[i]] = 1
            scores[row, col] = -1e9
        metrics = self.recalls_and_ndcgs_for_ks(scores, self.labels[:len(seqs)], self.metric_ks)
        self.labels[self.labels == 1] = 0
        return metrics

    @staticmethod
    def recalls_and_ndcgs_for_ks(scores, labels, ks):
        metrics = {}

        answer_count = labels.sum(1)

        labels_float = labels.float()
        rank = (-scores).argsort(dim=1)
        cut = rank
        for k in sorted(ks, reverse=True):
            cut = cut[:, :k]
            hits = labels_float.gather(1, cut)
            metrics['Recall@%d' % k] = \
                (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device),
                                         labels.sum(1).float())).mean().cpu().item()

            position = torch.arange(2, 2 + k)
            weights = 1 / torch.log2(position.float())
            dcg = (hits * weights.to(hits.device)).sum(1)
            idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
            ndcg = (dcg / idcg).mean()
            metrics['NDCG@%d' % k] = ndcg.cpu().item()

        return metrics
