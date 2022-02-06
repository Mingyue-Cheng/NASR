import time
import numpy as np
import torch
import torch.utils.data as Data
from dataset import TrainDataset, EvalDataset, NASDataset
from args import args, DATA
from train import NASTrainer, ModelTrainer
from model import SiameseSupernet, NASModel


def nas_train():
    data = DATA
    # using 95% of the user sequence to train and the rest to evaluate
    np.random.shuffle(data)
    split = int(np.floor(data.shape[0] * 0.95))
    data_train = data[:split]
    data_val = data[split:]

    method = [args.teacher_augmentation, args.student_augmentation]

    nas_train_dataset = NASDataset(data_train, args.mask_prob, args.max_len, args.num_item, args.device, 'train', args.paths_per_step, method)
    nas_val_dataset = NASDataset(data_val, args.mask_prob, args.max_len, args.num_item, args.device, 'val', 0, method)
    nas_train_dataloader = Data.DataLoader(nas_train_dataset, args.nas_batch_size, shuffle=True)
    nas_val_dataloader = Data.DataLoader(nas_val_dataset, args.nas_batch_size, shuffle=True)
    print('****dataloader ready****')

    nas_model = SiameseSupernet(args)
    print('****model initial ends****')

    trainer = NASTrainer(args, nas_model, nas_train_dataloader, nas_val_dataloader)
    print('****trainer ready****')

    start = time.perf_counter()
    encoding = trainer.search()
    end = time.perf_counter()
    print(encoding, end - start)
    return encoding


def model_train(encoding):
    train_dataset = TrainDataset(args.mask_prob, args.max_len, args.data_path, device=args.device)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    val_dataset = EvalDataset(args.max_len, args.sample_size, mode='val', enable_sample=args.enable_sample,
                              path=args.data_path, device=args.device)
    val_loader = Data.DataLoader(val_dataset, batch_size=args.val_batch_size)

    test_dataset = EvalDataset(args.max_len, args.sample_size, mode='test', enable_sample=args.enable_sample,
                               path=args.data_path, device=args.device)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    print('dataset initial ends')

    model = NASModel(args, encoding)
    print('model initial ends')

    trainer = ModelTrainer(args, model, train_loader, val_loader, test_loader)
    print('train process ready')

    trainer.train()


if __name__ == '__main__':
    # NAS stage, get the encoding representation of the best architecture
    encoding = nas_train()
    # use encoding to construct a hybrid network, and evaluate its performance
    model_train(encoding)
