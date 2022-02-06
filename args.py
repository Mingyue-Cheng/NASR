import argparse
import time
import os
import json
import pandas as pd

parser = argparse.ArgumentParser()
# dataset and dataloader args
parser.add_argument('--data_path', type=str, default=r'data\movielens30_processed.csv',
                    help='the data input path, should be a csv file and the user sequence should be padded by 0')
parser.add_argument('--save_path', type=str, default='test',
                    help='the output path, which stores results, args and model')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_len', type=int, default=30,
                    help='the max length of user sequence')
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--val_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--nas_batch_size', type=int, default=64,
                    help='batch size in NAS training stage')
parser.add_argument('--enable_sample', type=int, default=0,
                    help='0 for not using sample strategy in model evaluation and 1 for opposite')
parser.add_argument('--sample_size', type=int, default=100,
                    help='sample size in sampled evaluation')

# model args
parser.add_argument('--d_model', type=int, default=64,
                    help='embedding size')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--eval_per_steps', type=int, default=6000)
parser.add_argument('--enable_res_parameter', type=int, default=1,
                    help='whether use linear residual in each building block')
parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'bce', 'bpr'])
parser.add_argument('--neg_samples', type=int, default=100,
                    help='negative samples for bce and bpr loss')

# nas args
parser.add_argument('--mask_prob', type=float, default=0.25,
                    help='percentage of original sequence modifications')
parser.add_argument('--heads', type=list, default=[4, 8],
                    help='the head param in multi-head attention in Transformer')
parser.add_argument('--dilations', type=list, default=[[1, 4], [1, 2]],
                    help='the dilation param in Nextitnet')
parser.add_argument('--d_layer', type=list, default=[64],
                    help='embedding size of the candidate operations to perform on, in our work is set to the same as d_model')
parser.add_argument('--num_block', type=int, default=4,
                    help='block num in block-wise NAS')
parser.add_argument('--layers_per_block', type=int or list, default=4,
                    help='if set as an integer, the param will be repeated to a list matching number of blocks, '
                         'else the list length should be the same with number of blocks')
parser.add_argument('--paths_per_step', type=int, default=4,
                    help='paths randomly sampled during a training step in NAS')
parser.add_argument('--epoch_per_stage', type=int, default=1,
                    help='how many epochs will be trained for a block before it is ranked')
parser.add_argument('--nas_lr', type=float, default=0.001)
parser.add_argument('--teacher_augmentation', type=str, default='mask', choices=['mask', 'clip', 'permute'])
parser.add_argument('--student_augmentation', type=str, default='mask', choices=['mask', 'clip', 'permute'])

# train args
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_rate', type=float, default=0.99)
parser.add_argument('--lr_decay_steps', type=int, default=1000)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--num_epoch', type=int, default=40)
parser.add_argument('--metric_ks', type=list, default=[10, 20])
parser.add_argument('--best_metric', type=str, default='NDCG@20')

args = parser.parse_args()
# other args

DATA = pd.read_csv(args.data_path, header=None).values
num_item = DATA.max()
args.num_item = int(num_item)

if args.save_path == 'None':
    loss_str = args.loss_type
    path_str = 'test'
    args.save_path = path_str
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()
