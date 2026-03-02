from __future__ import print_function

import argparse
import pdb
import os
import math
import sys
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
from utils.file_utils import save_pkl, load_pkl
from utils.pro_core_utils import train
from utils.pro_utils import get_custom_exp_code
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
from multiprocessing import set_start_method
import os 

def main(args):
    #### Create Results Directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_cindex = []
    all_val_cindex = []
    folds = np.arange(start, end)

    for i in folds:
        print(i)
        start = timer()
        seed_torch(args.seed)
        results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))
        if os.path.isfile(results_pkl_path):
            print("Skipping Split %d" % i)
            continue


        train_dataset, val_dataset,test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        datasets = (train_dataset, val_dataset,test_dataset)
        print(len(train_dataset), len(val_dataset), len(test_dataset))

        if args.task_type == 'survival':
            results, test_cindex, val_cindex = train(datasets, i, args)

        all_test_cindex.append(test_cindex)
        all_val_cindex.append(val_cindex)

        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_cindex, 
        'val_auc': all_val_cindex})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=r'',
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')


parser.add_argument('--seed', type=int, default=2201, 
                    help='random seed for reproducible experiment (default: 1)')

parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', 
                    default=r'',
                    help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, 
                    default=r'', 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')



parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')

parser.add_argument('--model_type', type=str, choices=['stable',], default='stable', 
                    help='type of model (default: stable)')

parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite',       action='store_true', default=True, help='Whether or not to overwrite experiments (if already ran)')
parser.add_argument('--testing',         action='store_true', default=False, help='debugging tool')
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')

parser.add_argument('--gc',              type=int, default=4, help='Gradient Accumulation Step.')
parser.add_argument('--input_dim', type=int, default=1024, help='input dim (default: 1024)')

parser.add_argument('--depth', type=int, default=2, help='depth of stablemil (default: 2)')
parser.add_argument('--num_heads', type=int, default=8, help='heads for MHSA')
parser.add_argument('--k_neighbors', type=int, default= 8, help='neighbors for attention mask (default: 8)')
parser.add_argument('--max_dist', type=float, default= 3*np.sqrt(2), help='max distance for attention mask (default: 3*sqrt(2))')
parser.add_argument('--aggregate_num', type=int, default=256, help='aggregate_num for stableMIL (default: 256)')
parser.add_argument('--task', type=str, choices=['subtype','survival'], default='subtype', help='prediction task (default: subtype)')
parser.add_argument('--ref_size', type=int, default=512, help='referrence patch size ,used for calculate distance (default: 512)')
parser.add_argument('--drop', type=float, default=0.0, help='drop_rate_for_attention(defalut:0.25)')
parser.add_argument('--drop_path', type=float, default=0., help='drop rate for path (default: 0.)')
parser.add_argument('--ratio',type=int ,default=2)
parser.add_argument('--exp_code', default=r'stable',type=str,help='experiment code for saving results')
parser.add_argument('--csv_path',type = str, default=r'',help='csv_path 2 your label')

args = parser.parse_args()
print("Experiment Name:", args.exp_code)
print("agg num :", args.aggregate_num)
print("k num:", args.k_neighbors)
print("ratio:", args.ratio)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_custom_exp_code(args)
args.results_dir = os.path.join(args.results_dir , args.exp_code)

os.makedirs(args.results_dir , exist_ok=True)

print("Experiment Name:", args.exp_code)

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt}
print('\nLoad Dataset')

if 'survival' in args.task:
    args.n_classes = 4
    args.num_classes = 4
    
    study = '_'.join(args.task.split('_')[:2])
    if study == 'tcga_NSCLC' or study == 'tcga_kirp':
        combined_study = 'tcga_kidney'
    elif study == 'tcga_NSCLC' or study == 'tcga_COAD':
        combined_study = 'tcga_NSCLC'
    else:
        combined_study = study
    study_dir = '%s_20x_features' % combined_study
    dataset = Generic_MIL_Survival_Dataset(
                                        csv_path = args.csv_path,
                                           mode = args.mode,
                                            patch_size = args.ref_size,
                                           data_dir= os.path.join(args.data_root_dir, 'UNI_20_256'),
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           n_bins=4,
                                           state = 'dss_censored',
                                           label_col = 'dss_time',
                                           ignore=[])
else:
    raise NotImplementedError

if isinstance(dataset, Generic_MIL_Survival_Dataset):
    args.task_type = 'survival'
else:
    raise NotImplementedError

### Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)


if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
    print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
    sys.exit()

### Sets the absolute path of split_dir
# args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)

print("split_dir", args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})
print(args.exp_code)

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
