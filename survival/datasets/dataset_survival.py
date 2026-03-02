from __future__ import print_function, division
from ast import Not
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
import sys 
from torch.utils.data import Dataset

from utils.pro_utils import generate_split, nth

class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv', mode = 'path',need_id = False,
        state='censorship',shuffle = False, patch_size=512,seed = 7, print_info = True, n_bins = 4, ignore=[],
        patient_strat=False, label_col = None, filter_dict = {}, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.state = state
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None
        self.patch_size = patch_size
        print(self.patch_size)

        if csv_path.endswith('csv'):
            
            slide_data = pd.read_csv(csv_path)
        else :
            slide_data = pd.read_excel(csv_path)

        if 'case_id' not in slide_data:
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        if not label_col:
            label_col = 'survival'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        slide_data = slide_data.astype({self.state:int , label_col:float,'slide_id':str})
        patients_df  = slide_data.copy()

        uncensored_df = patients_df[patients_df[self.state] == 0]
        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id'].drop_duplicates():
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = list(slide_ids)
            patient_dict.update({patient:slide_ids})

        self.patient_dict = patient_dict
    
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            bool_state = slide_data.loc[i, self.state]
            key = (key, int(bool_state))
            print(bool_state)
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes=len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        self.slide_data = slide_data
        self.metadata = slide_data.columns[:11]
        self.mode = mode
        self.need_id = need_id
        self.cls_ids_prep()

        if print_info:
            self.summarize()


    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]


    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}


    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))


    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())             
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode,need_id=self.need_id,  data_dir=self.data_dir, label_col=self.label_col,state = self.state,patient_dict=self.patient_dict, num_classes=self.num_classes,patch_size=self.patch_size)
        else:
            split = None
            
        return split


    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path,dtype=str)

            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            test_split = self.get_split_from_df(all_splits=all_splits, split_key='test')
        return train_split, val_split,test_split


    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, mode: str='path',need_id = False,**kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.need_id = need_id
        self.use_h5 = True


    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]

        c = self.slide_data[self.state][idx]
        slide_id = self.patient_dict[case_id]
        data_dir = self.data_dir
        
        if not self.use_h5:
            raise NotImplementedError('Not implemented')
        
        else:   
            full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id[0]))
            with h5py.File(full_path,'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:] //self.patch_size
                features = torch.from_numpy(features)

            return (features,  coords , label ,event_time ,c , slide_id)


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, mode, need_id = False,data_dir=None, label_col=None, state = 'censorship',patient_dict=None, num_classes=2,patch_size=1024):
        self.use_h5 =  True
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.need_id = need_id
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.state = state
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        self.patch_size = patch_size
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]


        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

    def __len__(self):
        return len(self.slide_data)
