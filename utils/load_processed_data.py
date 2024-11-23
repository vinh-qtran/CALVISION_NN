import os

import numpy as np
import awkward as ak

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset,DataLoader

import pandas as pd

class LoadSCEPCALData:
    initial_particle_info = [
        'init_momentum',
        'init_eta',
        'init_phi',
    ]

    reduced_hits_info = [
        'hits_sum_energy_all',
        'hits_avg_eta_all',
        'hits_avg_phi_all',
        'hits_sum_energy_cut',
        'hits_avg_eta_cut',
        'hits_avg_phi_cut'
    ]

    full_hits_info = [
        'hits_energy',
        'hits_r',
        'hits_eta',
        'hits_phi',
    ]
    
    def __init__(self,
                 processed_data_dir,
                 read_full_hits_info=True,
                 recalculate_reduced_hits_info=True,
                 max_hit_count=None):
        
        # Set the processed data directory
        self.processed_data_dir = processed_data_dir

        # Set the flags for reading the full hits info, recalculating the reduced info, and dropping the uncut reduced info
        self.read_full_hits_info = read_full_hits_info
        self.recalculate_reduced_hits_info = recalculate_reduced_hits_info

        if self.recalculate_reduced_hits_info:
            assert self.read_full_hits_info, 'cannot recalculate reduced hits info without reading full hits info'

        # Get the list of fields to drop from the dataframe
        self.drop_fields = self._get_drop_fields_list()

        # Load the data
        self.max_hit_count = max_hit_count
        self._get_data()


    def _get_drop_fields_list(self):
        '''
        Get the list of fields to drop from the dataframe
        '''
        drop_fields = []

        if not self.read_full_hits_info:
            drop_fields += self.full_hits_info

        if self.recalculate_reduced_hits_info:
            drop_fields += self.reduced_hits_info

        return drop_fields

    def _get_single_df(self,file):
        '''
        Get a single dataframe from a file
        '''
        df = pd.read_hdf(file,key='SCEPCAL')
        return df
    
    def _get_combined_df(self):
        """
        Combines all data from the processed data directory into a single dictionary.
        """
        dfs = []

        for file in os.scandir(self.processed_data_dir):
            if file.name.endswith('.h5'):
                 try:
                    df = self._get_single_df(file.path)
                    dfs.append(df)
                 except:
                     print(f'Error reading {file.name}')
                     continue
        
        return pd.concat(dfs,ignore_index=True).drop(self.drop_fields,axis=1)
    
    def _get_initial_particle_data(self,df):
        """
        Get the initial particle info from the combined dataframe.
        """
        for field in self.initial_particle_info:
            self.data[field] = df[field].to_numpy()

    def _get_full_hits_data(self,df):
        """
        Get the full hits info from the combined dataframe.
        """
        sim_max_hit_count = df['hits_energy'].apply(len).max()
        self.max_hit_count = sim_max_hit_count if self.max_hit_count is None else self.max_hit_count

        for field in self.full_hits_info:
            self.data[field] = np.concatenate([np.pad(row, (0,sim_max_hit_count-len(row)), mode='constant', constant_values=0).reshape(1,-1) for row in df[field]],axis=0)

        for field in self.full_hits_info:
            self.data[field] = self.data[field][:,:self.max_hit_count]

    def _get_reduced_hits_data(self,df):
        """
        Get the reduced hits info from the combined dataframe.
        """
        if not self.recalculate_reduced_hits_info:
            for field in self.reduced_hits_info:
                self.data[field] = df[field].to_numpy()
    
    def _get_data(self):
        """
        Get the data from the combined dataframe.
        """
        df = self._get_combined_df()

        self.data = {}

        self._get_initial_particle_data(df)
        self._get_full_hits_data(df)
        self._get_reduced_hits_data(df)


def get_n_moment(data,n,weights=None,axis=1):
    center = np.average(data, weights=weights, axis=axis)

    if n == 1:
        return center
    
    return np.average(
        (data-center.reshape(-1,1))**n, weights=weights, axis=axis
    )

def get_reduced_statistics(data,weights=None,axis=1,reshape=False):
    mean = get_n_moment(data,1,weights,axis)
    variance = get_n_moment(data,2,weights,axis)
    skewness = get_n_moment(data,3,weights,axis)/(variance**(3/2))
    kurtosis = get_n_moment(data,4,weights,axis)/(variance**2)

    statistics = [mean,variance,skewness,kurtosis]
    if reshape:
        return [stats.reshape(-1,1) for stats in statistics]
    return statistics

class SCEPCALDataLoader:
    def __init__(self,
                 data,
                 input_target_transform=None,
                 train_val_ratio = (0.8,0.2),
                 batch_size=4096,
                 num_workers=4,
                 shuffle=True,
                 seed=42):
        
        self.data = data

        self.train_val_ratio = train_val_ratio

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        self.seed = seed
        if self.seed:
            torch.random.manual_seed(self.seed)

        self.input_target_transform = input_target_transform if input_target_transform is not None else self._simple_input_target_transform
        self.X, self.Y = self.input_target_transform(self.data)

        self.train_loader, self.val_loader = self._get_DataLoaders()

    def _simple_input_target_transform(self,data):
        init_momentum = data['init_momentum']

        hits_energy = data['hits_energy']
        hits_eta = data['hits_eta']
        hits_phi = data['hits_phi']

        X = np.concatenate([
            *get_reduced_statistics(hits_energy,reshape=True),
            *get_reduced_statistics(hits_eta,weights=hits_energy,reshape=True),
            *get_reduced_statistics(hits_phi,weights=hits_energy,reshape=True),
        ],axis=1)

        Y = init_momentum / np.sum(hits_energy,axis=1)

        return X,Y
    
    def _get_DataLoaders(self):
        N_sample = self.X.shape[0]

        N_train = int(N_sample*self.train_val_ratio[0])
        N_val = int(N_sample*self.train_val_ratio[1])

        all_dataset = TensorDataset(torch.from_numpy(self.X),torch.from_numpy(self.Y))

        train_dataset, val_dataset = torch.utils.data.random_split(
            all_dataset,
            [N_train,N_val]
        )

        train_loader = DataLoader(train_dataset,
                                  num_workers=self.num_workers,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle)
        val_loader = DataLoader(val_dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle)
        
        return train_loader,val_loader
    

if __name__ == '__main__':
    def inspect_data(data):
        print(' Shape:',data.shape)
        print(' Min:',np.min(data))
        print(' Max:',np.max(data))
        print(' Mean:',np.mean(data))
        print(' Std:',np.std(data))

    def inspect_data_loader(data_loader):
        print(' N_batches:',len(data_loader))
        print(' Batch Size:',data_loader.batch_size)
        print(' X:')
        for i, (X,Y) in enumerate(data_loader):
            if i == 0:
                inspect_data(X.numpy())
        print(' Y:')
        for i, (X,Y) in enumerate(data_loader):
            if i == 0:
                inspect_data(Y.numpy())

    def check_SCEPCAL_data_reader():
        SCEPCAL_data = LoadSCEPCALData('../../processed_data',read_full_hits_info=True,recalculate_reduced_hits_info=True,max_hit_count=None)

        for field in SCEPCAL_data.data:
            print(field)
            inspect_data(SCEPCAL_data.data[field])

    def check_SCEPCAL_data_loader():
        SCEPCAL_data = LoadSCEPCALData('../../processed_data',read_full_hits_info=True,recalculate_reduced_hits_info=True,max_hit_count=None)
        SCEPCAL_data_loader = SCEPCALDataLoader(SCEPCAL_data.data)

        print('X:')
        inspect_data(SCEPCAL_data_loader.X)
        print('Y:')
        inspect_data(SCEPCAL_data_loader.Y)

        print('Train Loader:')
        inspect_data_loader(SCEPCAL_data_loader.train_loader)
        print('Val Loader:')
        inspect_data_loader(SCEPCAL_data_loader.val_loader)

    #check_SCEPCAL_data_reader()
    #check_SCEPCAL_data_loader()