import os

import numpy as np
import awkward as ak

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

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
    ]

    uncut_reduced_hits_info = [
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
                 drop_uncut_reduced_hits_info=True,
                 max_hit_count=None):
        
        # Set the processed data directory
        self.processed_data_dir = processed_data_dir

        # Set the flags for reading the full hits info, recalculating the reduced info, and dropping the uncut reduced info
        self.read_full_hits_info = read_full_hits_info
        self.recalculate_reduced_hits_info = recalculate_reduced_hits_info
        self.drop_uncut_reduced_hits_info = drop_uncut_reduced_hits_info

        if self.recalculate_reduced_hits_info:
            assert self.read_full_hits_info, 'cannot recalculate reduced hits info without reading full hits info'

        # Get the list of fields to drop from the dataframe
        self.drop_fields = self._get_drop_fields_list()

        # Load the data
        self.max_hit_count = max_hit_count
        self._get_data_from_df()


    def _get_drop_fields_list(self):
        '''
        Get the list of fields to drop from the dataframe
        '''
        drop_fields = []

        if not self.read_full_hits_info:
            drop_fields += self.full_hits_info

        if self.recalculate_reduced_hits_info:
            drop_fields += self.reduced_hits_info

        if self.drop_uncut_reduced_hits_info:
            drop_fields += self.uncut_reduced_hits_info

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
    
    
    def _get_full_hits_info(self,df):
        """
        Get the full hits info from the combined dataframe.
        """
        if not self.max_hit_count:
            self.max_hit_count = df['hits_energy'].apply(len).max()

        full_hits_data = {}
        for field in self.full_hits_info:
            full_hits_data[field] = np.concatenate([np.pad(row, (0,self.max_hit_count-len(row)), mode='constant', constant_values=0).reshape(1,-1) for row in df[field]],axis=0)

        return full_hits_data
    
    def _get_data_from_df(self):
        """
        Get the data from the combined dataframe.
        """
        df = self._get_combined_df()
        
        self.initial_particle_data = {
            field : df[field].to_numpy() for field in self.initial_particle_info
        }

        self.full_hits_data = self._get_full_hits_info(df)

        if not self.recalculate_reduced_hits_info:
            self.reduced_hits_data = {
                field : df[field].to_numpy() for field in self.reduced_hits_info
            }

        if not self.drop_uncut_reduced_hits_info:
            self.uncut_reduced_hits_data = {
                field : df[field].to_numpy() for field in self.uncut_reduced_hits_info
            }


class SCEPCALReducedDataLoader:
    def __init__(self,
                 data,
                 input_func=None,
                 target_func=None,
                 batch_size=4096,
                 num_workers=4,
                 shuffle=True,
                 seed=42):
        
        self.data = data

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        self.seed = seed
        torch.random.manual_seed(seed)

    def _simple_input_func(self,data):
        pass

if __name__ == '__main__':
    def inspect_data(data):
        print(' Shape:',data.shape)
        print(' Min:',np.min(data))
        print(' Max:',np.max(data))
        print(' Mean:',np.mean(data))
        print(' Std:',np.std(data))

    def check_SCEPCAL_data_reader():
        SCEPCAL_data = LoadSCEPCALData('../../processed_data',read_full_hits_info=True,recalculate_reduced_hits_info=False,drop_uncut_reduced_hits_info=False)

        for field in SCEPCAL_data.initial_particle_info:
            print(field)
            inspect_data(SCEPCAL_data.initial_particle_data[field])
        print()

        for field in SCEPCAL_data.full_hits_info:
            print(field)
            inspect_data(SCEPCAL_data.full_hits_data[field])
        print()

        if not SCEPCAL_data.recalculate_reduced_hits_info:
            for field in SCEPCAL_data.reduced_hits_info:
                print(field)
                inspect_data(SCEPCAL_data.reduced_hits_data[field])
            print()

        if not SCEPCAL_data.drop_uncut_reduced_hits_info:
            for field in SCEPCAL_data.uncut_reduced_hits_info:
                print(field)
                inspect_data(SCEPCAL_data.uncut_reduced_hits_data[field])

    check_SCEPCAL_data_reader()