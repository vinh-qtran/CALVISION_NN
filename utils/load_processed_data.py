import os

import numpy as np
import awkward as ak

import pandas as pd

reduced_keys = [
    'init_momentum',
    'init_eta',
    'init_phi',

    'hits_sum_energy_all',
    'hits_avg_eta_all',
    'hits_avg_phi_all',

    'hits_sum_energy_cut',
    'hits_avg_eta_cut',
    'hits_avg_phi_cut'
]

full_data_keys = [
    'hits_energy',
    'hits_r',
    'hits_eta',
    'hits_phi',
]


def fix_avg_phi_cut(df):    
    hits_avg_x = []
    hits_avg_y = []

    for i in range(df.shape[0]):
        hits_energy = df['hits_energy'][i]
        hits_r = df['hits_r'][i]
        hits_phi = df['hits_phi'][i]

        hits_avg_x.append(np.sum(hits_energy*hits_r*np.cos(hits_phi))/np.sum(hits_energy))
        hits_avg_y.append(np.sum(hits_energy*hits_r*np.sin(hits_phi))/np.sum(hits_energy))

    hits_avg_phi = np.arctan2(hits_avg_y,hits_avg_x)

    df['hits_avg_phi_cut'] = [row for row in hits_avg_phi]

    print('Fixed hits_avg_phi_cut')

class LoadSCEPCALData:
    def __init__(self,
                 processed_data_dir,
                 reduce=True):
        self.processed_data_dir = processed_data_dir
        self.reduce = reduce

        self._get_all_data()

    def _get_single_df(self,file):
        df = pd.read_hdf(file,key='SCEPCAL')
        #fix_avg_phi_cut(df)
        #df.to_hdf(file,key='SCEPCAL',mode='w')
        return df
    
    def _get_all_data(self):
        dfs = []

        for file in os.scandir(self.processed_data_dir):
            if file.name.endswith('.h5'):
                try:
                    df = self._get_single_df(file)
                    dfs.append(df)
                except:
                    print(f'Error reading {file.name}')
                    continue
        
        combined_df = pd.concat(dfs,ignore_index=True).drop(columns=full_data_keys if self.reduce else None)

        self.data = {
            key : combined_df[key].to_numpy() if key in reduced_keys else np.array([i for i in df[key]],dtype=object) for key in combined_df.keys()
        }

if __name__ == '__main__':
    SCEPCAL_data = LoadSCEPCALData('processed_data')
    print(SCEPCAL_data.data)