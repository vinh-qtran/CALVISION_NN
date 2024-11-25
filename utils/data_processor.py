import os

import uproot
import numpy as np
import awkward as ak

from tqdm import tqdm

import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

class DataProcessor:
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

    def __init__(self, file, outpath, event_per_loop=100, energy_cut=2e-4):
        self.file = file
        self.outpath = outpath

        self.event_per_loop = event_per_loop
        self.energy_cut = energy_cut

        self.data = {}

        self._get_all_init_particle_data()
        print('Initial particles data loaded.')

        self._reformat_all_hits_data()
        print('Hits data loaded.')

        self._save_data()
        print('Data saved.')

    def _get_single_loop_init_particle_data(self, entry_start, entry_stop):
        data = self.file['MCParticles'].array(entry_start = entry_start, entry_stop = entry_stop)
        init_particles = data[data['MCParticles.generatorStatus'] == 1]

        init_px = np.array(init_particles['MCParticles.momentum.x']).flatten()
        init_py = np.array(init_particles['MCParticles.momentum.y']).flatten()
        init_pz = np.array(init_particles['MCParticles.momentum.z']).flatten()

        init_momentum = np.sqrt(init_px**2 + init_py**2 + init_pz**2)
        init_eta = np.arctanh(init_pz/init_momentum)
        init_phi = np.arctan2(init_py, init_px)

        return init_momentum, init_eta, init_phi
    
    def _get_all_init_particle_data(self):
        init_momentum = []
        init_eta = []
        init_phi = []

        for entry_start in tqdm(range(0,self.file.num_entries,self.event_per_loop)):
            entry_stop = entry_start + self.event_per_loop
            p, eta, phi = self._get_single_loop_init_particle_data(entry_start,min(entry_stop,self.file.num_entries))

            init_momentum.append(p)
            init_eta.append(eta)
            init_phi.append(phi)

        self.data['init_momentum'] = np.concatenate(init_momentum,axis=0)
        self.data['init_eta'] = np.concatenate(init_eta,axis=0)
        self.data['init_phi'] = np.concatenate(init_phi,axis=0)
    
    def _get_single_loop_hits_data(self, entry_start, entry_stop):
        hits = self.file['SCEPCAL_BE_readout'].array(entry_start=entry_start, entry_stop=entry_stop)

        hits_energy = ak.Array(hits['SCEPCAL_BE_readout.energy'])

        hits_x = ak.Array(hits['SCEPCAL_BE_readout.position.x'])
        hits_y = ak.Array(hits['SCEPCAL_BE_readout.position.y'])
        hits_z = ak.Array(hits['SCEPCAL_BE_readout.position.z'])

        hits_r = np.sqrt(hits_x**2 + hits_y**2 + hits_z**2)
        hits_eta = np.arctanh(hits_z/hits_r)
        hits_phi = np.arctan2(hits_y, hits_x)

        return hits_energy, hits_r, hits_eta, hits_phi
    
    def _get_all_hits_data(self):
        hits_energy = []
        hits_r = []
        hits_eta = []
        hits_phi = []

        for entry_start in tqdm(range(0,self.file.num_entries,self.event_per_loop)):
            entry_stop = entry_start + self.event_per_loop
            energy, r, eta, phi = self._get_single_loop_hits_data(entry_start,min(entry_stop,self.file.num_entries))

            hits_energy.append(energy)
            hits_r.append(r)
            hits_eta.append(eta)
            hits_phi.append(phi)

        hits_energy = ak.concatenate(hits_energy,axis=0)
        hits_r = ak.concatenate(hits_r,axis=0)
        hits_eta = ak.concatenate(hits_eta,axis=0)
        hits_phi = ak.concatenate(hits_phi,axis=0)

        return hits_energy, hits_r, hits_eta, hits_phi
    
    def _calculate_reduced_hits_data(self, hits_energy, hits_r, hits_eta, hits_phi):
        hits_sum_energy = ak.sum(hits_energy,axis=1).to_numpy()
        hits_avg_eta = ak.sum(hits_eta * hits_energy,axis=1).to_numpy() / hits_sum_energy

        hits_avg_x = ak.sum(hits_r * np.cos(hits_phi) * hits_energy,axis=1).to_numpy() / hits_sum_energy
        hits_avg_y = ak.sum(hits_r * np.sin(hits_phi) * hits_energy,axis=1).to_numpy() / hits_sum_energy

        hits_avg_phi = np.arctan2(hits_avg_y, hits_avg_x)

        return hits_sum_energy, hits_avg_eta, hits_avg_phi

    def _reformat_all_hits_data(self):
        hits_energy, hits_r, hits_eta, hits_phi = self._get_all_hits_data()
        self.data['hits_sum_energy_all'], self.data['hits_avg_eta_all'], self.data['hits_avg_phi_all'] = self._calculate_reduced_hits_data(hits_energy, hits_r, hits_eta, hits_phi)

        hits_energy_mask = hits_energy > self.energy_cut
        hits_data_cut = {
            'hits_energy': hits_energy[hits_energy_mask],
            'hits_r': hits_r[hits_energy_mask],
            'hits_eta': hits_eta[hits_energy_mask],
            'hits_phi': hits_phi[hits_energy_mask]
        }
        self.data['hits_sum_energy_cut'], self.data['hits_avg_eta_cut'], self.data['hits_avg_phi_cut'] = self._calculate_reduced_hits_data(hits_data_cut['hits_energy'], hits_data_cut['hits_r'], hits_data_cut['hits_eta'], hits_data_cut['hits_phi'])

        def reformat_array(array):
            return [row.to_numpy() for row in array]
        
        self.data['hits_energy'] = reformat_array(hits_data_cut['hits_energy'])
        self.data['hits_r'] = reformat_array(hits_data_cut['hits_r'])
        self.data['hits_eta'] = reformat_array(hits_data_cut['hits_eta'])
        self.data['hits_phi'] = reformat_array(hits_data_cut['hits_phi'])

    def _save_data(self):
        data = pd.DataFrame(self.data)
        data.to_hdf(self.outpath, key='SCEPCAL', mode='w')


if __name__ == '__main__':
    SCEPCAL_dir = '/ceph/submit/data/user/v/vinhtran/CERN/CALVISION/SCEPCAL/'
    for file in os.scandir(SCEPCAL_dir):
        if file.name.endswith('.root'):
            outpath = os.path.join('processed_data',file.name.replace('.root','.h5'))

            if os.path.exists(outpath):
                continue

            try:
                DataProcessor(uproot.open(file.path.replace('.root','.root:events')), outpath)
            except:
                print(f'Error processing {file.name}')
                continue