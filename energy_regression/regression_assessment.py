import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from energy_regression.models import DNN
from utils.load_processed_data import ReadSCEPCALData, SCEPCALDataLoader

import numpy as np

from scipy.optimize import curve_fit


class NNRegressionResolution:
    def __init__(self,
                 model : nn.Module,
                 model_state_dict : dict,
                 val_loader : DataLoader,
                 device = 'mps',
                 deviation_bins = np.linspace(-0.5,0.5,101)):
        
        self.device = device

        self.model = model
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device,dtype=torch.float32)

        self.val_loader = val_loader

        self.deviation_bins = deviation_bins

        self._get_predictions()
        self._get_resolution()

    def _get_predictions(self):
        self.model.eval()

        self.predictions = []
        self.targets = []

        with torch.no_grad():
            for val_inputs, val_targets in self.val_loader:
                val_inputs = val_inputs.to(self.device)
                val_targets = val_targets.to(self.device)

                val_outputs = self.model(val_inputs).flatten()

                self.predictions.append(val_outputs.cpu().numpy())
                self.targets.append(val_targets.cpu().numpy())

        self.predictions = np.concatenate(self.predictions)
        self.targets = np.concatenate(self.targets)

    def _get_resolution(self):
        self.deviations = (self.predictions - self.targets) / self.targets

        self.resolution,_ = np.histogram(self.deviations, bins=self.deviation_bins)
        self.resolution_err = np.sqrt(self.resolution)

        self.resolution_centers = (self.deviation_bins[1:] + self.deviation_bins[:-1]) / 2
        self.resolution_normalizer = 1 / np.sum(self.resolution) / np.diff(self.deviation_bins)


class LinearEnergyRegression:
    def __init__(self,
                 total_hits_energy : np.ndarray,
                 initial_momentum : np.ndarray,
                 deviation_bins = np.linspace(-0.5,0.5,101)):
        
        self.total_hits_energy = total_hits_energy
        self.initial_momentum = initial_momentum

        self.deviation_bins = deviation_bins

        self._fit()
        self._get_resolution()

    def linear_fit(self, x, a, b):
        return a*x + b
    
    def _fit(self):
        popt, pcov = curve_fit(self.linear_fit, self.total_hits_energy, self.initial_momentum)

        self.params = popt
        self.params_err = np.sqrt(np.diag(pcov))
    
    def _get_resolution(self):
        self.predicted_momentum = self.linear_fit(self.total_hits_energy, *self.params)
        self.deviations = (self.predicted_momentum - self.initial_momentum) / self.initial_momentum

        self.resolution,_ = np.histogram(self.deviations, bins=self.deviation_bins)
        self.resolution_err = np.sqrt(self.resolution)

        self.resolution_centers = (self.deviation_bins[1:] + self.deviation_bins[:-1]) / 2
        self.resolution_normalizer = 1 / np.sum(self.resolution) / np.diff(self.deviation_bins)