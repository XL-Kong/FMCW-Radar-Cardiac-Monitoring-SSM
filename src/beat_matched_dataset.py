"""
Dataset for beat-matched ECG and radar data.
This dataset loads pre-segmented beat-matched pairs and processes them for inference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.ndimage import zoom

from tools import smooth_ecg, low_pass_filter


class BeatMatchedDataset(Dataset):
    """Dataset for beat-matched ECG and radar segments."""
    
    def __init__(
        self,
        radar_segments: np.ndarray,
        ecg_segments: np.ndarray,
        fs: int = 128,
    ):
        """
        Args:
            radar_segments: Array of radar beat segments (N_beats, segment_length)
            ecg_segments: Array of ECG beat segments (N_beats, segment_length)
            fs: Sampling rate
        """
        self.radar_segments = radar_segments
        self.ecg_segments = ecg_segments
        self.fs = fs
        
        assert len(self.radar_segments) == len(self.ecg_segments), "Radar and ECG segment counts must match!"
    
    @staticmethod
    def stft(data: np.ndarray) -> np.ndarray:
        """Compute STFT for a signal."""
        fs = 128
        window = signal.windows.hamming(16)
        noverlap = 4
        nfft = 64
        
        # Ensure data is 1D
        if data.ndim > 1:
            data = data.flatten()
        
        _, _, s = signal.stft(data, fs=fs, window=window, nperseg=16, noverlap=noverlap, nfft=nfft)
        s[np.abs(s) < 1e-2] = 0
        s1 = np.abs(s)
        s2 = np.angle(s)
        s2 = s2 / np.pi * 0.5 + 0.5
        s = np.concatenate((s1, s2), axis=0)
        stft = s.T  # Shape: (time_bins, freq_bins*2)
        
        # Zoom to (64, 64) - handle variable dimensions
        if stft.ndim == 2:
            orig_h, orig_w = stft.shape
            if orig_h > 0 and orig_w > 0:
                try:
                    stft = zoom(stft, (64 / orig_h, 64 / orig_w), order=1)
                except Exception as e:
                    # Fallback: pad or crop to (64, 64)
                    if orig_h < 64:
                        stft = np.pad(stft, ((0, 64 - orig_h), (0, 0)), mode='edge')
                    elif orig_h > 64:
                        stft = stft[:64, :]
                    if orig_w < 64:
                        stft = np.pad(stft, ((0, 0), (0, 64 - orig_w)), mode='edge')
                    elif orig_w > 64:
                        stft = stft[:, :64]
        
        return stft
    
    def __len__(self):
        return len(self.radar_segments)
    
    def __getitem__(self, idx):
        radar_segment = self.radar_segments[idx]
        ecg_segment = self.ecg_segments[idx]
        
        # Apply processing as in original ECG_Radar_Dataset
        ecg_segment = low_pass_filter(ecg_segment, cutoff=10, fs=self.fs)
        ecg_segment = smooth_ecg(ecg_segment, window_size=3)
        
        # Compute STFT (already zoomed to 64x64 in stft method)
        radar_stft = self.stft(radar_segment)
        ecg_stft = self.stft(ecg_segment)
        
        # Convert to tensors
        radar_tensor = torch.tensor(radar_stft, dtype=torch.float32).unsqueeze(0)
        ecg_tensor = torch.tensor(ecg_stft, dtype=torch.float32).unsqueeze(0)
        
        return radar_tensor, ecg_tensor

