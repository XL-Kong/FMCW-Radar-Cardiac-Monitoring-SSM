from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.ndimage import zoom

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - allow dataset slicing without torch installed
    torch = None

    class Dataset:  # type: ignore
        pass


@dataclass
class STFTConfig:
    fs: int
    window_length: int
    noverlap: int
    nfft: int
    radar_bandpass: Tuple[float, float]
    ecg_bandpass: Tuple[float, float]
    filter_order: int


def _bandpass(sig: np.ndarray, fs: int, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, sig)


def load_resampled_pair(
    radar_path: Path,
    psg_path: Path,
    fs_target: int,
    radar_source_rate: int,
    ecg_source_rate: int,
    cfg: STFTConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    radar_raw = pd.read_csv(radar_path, delimiter=" ", header=None).values[:, 8]
    radar = signal.resample_poly(radar_raw, up=fs_target, down=radar_source_rate)
    radar = _bandpass(radar, fs_target, cfg.radar_bandpass[0], cfg.radar_bandpass[1], cfg.filter_order)

    psg_raw = pd.read_csv(psg_path, delimiter=" ").values[:, 1] * 1000
    psg = signal.resample_poly(psg_raw, up=fs_target, down=ecg_source_rate)
    psg = _bandpass(psg, fs_target, cfg.ecg_bandpass[0], cfg.ecg_bandpass[1], cfg.filter_order)

    limit = min(len(radar), len(psg))
    return radar[:limit], psg[:limit]


class STFTDataset(Dataset):
    """
    Minimal STFT dataset: loads radar + PSG, applies resampling & filtering, and
    returns paired STFT tensors for inference.
    """

    def __init__(
        self,
        radar: np.ndarray,
        psg: np.ndarray,
        segment_seconds: int,
        cfg: STFTConfig,
        fs: int,
    ):
        if torch is None:
            raise ImportError("torch is required to use STFTDataset; install torch/torchvision.")
        self.fs = fs
        self.cfg = cfg
        self.segment_length = segment_seconds * fs
        usable = min(len(radar), len(psg))
        usable = usable - (usable % self.segment_length)
        self.radar = radar[:usable]
        self.psg = psg[:usable]
        self.num_segments = usable // self.segment_length

    def __len__(self) -> int:
        return self.num_segments

    @staticmethod
    def stft(x: np.ndarray, cfg: STFTConfig) -> np.ndarray:
        """STFT matching original code exactly.
        
        Original code in FinalSubmission/dataset.py:
        - stft returns s.T which is (time_bins, freq_bins*2)
        - Then applies: zoom(radar_stft, (64/55, 64/66))
        - This means original STFT is (55, 66), then zoomed to (64, 64)
        """
        win = signal.windows.hamming(cfg.window_length)
        _, _, s = signal.stft(
            x,
            fs=cfg.fs,
            window=win,
            nperseg=cfg.window_length,
            noverlap=cfg.noverlap,
            nfft=cfg.nfft,
        )
        s[np.abs(s) < 1e-2] = 0
        mag = np.abs(s)
        ang = np.angle(s) / np.pi * 0.5 + 0.5
        stft = np.concatenate((mag, ang), axis=0).T  # Shape: (time_bins, freq_bins*2)
        # Original code uses fixed zoom: (64/55, 64/66) from (55, 66) to (64, 64)
        # But for shorter segments (e.g., beat-matched 120 samples), dimensions differ
        # Handle variable dimensions
        if stft.ndim == 2:
            orig_h, orig_w = stft.shape
            if orig_h > 0 and orig_w > 0:
                # Zoom to (64, 64) proportionally
                stft = zoom(stft, (64 / orig_h, 64 / orig_w), order=1)
        return stft

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.segment_length
        end = start + self.segment_length

        radar_seg = self.radar[start:end]
        psg_seg = self.psg[start:end]

        radar_stft = self.stft(radar_seg, self.cfg)
        psg_stft = self.stft(psg_seg, self.cfg)

        radar_t = torch.tensor(radar_stft, dtype=torch.float32).unsqueeze(0)
        psg_t = torch.tensor(psg_stft, dtype=torch.float32).unsqueeze(0)
        return radar_t, psg_t


def split_segments(
    radar: np.ndarray,
    psg: np.ndarray,
    fs: int,
    segment_seconds: int,
    segments_per_sample: int,
    start_offset_seconds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    seg_len = segment_seconds * fs
    start = max(0, start_offset_seconds * fs)
    total_needed = segments_per_sample * seg_len

    limit = min(len(radar), len(psg))
    if start + total_needed > limit:
        start = max(0, limit - total_needed)
    end = start + total_needed
    radar_slice = radar[start:end]
    psg_slice = psg[start:end]

    radar_segments = radar_slice.reshape(segments_per_sample, seg_len)
    psg_segments = psg_slice.reshape(segments_per_sample, seg_len)
    return radar_segments, psg_segments

