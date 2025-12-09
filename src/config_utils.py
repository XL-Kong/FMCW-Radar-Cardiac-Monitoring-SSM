"""Configuration and data loading utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from dataset import STFTConfig


def load_config(cfg_path: Path) -> Dict:
    """Load configuration from JSON file."""
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_stft_config(cfg: Dict) -> STFTConfig:
    """Create STFTConfig from configuration dictionary."""
    return STFTConfig(
        fs=cfg["data"]["fs_target"],
        window_length=cfg["stft"]["window_length"],
        noverlap=cfg["stft"]["noverlap"],
        nfft=cfg["stft"]["nfft"],
        radar_bandpass=tuple(cfg["filters"]["radar_bandpass"]),
        ecg_bandpass=tuple(cfg["filters"]["ecg_bandpass"]),
        filter_order=cfg["filters"]["order"],
    )


def load_sample(sample_path: Path, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """
    Load sample data.
    
    Returns:
        (radar, psg, seg_len, is_beat_matched)
        is_beat_matched: True if data is beat-matched, False if continuous
    """
    data = np.load(sample_path, allow_pickle=True)
    meta = data.get("meta", None)
    if meta is not None:
        if hasattr(meta, 'item'):
            meta = meta.item()
        if isinstance(meta, dict) and meta.get("segment_type") == "beat_matched":
            # Beat-matched data: segments are already separated
            radar_segments = data["radar"]  # (N_beats, beat_length)
            psg_segments = data["psg"]  # (N_beats, beat_length)
            return radar_segments, psg_segments, radar_segments.shape[1], True
    
    # Continuous data (old format)
    radar = data["radar"].reshape(-1)
    psg = data["psg"].reshape(-1)
    seg_seconds = cfg["data"]["segment_seconds"]
    fs = cfg["data"]["fs_target"]
    expected = cfg["data"]["segments_per_sample"] * seg_seconds * fs
    radar = radar[:expected]
    psg = psg[:expected]
    seg_len = seg_seconds * fs
    return radar, psg, seg_len, False

