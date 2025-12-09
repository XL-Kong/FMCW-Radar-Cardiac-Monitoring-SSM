"""Signal processing utilities."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1] range."""
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val == min_val:
        return np.zeros_like(x)
    return 2 * (x - min_val) / (max_val - min_val) - 1


def normalize_0to1(x: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] range."""
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val == min_val:
        return np.zeros_like(x)
    return (x - min_val) / (max_val - min_val)


def smooth_edges(sig: np.ndarray, frac: float = 0.08, min_len: int = 5) -> np.ndarray:
    """Apply a gentle taper on both ends to reduce edge slope/spikes."""
    if sig.size == 0:
        return sig
    edge_len = max(min_len, int(len(sig) * frac))
    edge_len = min(edge_len, len(sig) // 2)
    if edge_len <= 0:
        return sig
    w = np.ones_like(sig, dtype=np.float32)
    ramp = np.linspace(0, 1, edge_len, dtype=np.float32)
    w[:edge_len] = ramp
    w[-edge_len:] = ramp[::-1]
    return sig * w


def taper_edges_residual(x: np.ndarray, fs: float, t_taper: float = 0.2) -> np.ndarray:
    """
    Apply Hann taper to signal edges on residual (detrended) component.
    Preserves large-scale trend while suppressing edge artifacts.
    
    Args:
        x: Input signal
        fs: Sampling frequency (Hz)
        t_taper: Taper duration in seconds for each edge
    
    Returns:
        Tapered signal with trend preserved
    """
    N = len(x)
    if N == 0:
        return x
    
    n_taper = int(t_taper * fs)
    if 2 * n_taper >= N:
        # If taper is too long, use a smaller fraction
        n_taper = max(1, N // 10)
    
    if n_taper <= 0:
        return x
    
    # Fit linear trend as baseline
    t = np.arange(N, dtype=np.float32)
    coef = np.polyfit(t, x, 1)  # First-order polynomial fit
    trend = np.polyval(coef, t)
    resid = x - trend
    
    # Apply Hann taper to residual
    w = np.ones(N, dtype=np.float32)
    hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(2 * n_taper) / (2 * n_taper - 1)))
    w[:n_taper] = hann[:n_taper]
    w[-n_taper:] = hann[n_taper:]
    
    resid_taper = resid * w
    return trend + resid_taper


def crossfade(prev: np.ndarray, cur: np.ndarray, frac: float = 0.1, min_len: int = 5) -> np.ndarray:
    """
    Crossfade overlap region to ensure smooth transition between beats.
    Returns blended prev_tail+cur_body (already stitched).
    """
    if len(prev) == 0 or len(cur) == 0:
        return cur if len(prev) == 0 else prev
    overlap = max(min_len, int(min(len(prev), len(cur)) * frac))
    if overlap <= 0 or overlap >= len(prev) or overlap >= len(cur):
        return np.concatenate([prev, cur])
    w_prev = np.linspace(1, 0, overlap, dtype=np.float32)
    w_cur = np.linspace(0, 1, overlap, dtype=np.float32)
    tail_prev = prev[-overlap:] * w_prev
    head_cur = cur[:overlap] * w_cur
    stitched = np.concatenate([prev[:-overlap], tail_prev + head_cur, cur[overlap:]])
    return stitched


def inverse_stft(
    stft_arr: np.ndarray,
    cfg: dict,
    cutoff: float = 4.0,
    use_input_phase: bool = False,
    phase_override: np.ndarray | None = None,
    expected_samples: int | None = None,
) -> np.ndarray:
    """
    Inverse STFT matching original code exactly.
    
    Args:
        stft_arr: Reconstructed STFT from autoencoder (64, 64)
        cfg: Configuration dict
        cutoff: Lowpass filter cutoff in Hz (default 4.0)
        use_input_phase: If True, use input phase; if False, use reconstructed phase
        phase_override: Optional phase to override (if use_input_phase=True)
        expected_samples: Optional expected output length (if None, uses cfg segment_seconds * fs)
    """
    win_len = cfg["stft"]["window_length"]
    noverlap = cfg["stft"]["noverlap"]
    nfft = cfg["stft"]["nfft"]
    fs = cfg["data"]["fs_target"]
    if expected_samples is None:
        seg_seconds = cfg["data"]["segment_seconds"]
        expected_samples = int(seg_seconds * fs)

    # Reverse zoom: from (64, 64) back to (55, 66) as in original code
    from scipy.ndimage import zoom as scipy_zoom
    
    # Ensure input is 2D
    if stft_arr.ndim != 2:
        raise ValueError(f"Expected 2D STFT array, got shape {stft_arr.shape} with {stft_arr.ndim} dimensions")
    
    orig_h, orig_w = stft_arr.shape
    
    # If input is (64, 64), zoom to (55, 66)
    if orig_h == 64 and orig_w == 64:
        s = scipy_zoom(stft_arr, (55/64, 66/64), order=1)
    else:
        # For other sizes, try to zoom proportionally
        s = stft_arr
        if orig_h != 55 or orig_w != 66:
            # Try to resize to (55, 66)
            s = scipy_zoom(stft_arr, (55/orig_h, 66/orig_w), order=1)
    
    # Ensure exact dimensions (55, 66)
    if s.shape[0] != 55:
        if s.shape[0] > 55:
            s = s[:55, :]
        else:
            pad = np.tile(s[-1:, :], (55 - s.shape[0], 1))
            s = np.vstack([s, pad])
    
    if s.shape[1] != 66:
        if s.shape[1] > 66:
            s = s[:, :66]
        else:
            pad = np.tile(s[:, -1:], (1, 66 - s.shape[1]))
            s = np.hstack([s, pad])
    
    # Split into magnitude and phase (exactly as original)
    cols = s.shape[1]  # Should be 66
    amp = s[:, :cols // 2]  # Magnitude (55, 33)
    
    n_freq = cols // 2  # Should be 33
    
    if use_input_phase and phase_override is not None:
        # Use input phase (reverse zoom if needed)
        if phase_override.shape[0] == 64:
            # Reverse zoom to (55, n_freq)
            pha = scipy_zoom(phase_override, (55/64, n_freq/phase_override.shape[1]), order=1)
        else:
            pha = phase_override
        
        # Ensure exact dimensions (55, 33)
        if pha.shape[0] != 55:
            if pha.shape[0] > 55:
                pha = pha[:55, :]
            else:
                pad = np.tile(pha[-1:, :], (55 - pha.shape[0], 1))
                pha = np.vstack([pha, pad])
        if pha.shape[1] != n_freq:
            if pha.shape[1] > n_freq:
                pha = pha[:, :n_freq]
            else:
                pad = np.tile(pha[:, -1:], (1, n_freq - pha.shape[1]))
                pha = np.hstack([pha, pad])
        
        # Convert from radians to [0,1] then back (if needed)
        if pha.min() < -0.1 or pha.max() > 2*np.pi + 0.1:
            # Already in radians, convert to [0,1] scale then back
            pha_scaled = (pha / np.pi) * 0.5 + 0.5
            pha = 2 * (pha_scaled - 0.5) * np.pi
    else:
        # Use reconstructed phase (from decoder output)
        pha_scaled = s[:, n_freq:]  # Phase scaled to [0,1], should be (55, 33)
        # Ensure correct dimensions
        if pha_scaled.shape[1] != n_freq:
            if pha_scaled.shape[1] > n_freq:
                pha_scaled = pha_scaled[:, :n_freq]
            else:
                pad = np.tile(pha_scaled[:, -1:], (1, n_freq - pha_scaled.shape[1]))
                pha_scaled = np.hstack([pha_scaled, pad])
        pha = 2 * (pha_scaled - 0.5) * np.pi  # Convert to radians
    
    # Form complex spectrum: amp * exp(1j * pha)
    s_complex = amp * np.exp(1j * pha)
    
    # Transpose for istft: (freq, time)
    s_complex = s_complex.T
    
    # Inverse STFT
    _, reconstructed_signal = signal.istft(
        s_complex,
        fs=fs,
        window=signal.windows.hamming(win_len),
        nperseg=win_len,
        noverlap=noverlap,
        nfft=nfft,
    )
    reconstructed_signal = reconstructed_signal.real
    
    # Apply lowpass filter
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = butter(4, high, btype='lowpass')
    reconstructed_signal = filtfilt(b, a, reconstructed_signal)
    
    # Normalize to [-1, 1] as in original code
    reconstructed_signal = normalize_minmax(reconstructed_signal)
    
    # Truncate to exact expected length
    if len(reconstructed_signal) > expected_samples:
        reconstructed_signal = reconstructed_signal[:expected_samples]
    elif len(reconstructed_signal) < expected_samples:
        reconstructed_signal = np.pad(
            reconstructed_signal, (0, expected_samples - len(reconstructed_signal)), mode='edge'
        )
    
    return reconstructed_signal


def pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    if a.size == 0 or b.size == 0:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom == 0:
        return np.nan
    return float((a * b).sum() / denom)


def zscore(x: np.ndarray) -> np.ndarray:
    """Compute z-score normalization."""
    std = x.std()
    if std == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / std

