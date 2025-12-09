"""SSM processing utilities for beat-by-beat signal processing."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks

from ssm_models import CNN1D


def segment_radar_by_peaks(radar_signal: np.ndarray, height_threshold: float = 0.0, min_segment_length: int = 90) -> List[np.ndarray]:
    """
    Segment radar signal into beat-by-beat segments.
    
    Args:
        radar_signal: Input radar signal
        height_threshold: Peak height threshold
        min_segment_length: Minimum segment length
    
    Returns:
        List of segmented radar signals
    """
    # Find peak positions (negative peaks, i.e., minima)
    peaks, _ = find_peaks(-radar_signal, height=height_threshold, prominence=0.4, distance=70)
    segments = []
    
    if len(peaks) == 0:
        return [radar_signal]  # If no peaks found, return entire signal
    
    # Process part before first peak
    if peaks[0] > min_segment_length:
        segments.append(radar_signal[:peaks[0]])
    
    # Segment between each pair of peaks
    prev_end = peaks[0]
    for peak in peaks[1:]:
        if prev_end < peak:
            segments.append(radar_signal[prev_end:peak])
        prev_end = peak
    
    # Process part after last peak
    if len(radar_signal) - prev_end > min_segment_length:
        segments.append(radar_signal[prev_end:])
    
    return segments


def reshape_to_fixed_length(signals: torch.Tensor, target_length: int = 120) -> Tuple[torch.Tensor, int]:
    """
    Reshape beat segments of different lengths to fixed length.
    
    Args:
        signals: Input signal (batch_size, length)
        target_length: Target length
    
    Returns:
        reshaped_signals: Reshaped signals
        original_length: Original length
    """
    original_length = signals.size(1)
    reshaped_signals = torch.nn.functional.interpolate(
        signals.unsqueeze(0), size=target_length, mode='linear', align_corners=True
    ).squeeze(0)
    return reshaped_signals, original_length


def restore_original_length(signals: torch.Tensor, original_length: int) -> torch.Tensor:
    """
    Restore fixed-length output back to original length.
    
    Args:
        signals: Input signal (batch_size, target_length)
        original_length: Original length
    
    Returns:
        restored_signals: Restored signals
    """
    restored_signals = torch.nn.functional.interpolate(
        signals.unsqueeze(1), size=original_length, mode='linear', align_corners=True
    ).squeeze(0)
    return restored_signals


def apply_beat_by_beat_ssm(
    reconstructed_signal: np.ndarray,
    ssm_model_path: Path | None,
    device: str,
    target_length: int = 120,
) -> np.ndarray:
    """
    Apply beat-by-beat SSM processing to reconstructed signal.
    
    Args:
        reconstructed_signal: Signal reconstructed from autoencoder
        ssm_model_path: SSM model path (CNN1D model)
        device: Device
        target_length: Target length
    
    Returns:
        SSM-processed signal
    """
    if ssm_model_path is None or not ssm_model_path.exists():
        return reconstructed_signal
    
    # Segment into beats
    radar_segments = segment_radar_by_peaks(reconstructed_signal, height_threshold=0.0, min_segment_length=90)
    
    if len(radar_segments) == 0:
        return reconstructed_signal
    
    # Load SSM model (CNN1D)
    try:
        ssm_model = CNN1D()
        ssm_model.load_state_dict(torch.load(ssm_model_path, map_location=device))
        ssm_model.to(device)
        ssm_model.eval()
    except Exception as e:
        return reconstructed_signal
    
    # Process each beat segment
    predicted_ecg_concat = []
    
    with torch.no_grad():
        for radar_seg in radar_segments:
            if len(radar_seg) < 10:  # Skip too short segments
                continue
            
            # Reshape to fixed length
            radar_tensor = torch.tensor(radar_seg, dtype=torch.float32).unsqueeze(0)  # (1, length)
            radar_padded, original_length = reshape_to_fixed_length(radar_tensor, target_length=target_length)
            
            # Pass through SSM model
            predicted_ecg = ssm_model(radar_padded.to(device))
            
            # Restore original length
            predicted_ecg_restored = restore_original_length(predicted_ecg.cpu(), original_length)
            predicted_ecg_concat.append(predicted_ecg_restored.squeeze().numpy())
    
    if len(predicted_ecg_concat) == 0:
        return reconstructed_signal
    
    # Concatenate all beats
    final_signal = np.concatenate(predicted_ecg_concat, axis=0)
    
    # Truncate or pad if length doesn't match
    if len(final_signal) > len(reconstructed_signal):
        final_signal = final_signal[:len(reconstructed_signal)]
    elif len(final_signal) < len(reconstructed_signal):
        final_signal = np.pad(final_signal, (0, len(reconstructed_signal) - len(final_signal)), mode='edge')
    
    return final_signal

