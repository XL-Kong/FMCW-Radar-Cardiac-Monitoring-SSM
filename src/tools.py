"""
Utility functions for signal processing.
"""
import numpy as np
import math


def smooth_ecg(ecg_signal, window_size=3):
    """使用移动平均滤波对ECG信号进行平滑."""
    if window_size <= 1:
        return ecg_signal.astype(np.float32)
    pad = window_size // 2
    xp = np.pad(ecg_signal, (pad, pad), mode='edge')
    k = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(xp, k, mode='valid').astype(np.float32)


def low_pass_filter(x, cutoff=10, fs=128):
    """
    低通滤波器，去除高于 cutoff 的频率成分
    Args:
        x: 输入信号数据
        cutoff: 截止频率，默认为10 Hz
        fs: 采样率，默认为128 Hz
    Returns:
        滤波后的数据
    """
    alpha = math.exp(-2 * math.pi * cutoff / fs)
    y = np.zeros_like(x, dtype=np.float32)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + (1 - alpha) * x[i]
    return y

