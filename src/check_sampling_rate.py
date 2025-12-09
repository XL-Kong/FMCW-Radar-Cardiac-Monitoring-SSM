"""Diagnostic script to check sampling rate consistency and create a figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import STFTConfig, STFTDataset
from inference import inverse_stft, load_config, make_stft_config, pearsonr, zscore
from resnet_autoencoder import ResNetAutoencoder


def check_sampling_rates(sample_path: Path, cfg_path: Path, device: str = "cpu"):
    """Check sampling rates and create diagnostic figure."""
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(cfg_path)
    stft_cfg = make_stft_config(cfg)
    fs = cfg["data"]["fs_target"]
    seg_seconds = cfg["data"]["segment_seconds"]
    seg_len = seg_seconds * fs

    # Load sample
    data = np.load(sample_path, allow_pickle=True)
    radar_segments = data["radar"]  # (N, seg_len)
    psg_segments = data["psg"]  # (N, seg_len)

    # Load autoencoder
    ae_cfg = cfg["model"]["resnet_autoencoder"]
    ae_path = root / ae_cfg["pretrained_path"]
    autoencoder = ResNetAutoencoder(
        encoded_space_dim=ae_cfg["encoded_space_dim"],
        output_channels=ae_cfg["output_channels"],
        dropout=ae_cfg["dropout"],
    )
    autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
    autoencoder.to(device)
    autoencoder.eval()

    # Process first segment
    i = 0
    radar_seg = radar_segments[i]
    psg_seg = psg_segments[i]

    print(f"Segment {i}:")
    print(f"  Radar segment length: {len(radar_seg)} samples")
    print(f"  PSG segment length: {len(psg_seg)} samples")
    print(f"  Expected length: {seg_len} samples (={seg_seconds}s * {fs}Hz)")
    print(f"  Expected duration: {seg_seconds} seconds")

    # Create STFT
    radar_stft = STFTDataset.stft(radar_seg, stft_cfg)
    radar_t = torch.tensor(radar_stft, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Reconstruct
    with torch.no_grad():
        decoded = autoencoder(radar_t)
    stft_recon = decoded[0, 0].cpu().numpy()

    # Inverse STFT
    rec_wave = inverse_stft(stft_recon, cfg)

    print(f"  Reconstructed length: {len(rec_wave)} samples")
    print(f"  Reconstructed duration (if @{fs}Hz): {len(rec_wave)/fs:.3f} seconds")

    # Create diagnostic figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Time axes
    t_radar = np.arange(len(radar_seg)) / fs
    t_psg = np.arange(len(psg_seg)) / fs
    t_rec = np.arange(len(rec_wave)) / fs

    # Plot 1: Radar input
    axes[0].plot(t_radar, radar_seg, label=f"Radar input ({len(radar_seg)} samples, {len(radar_seg)/fs:.3f}s @ {fs}Hz)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Radar Input Signal")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: PSG reference
    axes[1].plot(t_psg, psg_seg, label=f"PSG/ECG ({len(psg_seg)} samples, {len(psg_seg)/fs:.3f}s @ {fs}Hz)", color='green')
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("PSG/ECG Reference Signal")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Comparison
    axes[2].plot(t_psg, zscore(psg_seg), label=f"PSG/ECG (z-scored)", color='green', alpha=0.7)
    axes[2].plot(t_rec, zscore(rec_wave), label=f"Reconstructed ({len(rec_wave)} samples, {len(rec_wave)/fs:.3f}s @ {fs}Hz)", color='red', alpha=0.7)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude (z-scored)")
    axes[2].set_title("Comparison: PSG vs Reconstructed Radar")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Compute PCC
    min_len = min(len(rec_wave), len(psg_seg))
    pcc = pearsonr(zscore(psg_seg[:min_len]), zscore(rec_wave[:min_len]))
    axes[2].text(0.02, 0.98, f"PCC = {pcc:.3f}", transform=axes[2].transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add sampling rate info
    info_text = f"Config: fs_target={fs}Hz, segment={seg_seconds}s, expected={seg_len} samples\n"
    info_text += f"Radar: {len(radar_seg)} samples, PSG: {len(psg_seg)} samples, Reconstructed: {len(rec_wave)} samples"
    fig.suptitle(info_text, fontsize=10, y=0.995)

    plt.tight_layout()
    output_path = root / "outputs" / "sampling_rate_check.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[ok] Diagnostic figure saved -> {output_path}")

    # Check consistency
    if len(radar_seg) != seg_len:
        print(f"[warn] Radar segment length mismatch: {len(radar_seg)} != {seg_len}")
    if len(psg_seg) != seg_len:
        print(f"[warn] PSG segment length mismatch: {len(psg_seg)} != {seg_len}")
    if len(rec_wave) != seg_len:
        print(f"[warn] Reconstructed length mismatch: {len(rec_wave)} != {seg_len}")
    if abs(len(rec_wave) / fs - seg_seconds) > 0.1:
        print(f"[warn] Reconstructed duration mismatch: {len(rec_wave)/fs:.3f}s != {seg_seconds}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check sampling rate consistency.")
    parser.add_argument("--sample", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "sample1.npz")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "config.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    check_sampling_rates(args.sample, args.config, args.device)

