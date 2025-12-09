"""
Script to regenerate samples based on reconstruction quality (not raw correlation).
This selects segments where the autoencoder actually produces good reconstructions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import STFTConfig, STFTDataset
from inference import inverse_stft, pearsonr, zscore
from resnet_autoencoder import ResNetAutoencoder


def load_config(cfg_path: Path) -> Dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_stft_config(cfg: Dict) -> STFTConfig:
    return STFTConfig(
        fs=cfg["data"]["fs_target"],
        window_length=cfg["stft"]["window_length"],
        noverlap=cfg["stft"]["noverlap"],
        nfft=cfg["stft"]["nfft"],
        radar_bandpass=tuple(cfg["filters"]["radar_bandpass"]),
        ecg_bandpass=tuple(cfg["filters"]["ecg_bandpass"]),
        filter_order=cfg["filters"]["order"],
    )


def regenerate_with_autoencoder_quality(
    sample_path: Path, cfg_path: Path, device: str, min_pcc: float = 0.5
) -> None:
    """Regenerate sample by selecting segments based on autoencoder reconstruction quality."""
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(cfg_path)
    stft_cfg = make_stft_config(cfg)

    # Load existing sample
    data = np.load(sample_path, allow_pickle=True)
    radar = data["radar"].reshape(-1)
    psg = data["psg"].reshape(-1)
    seg_seconds = cfg["data"]["segment_seconds"]
    fs = cfg["data"]["fs_target"]
    seg_len = seg_seconds * fs

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

    # Process all segments and compute reconstruction PCC
    num_segments = len(radar) // seg_len
    good_segments = []
    pcc_scores = []

    print(f"[info] Processing {num_segments} segments from {sample_path.name}...")

    with torch.no_grad():
        for i in range(num_segments):
            radar_seg = radar[i * seg_len : (i + 1) * seg_len]
            psg_seg = psg[i * seg_len : (i + 1) * seg_len]

            if len(radar_seg) != seg_len or len(psg_seg) != seg_len:
                continue

            # Create STFT
            radar_stft = STFTDataset.stft(radar_seg, stft_cfg)
            radar_t = (
                torch.tensor(radar_stft, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            # Reconstruct
            decoded = autoencoder(radar_t)
            stft_recon = decoded[0, 0].cpu().numpy()

            # Get input phase and magnitude stats
            nfreq = radar_stft.shape[1] // 2
            mag_in = radar_stft[:, :nfreq]
            phase_in = (radar_stft[:, nfreq:] - 0.5) / 0.5 * np.pi
            mag_mean = float(mag_in.mean())
            mag_std = float(mag_in.std())

            # Try both phase options
            rec_wave_input_phase = inverse_stft(
                stft_recon, cfg, mag_mean=mag_mean, mag_std=mag_std, phase_override=phase_in
            )
            rec_wave_recon_phase = inverse_stft(
                stft_recon, cfg, mag_mean=mag_mean, mag_std=mag_std, phase_override=None
            )

            # Compute PCC for both
            min_len1 = min(len(rec_wave_input_phase), len(psg_seg))
            min_len2 = min(len(rec_wave_recon_phase), len(psg_seg))

            pcc1 = pearsonr(
                zscore(psg_seg[:min_len1]), zscore(rec_wave_input_phase[:min_len1])
            )
            pcc2 = pearsonr(
                zscore(psg_seg[:min_len2]), zscore(rec_wave_recon_phase[:min_len2])
            )

            # Use best PCC
            pcc = max(pcc1, pcc2) if (np.isfinite(pcc1) and np.isfinite(pcc2)) else (
                pcc1 if np.isfinite(pcc1) else (pcc2 if np.isfinite(pcc2) else -1.0)
            )

            pcc_scores.append(pcc)
            if np.isfinite(pcc) and pcc >= min_pcc:
                good_segments.append(i)

    pcc_scores = np.array(pcc_scores)
    print(f"[info] Reconstruction PCC stats:")
    print(f"  Mean: {pcc_scores.mean():.3f}")
    print(f"  Max: {pcc_scores.max():.3f}")
    print(f"  Min: {pcc_scores.min():.3f}")
    print(f"  Segments >= {min_pcc}: {len(good_segments)}/{num_segments}")

    if not good_segments:
        print(f"[warn] No segments with reconstruction PCC >= {min_pcc}")
        print(f"[info] Using best {cfg['data']['segments_per_sample']} segments")
        sorted_idx = np.argsort(pcc_scores)[::-1]
        good_segments = sorted_idx[: cfg["data"]["segments_per_sample"]].tolist()
    else:
        # Keep up to segments_per_sample
        if len(good_segments) > cfg["data"]["segments_per_sample"]:
            # Sort by PCC and keep best
            good_segments = sorted(
                good_segments, key=lambda i: pcc_scores[i], reverse=True
            )[: cfg["data"]["segments_per_sample"]]
        elif len(good_segments) < cfg["data"]["segments_per_sample"]:
            # Fill with next best
            remaining = cfg["data"]["segments_per_sample"] - len(good_segments)
            others = [
                i
                for i in np.argsort(pcc_scores)[::-1]
                if i not in good_segments
            ]
            good_segments += others[:remaining]

    # Extract good segments
    radar_good = np.array([radar[i * seg_len : (i + 1) * seg_len] for i in good_segments])
    psg_good = np.array([psg[i * seg_len : (i + 1) * seg_len] for i in good_segments])

    # Save
    output_path = sample_path
    meta = data.get("meta", {})
    np.savez(
        output_path,
        radar=radar_good,
        psg=psg_good,
        meta=meta,
        keep_indices=np.array(good_segments, dtype=np.int32),
        pcc_scores=pcc_scores,
        reconstruction_pcc_scores=pcc_scores[good_segments],
    )
    print(f"[ok] Regenerated {output_path} with {len(good_segments)} segments (PCC >= {min_pcc})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate samples based on autoencoder reconstruction quality"
    )
    parser.add_argument(
        "--sample", type=Path, required=True, help="Path to sample npz file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "config.json",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-pcc", type=float, default=0.5, help="Minimum reconstruction PCC")
    args = parser.parse_args()

    regenerate_with_autoencoder_quality(args.sample, args.config, args.device, args.min_pcc)

