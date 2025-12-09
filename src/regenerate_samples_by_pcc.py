"""Regenerate sample files by running inference and keeping only segments with PCC >= 0.7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import STFTConfig, STFTDataset, load_resampled_pair
from inference import inverse_stft, load_config, make_stft_config, pearsonr, zscore
from resnet_autoencoder import ResNetAutoencoder


def validate_and_filter_sample(
    sample_path: Path,
    cfg: dict,
    device: str,
    min_pcc: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, list[int], list[float]]:
    """Run inference on a sample and return only segments with PCC >= min_pcc."""
    root = Path(__file__).resolve().parents[1]
    stft_cfg = make_stft_config(cfg)

    # Load sample
    data = np.load(sample_path, allow_pickle=True)
    radar_segments = data["radar"]  # (N, seg_len)
    psg_segments = data["psg"]  # (N, seg_len)
    seg_len = radar_segments.shape[1]

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

    # Process each segment
    good_indices = []
    pcc_scores = []
    radar_good = []
    psg_good = []

    with torch.no_grad():
        for i in range(len(radar_segments)):
            radar_seg = radar_segments[i]
            psg_seg = psg_segments[i]

            # Create STFT
            radar_stft = STFTDataset.stft(radar_seg, stft_cfg)
            radar_t = torch.tensor(radar_stft, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

            # Reconstruct
            encoded = autoencoder.encoder(radar_t)
            decoded = autoencoder.decoder(encoded)
            stft_recon = decoded[0, 0].cpu().numpy()

            # Inverse STFT to get waveform
            rec_wave = inverse_stft(stft_recon, cfg)

            # Compute PCC
            min_len = min(len(rec_wave), len(psg_seg))
            rec_wave_norm = zscore(rec_wave[:min_len])
            psg_seg_norm = zscore(psg_seg[:min_len])
            pcc = pearsonr(psg_seg_norm, rec_wave_norm)

            pcc_scores.append(pcc)
            if np.isfinite(pcc) and pcc >= min_pcc:
                good_indices.append(i)
                radar_good.append(radar_seg)
                psg_good.append(psg_seg)

    if not good_indices:
        print(f"[warn] {sample_path.name}: no segments with PCC >= {min_pcc}")
        return None, None, [], pcc_scores

    radar_filtered = np.array(radar_good)
    psg_filtered = np.array(psg_good)
    print(f"[ok] {sample_path.name}: {len(good_indices)}/{len(radar_segments)} segments pass PCC >= {min_pcc}")

    return radar_filtered, psg_filtered, good_indices, pcc_scores


def regenerate_samples(cfg_path: Path, min_pcc: float = 0.7, device: str = "cpu"):
    """Regenerate all sample files keeping only segments with PCC >= min_pcc."""
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(cfg_path)
    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    for source in cfg["data"]["sources"]:
        name = source["name"]
        sample_path = out_dir / f"{name}.npz"

        if not sample_path.exists():
            print(f"[skip] {name}: sample file not found")
            continue

        radar_filtered, psg_filtered, good_indices, pcc_scores = validate_and_filter_sample(
            sample_path, cfg, device, min_pcc
        )

        if radar_filtered is None:
            print(f"[skip] {name}: no valid segments, skipping regeneration")
            continue

        # Load original metadata
        data = np.load(sample_path, allow_pickle=True)
        meta = data.get("meta", {}).item() if hasattr(data.get("meta", {}), "item") else data.get("meta", {})

        # Save filtered sample
        output_path = out_dir / f"{name}.npz"
        np.savez(
            output_path,
            radar=radar_filtered,
            psg=psg_filtered,
            meta=meta,
            keep_indices=np.array(good_indices, dtype=np.int32),
            pcc_scores=np.array(pcc_scores, dtype=np.float32),
        )
        print(f"[ok] regenerated {output_path} ({len(good_indices)} segments)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate samples based on reconstruction PCC.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "config.json",
    )
    parser.add_argument("--min-pcc", type=float, default=0.7, help="Minimum PCC threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    regenerate_samples(args.config, args.min_pcc, args.device)

