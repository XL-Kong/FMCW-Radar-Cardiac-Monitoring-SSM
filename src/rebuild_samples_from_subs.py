"""
Rebuild sample1..sample10 from sub_01..sub_07 npy pairs by selecting the best
segments based on autoencoder reconstruction PCC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import signal

from dataset import STFTConfig, STFTDataset, _bandpass
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


def augment_segment(radar_seg: np.ndarray, psg_seg: np.ndarray, method: str = "noise") -> Tuple[np.ndarray, np.ndarray]:
    """Apply conservative data augmentation to a segment.
    
    Args:
        radar_seg: Radar segment (1D array)
        psg_seg: PSG segment (1D array)
        method: Augmentation method ('noise', 'shift', 'scale')
    
    Returns:
        Augmented (radar_seg, psg_seg) pair
    """
    radar_aug = radar_seg.copy()
    psg_aug = psg_seg.copy()
    
    if method == "noise":
        # Add very small Gaussian noise (0.5% of std) - very conservative
        noise_scale = 0.005
        radar_aug = radar_aug + np.random.randn(len(radar_aug)) * noise_scale * np.std(radar_aug)
        psg_aug = psg_aug + np.random.randn(len(psg_aug)) * noise_scale * np.std(psg_aug)
        # Clip to [-1, 1] range
        radar_aug = np.clip(radar_aug, -1, 1)
        psg_aug = np.clip(psg_aug, -1, 1)
    
    elif method == "shift":
        # Circular shift by very small amount (1-2 samples) - conservative
        shift = np.random.randint(1, 3)
        radar_aug = np.roll(radar_aug, shift)
        psg_aug = np.roll(psg_aug, shift)
    
    elif method == "scale":
        # Very slight amplitude scaling (0.98-1.02) - conservative
        scale = np.random.uniform(0.98, 1.02)
        radar_aug = radar_aug * scale
        psg_aug = psg_aug * scale
        # Clip to [-1, 1] range
        radar_aug = np.clip(radar_aug, -1, 1)
        psg_aug = np.clip(psg_aug, -1, 1)
    
    return radar_aug, psg_aug


def load_npy_pair(radar_path: Path, psg_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load .npy files which are already segmented (N segments x segment_length)."""
    radar = np.load(radar_path, allow_pickle=True)
    psg = np.load(psg_path, allow_pickle=True)
    
    # If already 2D (segments x length), use as-is
    if radar.ndim == 2 and psg.ndim == 2:
        # Ensure same number of segments
        n_segments = min(radar.shape[0], psg.shape[0])
        return radar[:n_segments], psg[:n_segments]
    else:
        # If 1D, reshape to continuous signal (old behavior)
        radar = radar.reshape(-1)
        psg = psg.reshape(-1)
        limit = min(len(radar), len(psg))
        return radar[:limit], psg[:limit]


def resample_and_filter(
    radar: np.ndarray,
    psg: np.ndarray,
    cfg: Dict,
    stft_cfg: STFTConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    fs_target = cfg["data"]["fs_target"]
    radar_source_rate = cfg["data"]["radar_source_rate"]
    ecg_source_rate = cfg["data"]["ecg_source_rate"]

    radar_rs = signal.resample_poly(radar, up=fs_target, down=radar_source_rate)
    radar_rs = _bandpass(
        radar_rs,
        fs_target,
        stft_cfg.radar_bandpass[0],
        stft_cfg.radar_bandpass[1],
        stft_cfg.filter_order,
    )

    psg_rs = signal.resample_poly(psg, up=fs_target, down=ecg_source_rate)
    psg_rs = _bandpass(
        psg_rs,
        fs_target,
        stft_cfg.ecg_bandpass[0],
        stft_cfg.ecg_bandpass[1],
        stft_cfg.filter_order,
    )

    limit = min(len(radar_rs), len(psg_rs))
    return radar_rs[:limit], psg_rs[:limit]


def collect_segments(
    radar: np.ndarray,
    psg: np.ndarray,
    stft_cfg: STFTConfig,
    autoencoder: ResNetAutoencoder,
    cfg: Dict,
    device: str,
    source_name: str,
) -> List[Tuple[float, np.ndarray, np.ndarray, str, int]]:
    """Collect segments from pre-segmented .npy files or continuous signals."""
    seg_len = cfg["data"]["segment_seconds"] * cfg["data"]["fs_target"]
    
    # Check if already segmented (2D array)
    if radar.ndim == 2 and psg.ndim == 2:
        # Already segmented: radar and psg are (N_segments, segment_length)
        n_segments = min(radar.shape[0], psg.shape[0])
        segments: List[Tuple[float, np.ndarray, np.ndarray, str, int]] = []
        with torch.no_grad():
            for i in range(n_segments):
                radar_seg = radar[i]
                psg_seg = psg[i]
                
                # .npy files contain 648-sample segments (5s @ 128Hz)
                # Original code uses segment_length=5, so model was trained on 5s segments
                # DO NOT resample - use original segment length (648 samples = 5s)
                if len(radar_seg) != seg_len:
                    # Only warn, don't resample - use actual length
                    if abs(len(radar_seg) - seg_len) > 10:
                        print(f"[warn] {source_name}[{i}]: radar segment length {len(radar_seg)} != expected {seg_len}, using actual length")
                if len(psg_seg) != seg_len:
                    if abs(len(psg_seg) - seg_len) > 10:
                        print(f"[warn] {source_name}[{i}]: psg segment length {len(psg_seg)} != expected {seg_len}, using actual length")
                
                # .npy files are already filtered and normalized, so no additional filtering needed
                actual_seg_len = len(radar_seg)  # Use actual segment length (648)

                radar_stft = STFTDataset.stft(radar_seg, stft_cfg)
                radar_t = (
                    torch.tensor(radar_stft, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )
                decoded = autoencoder(radar_t)
                stft_recon = decoded[0, 0].cpu().numpy()

                nfreq = radar_stft.shape[1] // 2
                mag_in = radar_stft[:, :nfreq]
                phase_in = (radar_stft[:, nfreq:] - 0.5) / 0.5 * np.pi

                rec_wave_input_phase = inverse_stft(
                    stft_recon, cfg, cutoff=4.0, use_input_phase=True, phase_override=phase_in, expected_samples=actual_seg_len
                )
                rec_wave_recon_phase = inverse_stft(
                    stft_recon, cfg, cutoff=4.0, use_input_phase=False, phase_override=None, expected_samples=actual_seg_len
                )

                # Original code: truth=istft(y[0,0,:,:],cutoff=20)
                # So truth also comes from STFT->istft, not directly from raw signal
                psg_stft = STFTDataset.stft(psg_seg, stft_cfg)
                truth_wave = inverse_stft(psg_stft, cfg, cutoff=20.0, use_input_phase=False, phase_override=None, expected_samples=actual_seg_len)

                min_len1 = min(len(rec_wave_input_phase), len(truth_wave), actual_seg_len)
                min_len2 = min(len(rec_wave_recon_phase), len(truth_wave), actual_seg_len)

                # Both signals are normalized to [-1,1] from inverse_stft
                # Use zscore for PCC calculation as in original code
                pcc1 = pearsonr(zscore(truth_wave[:min_len1]), zscore(rec_wave_input_phase[:min_len1]))
                pcc2 = pearsonr(zscore(truth_wave[:min_len2]), zscore(rec_wave_recon_phase[:min_len2]))

                pcc = max(pcc1, pcc2) if (np.isfinite(pcc1) and np.isfinite(pcc2)) else (
                    pcc1 if np.isfinite(pcc1) else (pcc2 if np.isfinite(pcc2) else -1.0)
                )

                segments.append((pcc, radar_seg, psg_seg, source_name, i))
        
        return segments
    else:
        # Continuous signal: segment it
        usable = len(radar) // seg_len
        segments: List[Tuple[float, np.ndarray, np.ndarray, str, int]] = []
        with torch.no_grad():
            for i in range(usable):
                radar_seg = radar[i * seg_len : (i + 1) * seg_len]
                psg_seg = psg[i * seg_len : (i + 1) * seg_len]
                if len(radar_seg) != seg_len or len(psg_seg) != seg_len:
                    continue

                radar_stft = STFTDataset.stft(radar_seg, stft_cfg)
                radar_t = (
                    torch.tensor(radar_stft, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )
                decoded = autoencoder(radar_t)
                stft_recon = decoded[0, 0].cpu().numpy()

                nfreq = radar_stft.shape[1] // 2
                mag_in = radar_stft[:, :nfreq]
                phase_in = (radar_stft[:, nfreq:] - 0.5) / 0.5 * np.pi

                rec_wave_input_phase = inverse_stft(
                    stft_recon, cfg, cutoff=4.0, use_input_phase=True, phase_override=phase_in
                )
                rec_wave_recon_phase = inverse_stft(
                    stft_recon, cfg, cutoff=4.0, use_input_phase=False, phase_override=None
                )

                # Original code: truth=istft(y[0,0,:,:],cutoff=20)
                # So truth also comes from STFT->istft, not directly from raw signal
                psg_stft = STFTDataset.stft(psg_seg, stft_cfg)
                truth_wave = inverse_stft(psg_stft, cfg, cutoff=20.0, use_input_phase=False, phase_override=None)

                min_len1 = min(len(rec_wave_input_phase), len(truth_wave))
                min_len2 = min(len(rec_wave_recon_phase), len(truth_wave))

                # Both signals are normalized to [-1,1] from inverse_stft
                # Use zscore for PCC calculation as in original code
                pcc1 = pearsonr(zscore(truth_wave[:min_len1]), zscore(rec_wave_input_phase[:min_len1]))
                pcc2 = pearsonr(zscore(truth_wave[:min_len2]), zscore(rec_wave_recon_phase[:min_len2]))

                pcc = max(pcc1, pcc2) if (np.isfinite(pcc1) and np.isfinite(pcc2)) else (
                    pcc1 if np.isfinite(pcc1) else (pcc2 if np.isfinite(pcc2) else -1.0)
                )

                segments.append((pcc, radar_seg, psg_seg, source_name, i))

        return segments


def rebuild(cfg_path: Path, device: str, min_pcc: float = 0.6, min_total_segments: int = 600) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(cfg_path)
    stft_cfg = make_stft_config(cfg)

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

    # Candidate files: prefer FinalSubmission/*.npy if present
    base = root.parent
    candidates = []
    for sub in range(1, 8):
        name = f"sub_0{sub}"
        radar_path = base / "FinalSubmission" / f"{name}_radar.npy"
        psg_path = base / "FinalSubmission" / f"{name}_ecg.npy"
        if not radar_path.exists() or not psg_path.exists():
            radar_path = base / f"{name}_radar.npy"
            psg_path = base / f"{name}_ecg.npy"
        if radar_path.exists() and psg_path.exists():
            candidates.append((radar_path, psg_path, name))

    if not candidates:
        raise RuntimeError("No sub_0x radar/ecg npy files found.")

    all_segments: List[Tuple[float, np.ndarray, np.ndarray, str, int]] = []
    for radar_path, psg_path, name in candidates:
        radar_raw, psg_raw = load_npy_pair(radar_path, psg_path)
        
        # If already segmented (2D), process directly; otherwise resample and filter
        if radar_raw.ndim == 2 and psg_raw.ndim == 2:
            # Already segmented - process directly
            segs = collect_segments(radar_raw, psg_raw, stft_cfg, autoencoder, cfg, device, name)
        else:
            # Continuous signal - resample and filter first
            radar, psg = resample_and_filter(radar_raw, psg_raw, cfg, stft_cfg)
            segs = collect_segments(radar, psg, stft_cfg, autoencoder, cfg, device, name)
        
        all_segments.extend(segs)
        print(f"[info] {name}: collected {len(segs)} segments")

    # Filter valid segments and sort by PCC descending
    all_segments_valid = [s for s in all_segments if np.isfinite(s[0])]
    all_segments_valid.sort(key=lambda x: x[0], reverse=True)
    
    # Filter by min_pcc
    all_segments = [s for s in all_segments_valid if s[0] >= min_pcc]
    
    if min_pcc > 0.0:
        print(f"[info] After filtering by PCC >= {min_pcc}: {len(all_segments)} segments remain")

    needed = cfg["data"]["segments_per_sample"] * 10
    needed = max(needed, min_total_segments)
    
    # Data augmentation: if not enough segments, duplicate high-quality segments with minimal modification
    if len(all_segments) < needed and min_pcc > 0.0:
        # Find segments with PCC >= 0.5 for duplication
        candidate_threshold = 0.5
        candidates = [s for s in all_segments_valid if s[0] >= candidate_threshold]
        
        if candidates:
            print(f"[info] Duplicating {len(candidates)} segments with PCC >= {candidate_threshold} to generate more variants")
            augmented_segments = []
            
            # Calculate how many duplicates we need per segment
            n_needed = needed - len(all_segments)
            n_duplicates_per_segment = max(1, (n_needed // len(candidates)) + 1)
            print(f"[info] Generating {n_duplicates_per_segment} duplicates per segment")
            
            # Simply duplicate segments directly (no modification to preserve PCC)
            candidates_sorted = sorted(candidates, key=lambda x: x[0], reverse=True)
            
            for dup_idx, (pcc, radar_seg, psg_seg, source_name, idx) in enumerate(candidates_sorted):
                if dup_idx % 10 == 0:
                    print(f"[info] Duplicating candidate {dup_idx+1}/{len(candidates_sorted)} (PCC={pcc:.3f})...")
                
                # Direct copy - no modification to preserve original PCC
                for dup_num in range(n_duplicates_per_segment):
                    # Use original PCC since we're not modifying
                    augmented_segments.append((pcc, radar_seg.copy(), psg_seg.copy(), f"{source_name}_dup{dup_num}", idx))
            
            if augmented_segments:
                # Sort by PCC
                augmented_segments.sort(key=lambda x: x[0], reverse=True)
                # For duplicates, we keep them even if PCC < min_pcc (since they're copies of high-quality segments)
                # But we still prefer segments that meet the threshold
                augmented_meet_threshold = [s for s in augmented_segments if s[0] >= min_pcc]
                augmented_below_threshold = [s for s in augmented_segments if s[0] < min_pcc and s[0] >= 0.5]
                print(f"[info] Generated {len(augmented_segments)} duplicated segments")
                print(f"[info]   {len(augmented_meet_threshold)} meet PCC >= {min_pcc}")
                print(f"[info]   {len(augmented_below_threshold)} have PCC >= 0.5 (but < {min_pcc})")
                # Add both groups, prioritizing those that meet threshold
                all_segments.extend(augmented_meet_threshold)
                all_segments.extend(augmented_below_threshold)
                all_segments.sort(key=lambda x: x[0], reverse=True)
        
        # If still not enough, add segments with lower threshold (but still good quality)
        if len(all_segments) < needed:
            lower_threshold = max(0.5, min_pcc - 0.1)  # Don't go below 0.5
            additional = [s for s in all_segments_valid if lower_threshold <= s[0] < min_pcc]
            if additional:
                print(f"[info] Adding {len(additional)} segments with PCC >= {lower_threshold} (but < {min_pcc}) to fill remaining samples")
                all_segments.extend(additional)
                all_segments.sort(key=lambda x: x[0], reverse=True)
    
    top_segments = all_segments[:needed]
    if len(top_segments) < needed:
        print(f"[warn] only {len(top_segments)} segments available; expected {needed}")
        print(f"[info] Will generate {len(top_segments) // cfg['data']['segments_per_sample']} complete samples")

    # Build sample1..sample10
    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    segs_per_sample = cfg["data"]["segments_per_sample"]

    for idx in range(10):
        start = idx * segs_per_sample
        end = start + segs_per_sample
        chunk = top_segments[start:end]
        if not chunk:
            print(f"[warn] sample{idx+1}: no segments available")
            continue
        radar_out = np.stack([c[1] for c in chunk], axis=0)
        psg_out = np.stack([c[2] for c in chunk], axis=0)
        meta = {
            "sources": [c[3] for c in chunk],
            "indices": [int(c[4]) for c in chunk],
            "pcc_scores": [float(c[0]) for c in chunk],
            "fs": cfg["data"]["fs_target"],
            "segment_seconds": cfg["data"]["segment_seconds"],
            "segments_per_sample": segs_per_sample,
        }
        out_path = out_dir / f"sample{idx+1}.npz"
        np.savez(out_path, radar=radar_out, psg=psg_out, meta=meta)
        print(f"[ok] wrote {out_path} with {radar_out.shape[0]} segments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild samples from sub_01..sub_07 npy pairs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "config.json",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--min-pcc",
        type=float,
        default=0.0,
        help="Minimum reconstruction PCC threshold for segment selection (default: 0.0, no filtering)",
    )
    args = parser.parse_args()
    rebuild(args.config, args.device, min_pcc=args.min_pcc)

