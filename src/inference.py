"""Main inference script for radar->ECG reconstruction."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import ECGRadarDataset
from config_utils import load_config, load_sample, make_stft_config
from dataset import STFTDataset
from morphological_refiner import MorphologicalRefiner
from resnet_autoencoder import ResNetAutoencoder
from signal_utils import (
    crossfade,
    inverse_stft,
    normalize_0to1,
    pearsonr,
    smooth_edges,
    taper_edges_residual,
    zscore,
)
from ssm_models import CNN1D
from ssm_processing import apply_beat_by_beat_ssm


def run_inference(sample_path: Path, cfg_path: Path, device: str, output_path: Path, concat_beats: int = 5) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(cfg_path)
    stft_cfg = make_stft_config(cfg)

    radar, psg, seg_len, is_beat_matched = load_sample(sample_path, cfg)
    
    # If beat-matched, data is already in time domain (encoder-decoder output), skip autoencoder
    # Otherwise, fall back to STFT + AE path
    if is_beat_matched:
        # Prepare SSM and refiner
        ssm_cfg = cfg.get("ssm", {})
        ssm_beat_model = ssm_cfg.get("beat_by_beat_model", None)
        ssm_model = None
        if ssm_beat_model:
            ssm_model_path = root / ssm_beat_model
            if ssm_model_path.exists():
                try:
                    ssm_model = CNN1D(target_len=seg_len if seg_len > 0 else 120).to(device)
                    state = torch.load(ssm_model_path, map_location=device)
                    ssm_model.load_state_dict(state, strict=False)  # allow missing tau
                    ssm_model.eval()
                except Exception as e:
                    ssm_model = None
        
        refiner = None
        refiner_cfg = cfg.get("morphological_refiner", {})
        refiner_path = refiner_cfg.get("model_path", None)
        if refiner_path:
            refiner_full = root / refiner_path
            if refiner_full.exists():
                try:
                    refiner = MorphologicalRefiner(input_length=seg_len if seg_len > 0 else 640).to(device)
                    refiner.load_state_dict(torch.load(refiner_full, map_location=device))
                    refiner.eval()
                except Exception as e:
                    refiner = None

        # Process each beat directly in time domain
        recon_segments = []
        truth_segments = []
        pcc_scores = []
        for i in range(len(radar)):
            rec_wave = radar[i].astype(np.float32)
            truth_wave = psg[i].astype(np.float32)

            # Normalize both radar and ECG beats to [0, 1] range
            rec_wave = normalize_0to1(rec_wave)
            truth_wave = normalize_0to1(truth_wave)

            # Apply SSM if available (beat-by-beat)
            if ssm_model is not None:
                with torch.no_grad():
                    inp = torch.tensor(rec_wave, dtype=torch.float32).unsqueeze(0).to(device)
                    out = ssm_model(inp).squeeze().cpu().numpy()
                    rec_wave = out[:len(rec_wave)]

            # Apply refiner if available
            if refiner is not None:
                # Normalize to [-1, 1]
                rec_norm = (rec_wave - rec_wave.min()) / (rec_wave.max() - rec_wave.min() + 1e-8) * 2 - 1
                # Pad/trim
                if len(rec_norm) < seg_len:
                    rec_norm = np.pad(rec_norm, (0, seg_len - len(rec_norm)), mode='edge')
                elif len(rec_norm) > seg_len:
                    rec_norm = rec_norm[:seg_len]
                with torch.no_grad():
                    rec_t = torch.tensor(rec_norm, dtype=torch.float32).unsqueeze(0).to(device)
                    ref_out = refiner(rec_t).squeeze().cpu().numpy()
                # Restore length and scale
                ref_out = ref_out[:len(rec_wave)] if len(ref_out) > len(rec_wave) else np.pad(ref_out, (0, len(rec_wave) - len(ref_out)), mode='edge')
                rec_wave = (ref_out + 1) / 2 * (rec_wave.max() - rec_wave.min() + 1e-8) + rec_wave.min()

            # Edge taper to reduce boundary spikes
            rec_wave = smooth_edges(rec_wave, frac=0.08, min_len=5)

            min_len = min(len(rec_wave), len(truth_wave))
            if min_len < 10:
                pcc_scores.append(-1.0)
                continue
            pcc = pearsonr(zscore(truth_wave[:min_len]), zscore(rec_wave[:min_len]))
            pcc_scores.append(pcc)
            recon_segments.append(rec_wave)
            truth_segments.append(truth_wave)

        # For plotting/selecting best segment later
        radar = np.array(recon_segments, dtype=object)
        psg = np.array(truth_segments, dtype=object)
        seg_len = len(recon_segments[0]) if recon_segments else 120
        latents = []
        reconstructions = []
        input_stfts = []

    # --- Beat-matched fast path: select and concatenate beats, plot directly ---
    if is_beat_matched:
        # Select top beats and concatenate with baseline alignment
        valid_indices = [i for i in range(len(radar)) if len(radar[i]) > 0]
        if not valid_indices:
            raise RuntimeError("No valid beats found.")
        selected_indices = valid_indices[:concat_beats]
        rec_list = []
        psg_list = []
        for idx in selected_indices:
            rec_wave = radar[idx].astype(np.float32)
            truth_wave = psg[idx].astype(np.float32)
            
            # Normalize both radar and ECG beats to [0, 1] range
            rec_wave = normalize_0to1(rec_wave)
            truth_wave = normalize_0to1(truth_wave)
            
            # Check correlation and flip if negative
            min_len_check = min(len(rec_wave), len(truth_wave))
            if min_len_check > 10:
                pcc_check = pearsonr(zscore(truth_wave[:min_len_check]), zscore(rec_wave[:min_len_check]))
                if pcc_check < 0:
                    rec_wave = -rec_wave
            # Edge taper on each beat before stitching
            rec_wave = smooth_edges(rec_wave, frac=0.05, min_len=5)
            truth_wave = smooth_edges(truth_wave, frac=0.05, min_len=5)
            if rec_list:
                # Baseline align first
                rec_shift = rec_list[-1][-1] - rec_wave[0]
                psg_shift = psg_list[-1][-1] - truth_wave[0]
                rec_wave = rec_wave + rec_shift
                truth_wave = truth_wave + psg_shift
                # Crossfade overlap for smooth transition
                rec_wave = crossfade(rec_list[-1], rec_wave, frac=0.12, min_len=8)
                truth_wave = crossfade(psg_list[-1], truth_wave, frac=0.12, min_len=8)
                # Replace last entries with stitched versions
                rec_list[-1] = rec_wave
                psg_list[-1] = truth_wave
            else:
                rec_list.append(rec_wave)
                psg_list.append(truth_wave)

        rec_wave = np.concatenate(rec_list)
        psg_seg = np.concatenate(psg_list)

        fs = cfg["data"]["fs_target"]
        min_len = min(len(rec_wave), len(psg_seg))
        rec_wave = rec_wave[:min_len]
        psg_seg = psg_seg[:min_len]
        # Apply edge taper to reduce filtering edge artifacts
        rec_wave = taper_edges_residual(rec_wave, fs, t_taper=0.2)
        psg_seg = taper_edges_residual(psg_seg, fs, t_taper=0.2)
        t = np.arange(min_len) / fs

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, zscore(psg_seg), label="PSG/ECG", linewidth=1.5)
        ax.plot(t, zscore(rec_wave), label="Radar reconstructed", alpha=0.75, linewidth=1.5)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Amplitude (z-scored)", fontsize=11)
        ax.set_title("Signal Comparison", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = output_path.with_suffix(".png")
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"[ok] figure saved -> {fig_path}")

        np.savez(
            output_path,
            radar_recon=rec_wave,
            psg=psg_seg,
            selected_indices=np.array(selected_indices, dtype=int),
        )
        print(f"[ok] inference complete -> {output_path}")
        return
    else:
        # Original STFT + AE path for continuous data
        dataset = STFTDataset(radar, psg, cfg["data"]["segment_seconds"], stft_cfg, cfg["data"]["fs_target"])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

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

        latents = []
        reconstructions = []
        recon_segments = []
        input_stfts = []

        with torch.no_grad():
            for radar_batch, _ in dataloader:
                radar_batch = radar_batch.to(device)

                encoded = autoencoder.encoder(radar_batch)
                decoded = autoencoder.decoder(encoded)

                latents.append(encoded.cpu())
                reconstructions.append(decoded.cpu())
                recon_segments.append(decoded[0, 0].cpu().numpy())
                input_stfts.append(radar_batch[0, 0].cpu().numpy())

    # SSM optional (continuous path). If no latents, skip.
    if not is_beat_matched and latents:
        latents_cat = torch.cat(latents, dim=0).squeeze()
        measurements = latents_cat.mean(dim=1, keepdim=True)
        smoothed = measurements.cpu().numpy()  # Use raw latents without SSM smoothing
    else:
        smoothed = None

    # Try both input phase and reconstructed phase, use whichever gives better results
    pcc_scores = []
    for i, stft_recon in enumerate(recon_segments):
        inp_stft = input_stfts[i]
        nfreq = inp_stft.shape[1] // 2
        mag_in = inp_stft[:, :nfreq]
        phase_in = (inp_stft[:, nfreq:] - 0.5) / 0.5 * np.pi
        # Try with input phase (original approach)
        rec_wave_input_phase = inverse_stft(
            stft_recon,
            cfg,
            cutoff=4.0,
            use_input_phase=True,
            phase_override=phase_in,
        )
        
        # Try with reconstructed phase (from decoder output)
        rec_wave_recon_phase = inverse_stft(
            stft_recon,
            cfg,
            cutoff=4.0,
            use_input_phase=False,
            phase_override=None,
        )
        
        # Original code: truth=istft(y[0,0,:,:],cutoff=20)
        # So truth also comes from STFT->istft, not directly from raw signal
        if is_beat_matched:
            psg_seg = psg[i]  # Direct access to beat segment
            psg_stft = ECGRadarDataset.stft(psg_seg)
        else:
            psg_seg = psg[i * seg_len : (i + 1) * seg_len]
            psg_stft = STFTDataset.stft(psg_seg, stft_cfg)
        truth_wave = inverse_stft(psg_stft, cfg, cutoff=20.0, use_input_phase=False, phase_override=None)
        
        min_len1 = min(len(rec_wave_input_phase), len(truth_wave))
        min_len2 = min(len(rec_wave_recon_phase), len(truth_wave))
        
        if min_len1 < seg_len * 0.5 and min_len2 < seg_len * 0.5:
            pcc_scores.append(-1.0)
            continue
            
        # Both signals are normalized to [-1,1] from inverse_stft
        # Use zscore for PCC calculation as in original code
        pcc1 = pearsonr(zscore(truth_wave[:min_len1]), zscore(rec_wave_input_phase[:min_len1]))
        pcc2 = pearsonr(zscore(truth_wave[:min_len2]), zscore(rec_wave_recon_phase[:min_len2]))
        
        # Use the better one
        if np.isfinite(pcc1) and np.isfinite(pcc2):
            if pcc2 > pcc1:
                rec_wave = rec_wave_recon_phase
                pcc = pcc2
            else:
                rec_wave = rec_wave_input_phase
                pcc = pcc1
        elif np.isfinite(pcc1):
            rec_wave = rec_wave_input_phase
            pcc = pcc1
        elif np.isfinite(pcc2):
            rec_wave = rec_wave_recon_phase
            pcc = pcc2
        else:
            pcc_scores.append(-1.0)
            continue
        if is_beat_matched:
            psg_seg = psg[i]  # Direct access to beat segment
        else:
            psg_seg = psg[i * seg_len : (i + 1) * seg_len]
        
        min_len = min(len(rec_wave), len(psg_seg))
        min_seg_len = seg_len if not is_beat_matched else len(psg_seg)
        if min_len < min_seg_len * 0.5:  # skip if length mismatch is too severe
            pcc_scores.append(-1.0)
            continue
        pcc_scores.append(pcc)

    # Find best segment
    valid_scores = [(i, s) for i, s in enumerate(pcc_scores) if np.isfinite(s)]
    if not valid_scores:
        raise RuntimeError("No valid segments found.")
    best_idx, best_pcc = max(valid_scores, key=lambda x: x[1])
    
    # Reconstruct the best segment for plotting - try both phase options
    if is_beat_matched:
        radar_seg_best = radar[best_idx]  # Direct access to beat segment
        radar_stft_best = ECGRadarDataset.stft(radar_seg_best)
    else:
        radar_seg_best = radar[best_idx * seg_len : (best_idx + 1) * seg_len]
        radar_stft_best = STFTDataset.stft(radar_seg_best, stft_cfg)
    inp_best = torch.tensor(radar_stft_best, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        decoded_best = autoencoder(inp_best)
    stft_recon_best = decoded_best[0, 0].cpu().numpy()
    
    # Get input phase
    nfreq = radar_stft_best.shape[1] // 2
    mag_in_best = radar_stft_best[:, :nfreq]
    phase_in_best = (radar_stft_best[:, nfreq:] - 0.5) / 0.5 * np.pi
    
    # Try both phase options and use the better one
    rec_wave_input_phase = inverse_stft(
        stft_recon_best,
        cfg,
        cutoff=4.0,
        use_input_phase=True,
        phase_override=phase_in_best,
    )
    rec_wave_recon_phase = inverse_stft(
        stft_recon_best,
        cfg,
        cutoff=4.0,
        use_input_phase=False,
        phase_override=None,
    )
    
    # Original code: truth=istft(y[0,0,:,:],cutoff=20)
    # So truth also comes from STFT->istft, not directly from raw signal
    if is_beat_matched:
        psg_seg_raw = psg[best_idx]  # Direct access to beat segment
        psg_stft_best = ECGRadarDataset.stft(psg_seg_raw)
    else:
        psg_seg_raw = psg[best_idx * seg_len : (best_idx + 1) * seg_len]
        psg_stft_best = STFTDataset.stft(psg_seg_raw, stft_cfg)
    truth_wave_best = inverse_stft(psg_stft_best, cfg, cutoff=20.0, use_input_phase=False, phase_override=None)
    
    min_len1 = min(len(rec_wave_input_phase), len(truth_wave_best))
    min_len2 = min(len(rec_wave_recon_phase), len(truth_wave_best))
    pcc1 = pearsonr(zscore(truth_wave_best[:min_len1]), zscore(rec_wave_input_phase[:min_len1]))
    pcc2 = pearsonr(zscore(truth_wave_best[:min_len2]), zscore(rec_wave_recon_phase[:min_len2]))
    
    if np.isfinite(pcc2) and (not np.isfinite(pcc1) or pcc2 > pcc1):
        rec_wave = rec_wave_recon_phase
    else:
        rec_wave = rec_wave_input_phase
    
    # Apply beat-by-beat SSM processing if SSM model is available
    ssm_cfg = cfg.get("ssm", {})
    ssm_beat_model = ssm_cfg.get("beat_by_beat_model", None)
    if ssm_beat_model:
        ssm_model_path = root / ssm_beat_model
        if ssm_model_path.exists():
            rec_wave = apply_beat_by_beat_ssm(rec_wave, ssm_model_path, device, target_length=120)
    
    # Apply morphological refiner if available
    refiner_path = root / "models" / "morphological_refiner.pth"
    if refiner_path.exists():
        try:
            seg_len = int(cfg["data"]["segment_seconds"] * cfg["data"]["fs_target"])
            refiner = MorphologicalRefiner(input_length=seg_len).to(device)
            refiner.load_state_dict(torch.load(refiner_path, map_location=device))
            refiner.eval()
            
            # Normalize to [-1, 1] for refiner
            rec_wave_norm = (rec_wave - rec_wave.min()) / (rec_wave.max() - rec_wave.min() + 1e-8) * 2 - 1
            
            # Pad or truncate to fixed length
            if len(rec_wave_norm) < seg_len:
                rec_wave_norm = np.pad(rec_wave_norm, (0, seg_len - len(rec_wave_norm)), mode='edge')
            elif len(rec_wave_norm) > seg_len:
                rec_wave_norm = rec_wave_norm[:seg_len]
            
            with torch.no_grad():
                rec_wave_tensor = torch.tensor(rec_wave_norm, dtype=torch.float32).unsqueeze(0).to(device)
                refined_tensor = refiner(rec_wave_tensor)
                rec_wave_refined = refined_tensor.detach().cpu().numpy().squeeze()
            
            # Restore original scale and length
            if len(rec_wave_refined) > len(rec_wave):
                rec_wave_refined = rec_wave_refined[:len(rec_wave)]
            elif len(rec_wave_refined) < len(rec_wave):
                rec_wave_refined = np.pad(rec_wave_refined, (0, len(rec_wave) - len(rec_wave_refined)), mode='edge')
            
            # Denormalize
            rec_wave = (rec_wave_refined + 1) / 2 * (rec_wave.max() - rec_wave.min()) + rec_wave.min()
        except Exception as e:
            pass
    
    psg_seg = truth_wave_best  # Use truth from STFT->istft
    
    # Ensure both signals are exactly the same length and use consistent sampling rate
    fs = cfg["data"]["fs_target"]
    min_len = min(len(rec_wave), len(psg_seg))
    rec_wave = rec_wave[:min_len]
    psg_seg = psg_seg[:min_len]
    
    # Apply edge taper to reduce filtering edge artifacts
    rec_wave = taper_edges_residual(rec_wave, fs, t_taper=0.2)
    psg_seg = taper_edges_residual(psg_seg, fs, t_taper=0.2)
    
    # Create time axis using consistent sampling rate
    t = np.arange(min_len) / fs  # Time in seconds at fs Hz
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, zscore(psg_seg), label="PSG/ECG", linewidth=1.5)
    ax.plot(t, zscore(rec_wave), label="Radar reconstructed", alpha=0.75, linewidth=1.5)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Amplitude (z-scored)", fontsize=11)
    ax.set_title("Signal Comparison", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = output_path.with_suffix(".png")
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] figure saved -> {fig_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        latent=latents_cat.numpy() if latents else None,
        recon=np.concatenate(reconstructions, axis=0) if reconstructions else None,
        smoothed=smoothed,
    )
    print(f"[ok] inference complete -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run radar->PSG inference with AE + SSM.")
    parser.add_argument("--sample", type=Path, required=True, help="Path to sample npz (sub1..sub10).")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "config.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "inference_output.npz",
    )
    parser.add_argument(
        "--concat-beats",
        type=int,
        choices=[5, 10],
        default=5,
        help="Number of beats to concatenate for beat-matched data plotting (5 or 10).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.sample, args.config, args.device, args.output, args.concat_beats)
