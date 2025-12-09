# FMCW-Radar-Cardiac-Monitoring-SSM

This repository provides the official implementation of the cardiopulmonary signal reconstruction framework proposed in the following manuscript:
X. Kong et al., IEEE Transactions on Instrumentation and Measurement (TIM), 2025.
(Link will be added upon publication.)

## Structure

- `configs/config.json` – Configuration file containing radar/PSG parameters (resampling, filters, STFT), model hyperparameters, and data sources.
- `src/inference.py` – Main inference script (STFT → ResNet AE → Beat-by-beat SSM → Morphological Refiner).
- `src/config_utils.py` – Configuration loading and data loading utilities.
- `src/signal_utils.py` – Signal processing utilities (normalization, STFT inverse, edge tapering, etc.).
- `src/ssm_models.py` – SSM model definitions (StrictSSM, CNN1D).
- `src/ssm_processing.py` – SSM processing utilities for beat-by-beat signal processing.
- `src/dataset.py` – STFTDataset for continuous signal processing.
- `src/beat_matched_dataset.py` – BeatMatchedDataset for beat-aligned signal processing.
- `src/resnet_autoencoder.py` – ResNet-based autoencoder implementation.
- `src/morphological_refiner.py` – CNN-based morphological refiner for ECG feature enhancement.
- `src/tools.py` – Utility functions (smoothing, filtering).
- `models/` – Contains:
  - `model_2024_08_23__162411.pth` – Pretrained ResNet autoencoder
  - `sub01.pt` – Beat-by-beat SSM model (CNN1D)
  - `morphological_refiner.pth` – Morphological refiner model
  - `template.npy` – ECG template for beat matching
  - `ssm_parameters.pth` – Legacy SSM parameters (not used in current pipeline)
- `data/` – Contains `sub1.npz` through `sub10.npz` (beat-matched data samples).
- `outputs/` – Inference results (figures and npz files).

## Environment Setup

- Use the conda environment `torch2`:
  ```bash
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate torch2
  ```
- Core packages: `torch`, `torchvision`, `torchaudio`, `numpy`, `pandas`, `scipy`, `opencv-python`, `matplotlib`, `tqdm`.

## Data

For data protection and privacy reasons, the provided data files (`sub1.npz` through `sub10.npz`) contain **autoencoder-decoder output data**, not raw radar/ECG signals. The original raw data processing pipeline is provided as a reference file `RX2_processed.txt`, and the complete data processing workflow can be found in `ModelEval.ipynb`.

**Data Location**: `JournalSubmission/data/sub1.npz` through `sub10.npz`

Each sample contains beat-matched segments with the following structure:
- `radar_recon`: Reconstructed radar signals (already processed through autoencoder-decoder)
- `psg`: PSG/ECG reference signals
- `radar_raw`: Original radar beat segments (for reference)
- `meta`: Metadata including sources, indices, and segment type

## Inference Pipeline

The inference pipeline consists of four stages:

1. **STFT Transformation**: Converts time-domain signals to frequency-time representation (64×64).
2. **ResNet Autoencoder**: Reconstructs ECG from radar STFT using pretrained weights.
3. **Beat-by-Beat SSM**: Processes reconstructed signal beat-by-beat using CNN1D model (`sub01.pt`).
4. **Morphological Refiner**: Refines morphological features (peaks, slopes) using CNN refiner.

### Run Inference

Example on `sub1`:
```bash
python JournalSubmission/src/inference.py \
  --sample JournalSubmission/data/sub1.npz \
  --output JournalSubmission/outputs/sub1_infer.npz
```

**Output**:
- `sub1_infer.npz` – Contains reconstructed signals and metadata
- `sub1_infer.png` – Comparison plot: PSG/ECG vs. Radar reconstructed signal

### Inference Features

- **Automatic Data Type Detection**: Handles both beat-matched and continuous data formats
- **Dual Phase Reconstruction**: Tries both input phase and reconstructed phase, selects the better one
- **Beat-by-Beat SSM**: Automatically applied if `sub01.pt` is available
- **Morphological Refinement**: Automatically applied if `morphological_refiner.pth` is available

## Configuration

Key parameters in `configs/config.json`:

- **Data**:
  - `fs_target`: 128 Hz (target sampling rate)
  - `segment_seconds`: 5 (for continuous data; beat-matched uses variable beat lengths)
  - `segments_per_sample`: 60 (for continuous data; beat-matched uses ~36 beats)

- **STFT**:
  - `window`: "hamming"
  - `window_length`: 16
  - `noverlap`: 4
  - `nfft`: 64

- **Filters**:
  - `radar_bandpass`: [0.4, 5.0] Hz
  - `ecg_bandpass`: [0.3, 30.0] Hz
  - `order`: 4

- **Models**:
  - `resnet_autoencoder.pretrained_path`: "models/model_2024_08_23__162411.pth"
  - `ssm.beat_by_beat_model`: "models/sub01.pt"
  - `morphological_refiner.model_path`: "models/morphological_refiner.pth"
