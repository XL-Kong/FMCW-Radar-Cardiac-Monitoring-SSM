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
  - `ssm_parameters.pth` – SSM parameters
  - `sub01.pt` – Beat-by-beat SSM model
  - `morphological_refiner.pth` – Morphological refiner model
  - `template.npy` – ECG template for beat matching

- `data/` – Contains `sub1.npz` through `sub10.npz` (beat-matched data samples).
- `outputs/` – Inference results (figures and npz files).

## Environment Setup

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

**Required packages and versions:**
- `torch==2.3.1` - PyTorch deep learning framework
- `torchvision==0.18.1` - Computer vision utilities for PyTorch
- `numpy==1.24.4` - Numerical computing
- `pandas==2.2.3` - Data manipulation and analysis
- `scipy==1.15.0` - Scientific computing (signal processing, filtering)
- `matplotlib==3.10.0` - Plotting and visualization

See `requirements.txt` for the complete list of dependencies.

## Data

For data protection and privacy reasons, the provided data files (`sub1.npz` through `sub10.npz`) contain **autoencoder-decoder output data**, not raw radar/ECG signals. The original raw data processing pipeline is provided as a reference file `RX2_processed.txt`, and the complete data processing workflow can be found in `ModelEval.ipynb`.

**Data Location**: `JournalSubmission/data/sub1.npz` through `sub10.npz`

**Data Processing Reference**: The radar and PSG data processing methods are based on the [SleepLab repository](https://github.com/XL-Kong/SleepLab). This repository will continue to be updated with additional processing utilities and examples.

Each sample file contains segments with the following structure:
- `radar_recon`: Radar signals (already processed through autoencoder-decoder)
- `psg`: PSG/ECG reference signals
- `radar_raw`: Original radar beat segments (for reference)
- `meta`: Metadata including sources, indices, and segment type

## Inference Pipeline

The inference pipeline consists of four stages:

1. **STFT Transformation**: Converts time-domain signals to frequency-time representation (64×64).
2. **ResNet Autoencoder**: Reconstructs ECG from radar STFT using pretrained weights.
3. **Beat-by-Beat SSM**: Processes reconstructed signal beat-by-beat.


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


## Configuration

All configuration parameters are defined in `configs/config.json`. Each parameter includes inline comments explaining its purpose and usage. Please refer to the configuration file for detailed parameter descriptions.
