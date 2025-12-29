# Demucs Adversarial Attack & Defense

Adversarial attack and defense pipeline for the Demucs music source separation model.

## Overview

This project implements adversarial attacks on the Demucs source separation model and evaluates defense mechanisms. It processes songs from the MUSDB18 dataset, performs adversarial perturbations, and evaluates separation quality using SDR, SIR, and SAR metrics.

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg (required for MUSDB18 processing)

## Installation

### Option 1: Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/buffolu/Evasion-attack
cd Evasion-attack

# Create virtual environment with UV
uv venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install the package in editable mode
uv pip install -e .

# Install evaluation library (choose one)
uv pip install museval  # Recommended
# OR
uv pip install mir_eval  # Legacy alternative
```

### Option 2: Using Standard venv

```bash
# Clone the repository
git clone <repository-url>
cd Evasion-attack

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install the package in editable mode
pip install -e .

# Install evaluation library (choose one)
pip install museval  # Recommended
# OR
pip install mir_eval  # Legacy alternative
```

## Configuration

Edit `config/default.yaml` to customize attack/defense parameters:

```yaml
model: htdemucs
device: cuda

attack:
  iterations: 4000
  epsilon: 0.01      # Perturbation magnitude (0.001-0.1)
  lr: 0.001          # Learning rate

defense:
  epsilon: 0.003     # Defense noise magnitude
```

### Parameter Guidelines

- **epsilon**: Controls perturbation strength
  - `0.001-0.005`: Minimal audible artifacts, weaker attack
  - `0.01-0.05`: Noticeable degradation, stronger attack
  - `0.05+`: Severe degradation with audible noise

- **iterations**: Number of attack iterations (2000-5000 typical)

- **lr**: Learning rate for perturbation optimization (0.0001-0.001)

## Usage

Run the complete attack & defense pipeline:

```bash
evasion-attack --input path/to/input --output path/to/outputs --config config/default.yaml
```

### Arguments

- `--input`: Path to input folder containing song subfolders (required)
- `--output`: Output directory (default: `outputs`)
- `--config`: Configuration file (default: `config/default.yaml`)

### Example

```bash
# Process all songs in the input folder
evasion-attack --input C:\data\input --output C:\data\outputs

# Use custom config
evasion-attack --input ./input --output ./results --config config/custom.yaml
```

## Output Structure

The pipeline generates the following structure for each song:

```
outputs/
└── Song Name/
    ├── baseline/
    │   └── sources/
    │       ├── drums.wav
    │       ├── bass.wav
    │       ├── other.wav
    │       └── vocals.wav
    ├── attack/
    │   └── sources/
    │       ├── drums.wav
    │       ├── bass.wav
    │       ├── other.wav
    │       └── vocals.wav
    ├── defense/
    │   └── sources/
    │       ├── drums.wav
    │       ├── bass.wav
    │       ├── other.wav
    │       └── vocals.wav
    ├── visualizations/
    │   ├── sdr_comparison.png
    │   ├── sir_comparison.png
    │   ├── sar_comparison.png
    │   └── summary.png
    └── results_summary.json
```

## Evaluation Metrics

The pipeline computes three standard BSS eval metrics:

- **SDR (Signal-to-Distortion Ratio)**: Overall separation quality
- **SIR (Signal-to-Interference Ratio)**: Interference from other sources
- **SAR (Signal-to-Artifacts Ratio)**: Processing artifacts

Higher values indicate better separation quality.

## Project Structure

```
Evasion-attack/
├── src/
│   └── demucs_attack/
│       ├── cli.py                    # Main CLI entry point
│       ├── model/
│       │   └── demucs_model.py       # Demucs model wrapper
│       ├── attack/
│       │   ├── optimizer.py          # Adversarial attack
│       │   ├── loss.py               # Loss functions
│       │   └── apply_model_grad.py   # Gradient computation
│       ├── defense/
│       │   └── run_defense.py        # Defense mechanisms
│       ├── audio/
│       │   └── io.py                 # Audio loading/saving
│       └── utils/
│           ├── evaluations.py        # BSS eval metrics
│           ├── visualization.py      # Plotting functions
│             
├── config/
│   └── default.yaml                  # Configuration file
├── scripts/
│   └── musdb18_cutter.py            # Dataset preparation
├── pyproject.toml                    # Package configuration
└── README.md                         # This file
```

## Acknowledgments

- Demucs model: https://github.com/facebookresearch/demucs
- MUSDB18 dataset: https://sigsep.github.io/datasets/musdb.html
