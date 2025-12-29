"""
Audio evaluation utilities for source separation metrics.
Compatible with the Demucs attack/defense pipeline.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Union

# Try to import evaluation libraries
try:
    import museval
    USE_MUSEVAL = True
except ImportError:
    from mir_eval.separation import bss_eval_sources
    USE_MUSEVAL = False


def compare_audio(
        reference_path: Union[str, Path],
        estimated_path: Union[str, Path]
) -> Dict[str, float]:
    """
    Compare two audio files and return evaluation metrics.

    Args:
        reference_path: Path to reference (ground truth) audio
        estimated_path: Path to estimated (predicted) audio

    Returns:
        Dictionary containing SDR, SIR, SAR metrics
    """
    # Load audio files
    ref_audio, ref_sr = sf.read(str(reference_path))
    est_audio, est_sr = sf.read(str(estimated_path))

    # Convert to mono if stereo
    if len(ref_audio.shape) > 1:
        ref_audio = np.mean(ref_audio, axis=1)
    if len(est_audio.shape) > 1:
        est_audio = np.mean(est_audio, axis=1)

    # Ensure same length
    min_len = min(len(ref_audio), len(est_audio))
    ref_audio = ref_audio[:min_len]
    est_audio = est_audio[:min_len]

    # Check for silent sources
    ref_energy = np.sum(ref_audio ** 2)
    est_energy = np.sum(est_audio ** 2)

    # If either source is silent, return NaN
    if ref_energy < 1e-10 or est_energy < 1e-10:
        return {
            "SDR": float('nan'),
            "SIR": float('nan'),
            "SAR": float('nan')
        }

    # Reshape for evaluation (2D arrays: n_sources x n_samples)
    reference = ref_audio.reshape(1, -1)
    estimated = est_audio.reshape(1, -1)

    # Compute BSS eval metrics
    if USE_MUSEVAL:
        scores = museval.evaluate(reference, estimated)
        sdr = float(np.median(scores[0]['SDR']))
        sir = float(np.median(scores[0]['SIR']))
        sar = float(np.median(scores[0]['SAR']))
    else:
        sdr, sir, sar, _ = bss_eval_sources(reference, estimated, compute_permutation=False)
        sdr = float(sdr[0])
        sir = float(sir[0])
        sar = float(sar[0])

    return {
        "SDR": sdr,
        "SIR": sir,
        "SAR": sar
    }