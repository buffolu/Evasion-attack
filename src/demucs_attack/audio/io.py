import torchaudio
import torch


def load_audio(path: str, device: torch.device | None = None):
    """
    Load audio from disk.

    Returns:
        waveform: Tensor [C, T]
        sample_rate: int
    """
    waveform, sr = torchaudio.load(path)

    if device is not None:
        waveform = waveform.to(device)

    return waveform, sr


def save_audio(path: str, waveform: torch.Tensor, sample_rate: int):
    """
    Save audio to disk.
    """
    torchaudio.save(
        path,
        waveform.detach().cpu(),
        sample_rate,
    )
