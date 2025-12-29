import torch
import numpy as np


def add_defense_noise(audio, epsilon):
    """
    Add defense noise to audio with L-infinity constraint.
    
    Args:
        audio: Audio signal as torch.Tensor or numpy array
        epsilon: Maximum absolute value of noise (L-infinity bound)
        
    Returns:
        tuple: (defended_audio, noise) with same type and dtype as input
    """
    # Check if input is torch tensor
    is_torch = isinstance(audio, torch.Tensor)

    if is_torch:
        device = audio.device
        dtype = audio.dtype
        audio_np = audio.cpu().numpy()
    else:
        device = None
        dtype = None
        audio_np = audio

    # Generate random noise of the same shape as the audio
    noise = np.random.uniform(-1, 1, size=audio_np.shape)

    # Scale noise to respect the L-infinity constraint
    # First normalize to [-1, 1]
    max_abs_val = np.max(np.abs(noise))
    if max_abs_val > 0:  # Avoid division by zero
        noise = noise / max_abs_val

    # Then scale by epsilon to respect the L-infinity bound
    noise = noise * epsilon

    # Add the noise to the audio
    defended_audio = audio_np + noise

    # Clip to [-1, 1] to avoid distortion (assuming audio is normalized)
    defended_audio = np.clip(defended_audio, -1.0, 1.0)

    # Convert back to torch if needed, preserving dtype
    if is_torch:
        defended_audio = torch.from_numpy(defended_audio).to(dtype).to(device)
        noise = torch.from_numpy(noise).to(dtype).to(device)

    return defended_audio, noise