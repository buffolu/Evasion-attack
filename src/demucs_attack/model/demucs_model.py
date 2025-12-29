import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model


def load_demucs(model_name: str, device: torch.device):
    """
    Load the standard (pretrained) Demucs model.
    """
    model = get_model(model_name)
    model.to(device)
    model.eval()
    return model


def separate(
        model,
        mixture: torch.Tensor,
        device: torch.device,
        with_grad: bool = False,
):
    """
    Run Demucs separation.

    mixture: [C, T] or [1, C, T]
    returns: [S, C, T]
    """
    if mixture.dim() == 2:
        mixture = mixture.unsqueeze(0)

    if with_grad:
        sources = apply_model(model, mixture, device=device)
    else:
        with torch.no_grad():
            sources = apply_model(model, mixture, device=device)

    return sources.squeeze(0)
