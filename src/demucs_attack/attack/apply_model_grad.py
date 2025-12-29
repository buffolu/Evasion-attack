import torch
from demucs.apply import apply_model as demucs_apply_model
from contextlib import contextmanager




@contextmanager
def disable_no_grad():
    """
    Monkey Patch to bypass demucs internal code, this will allow gradients to flow backwards

    these are the lines in demucs that block the flow of gradient backwards.
    with th.no_grad():
        out = model(padded_mix)
    """
    original_no_grad = torch.no_grad
    torch.no_grad = torch.enable_grad
    try:
        yield
    finally:
        torch.no_grad = original_no_grad


def apply_model_grad(model, mix, device=None):
    with disable_no_grad():
        return demucs_apply_model(
            model,
            mix,
            device=device,
            split=False,
            overlap=0.25,
            shifts=0
        )
