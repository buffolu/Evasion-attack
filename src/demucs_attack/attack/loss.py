import torch
import torch.nn.functional as F


def spectral_loss(
        estimated: torch.Tensor,
        target: torch.Tensor,
        device: torch.device,
        scales=(4096, 2048, 1024, 512, 256),
):
    loss = 0.0

    for scale in scales:
        if estimated.shape[-1] < scale:
            continue

        window = torch.hann_window(scale, device=device)

        est_spec = torch.stft(
            estimated,
            n_fft=scale,
            hop_length=scale // 4,
            window=window,
            return_complex=True,
        )

        tgt_spec = torch.stft(
            target,
            n_fft=scale,
            hop_length=scale // 4,
            window=window,
            return_complex=True,
        )

        loss += F.l1_loss(est_spec.abs(), tgt_spec.abs())

    return loss


def time_loss(estimated, target):
    return F.l1_loss(estimated, target)


def combined_source_loss(estimated, target, device):
    return -(
            0.8 * spectral_loss(estimated, target, device)
            + 0.2 * time_loss(estimated, target)
    )
