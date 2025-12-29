import os
import sys
import argparse

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from demucs.pretrained import get_model
from demucs.apply import apply_model
from apply_model_grad import apply_model_grad


# -------------------------
# CUDA CHECK (HARD REQUIREMENT)
# -------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is REQUIRED for this script. No CPU fallback is allowed.")

DEVICE = torch.device("cuda")


# -------------------------
# LOSS FUNCTION
# -------------------------
def compute_loss(model, perturbed_audio, target_sources):
    """
    CUDA-only loss computation
    """

    # Ensure batch dimension
    if perturbed_audio.dim() == 2:
        perturbed_audio = perturbed_audio.unsqueeze(0)

    # Forward pass (with gradients)
    estimated_sources = apply_model_grad(
        model,
        perturbed_audio,
        device=DEVICE
    ).squeeze(0)  # [4, 2, samples]

    # Collect target sources (exclude mixture)
    target_tensors = [
        src for k, src in target_sources.items() if k != "mixture"
    ]

    def spectral_loss(estimated, target, scales=(4096, 2048, 1024, 512, 256)):
        loss = 0.0
        for scale in scales:
            if estimated.shape[-1] < scale:
                continue

            window = torch.hann_window(scale, device=DEVICE)

            est_spec = torch.stft(
                estimated,
                n_fft=scale,
                hop_length=scale // 4,
                window=window,
                return_complex=True
            )

            tgt_spec = torch.stft(
                target,
                n_fft=scale,
                hop_length=scale // 4,
                window=window,
                return_complex=True
            )

            loss += F.l1_loss(
                torch.abs(est_spec),
                torch.abs(tgt_spec),
                reduction="mean"
            )

        return loss

    def time_loss(estimated, target):
        return F.l1_loss(estimated, target, reduction="mean")

    total_loss = 0.0
    for est, tgt in zip(estimated_sources, target_tensors):
        total_loss += (
                0.8 * spectral_loss(est, tgt)
                + 0.2 * time_loss(est, tgt)
        )

    # Maximize degradation → return negative loss
    return -total_loss


# -------------------------
# ADVERSARIAL ATTACK
# -------------------------
def run_adversarial_attack(
        model,
        sources,
        iterations=5000,
        epsilon=0.00035,
        lr=0.01,
):
    print("=== CUDA Adversarial Attack ===")
    print(f"Iterations: {iterations}")
    print(f"Epsilon: {epsilon}")
    print(f"Learning rate: {lr}")
    print(f"Device: {DEVICE}")

    # Mixture (GPU)
    audio = sources["mixture"].detach()

    # Perturbation (GPU, trainable)
    perturbation = torch.zeros_like(audio, device=DEVICE, requires_grad=True)

    optimizer = torch.optim.Adam([perturbation], lr=lr)

    best_loss = float("inf")
    best_perturbation = None

    for i in tqdm(range(iterations)):
        optimizer.zero_grad()

        # Clamp perturbation
        perturbation.data.clamp_(-epsilon, epsilon)

        perturbed_audio = audio + perturbation

        loss = compute_loss(model, perturbed_audio, sources)
        loss.backward()
        optimizer.step()

        perturbation.data.clamp_(-epsilon, epsilon)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_perturbation = perturbation.detach().clone()
            print(f"[+] New best loss: {best_loss:.6f}")

        if (i + 1) % 1000 == 0:
            print(f"Iter {i+1}/{iterations} | Loss: {loss.item():.6f}")

    return best_perturbation


# -------------------------
# ARGUMENTS
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser("CUDA-only Demucs adversarial attack")

    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", default="./attack_results")
    parser.add_argument("--model", default="htdemucs")

    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--epsilon", type=float, default=0.0005)
    parser.add_argument("--lr", type=float, default=0.5)

    return parser.parse_args()


# -------------------------
# MAIN
# -------------------------
def main():
    args = parse_args()

    song_name = os.path.splitext(os.path.basename(args.input_file))[0]
    song_dir = os.path.join(args.output_dir, song_name)

    os.makedirs(song_dir, exist_ok=True)
    originals_dir = os.path.join(song_dir, "separation_prior_attack")
    estimated_dir = os.path.join(song_dir, "separation_after_attack")

    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(estimated_dir, exist_ok=True)

    print(f"Song: {song_name}")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    # Load model (GPU)
    model = get_model(args.model).to(DEVICE).eval()

    # Load mixture (CPU → GPU)
    mixture_cpu, sr = torchaudio.load(args.input_file)
    mixture = mixture_cpu.to(DEVICE)

    torchaudio.save(
        os.path.join(originals_dir, "mixture.wav"),
        mixture_cpu,
        sr
    )

    # Original separation (GPU)
    with torch.no_grad():
        srcs = apply_model(
            model,
            mixture.unsqueeze(0),
            device=DEVICE
        ).squeeze(0)

    names = ["drums", "bass", "other", "vocals"]

    sources = {"mixture": mixture}
    for i, name in enumerate(names):
        sources[name] = srcs[i]
        torchaudio.save(
            os.path.join(originals_dir, f"{name}.wav"),
            srcs[i].cpu(),
            sr
        )

    # Run attack
    perturbation = run_adversarial_attack(
        model,
        sources,
        iterations=args.iterations,
        epsilon=args.epsilon,
        lr=args.lr,
    )

    # Save perturbation & perturbed mixture
    perturbed = (mixture + perturbation).detach()

    torchaudio.save(
        os.path.join(song_dir, "perturbed_mixture.wav"),
        perturbed.cpu(),
        sr
    )

    torchaudio.save(
        os.path.join(song_dir, "final_perturbation.wav"),
        perturbation.cpu(),
        sr
    )

    # Separate perturbed audio
    with torch.no_grad():
        pert_srcs = apply_model(
            model,
            perturbed.unsqueeze(0),
            device=DEVICE
        ).squeeze(0)

    for i, name in enumerate(names):
        torchaudio.save(
            os.path.join(estimated_dir, f"{name}.wav"),
            pert_srcs[i].cpu(),
            sr
        )

    print("Attack completed successfully.")
    print(f"Results saved to: {song_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
