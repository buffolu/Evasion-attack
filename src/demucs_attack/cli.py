"""
CLI for Demucs adversarial attack and defense pipeline
using ground-truth stem folders as input.
"""

import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
import time

from demucs_attack.audio.io import load_audio, save_audio
from demucs_attack.model.demucs_model import load_demucs, separate
from demucs_attack.attack.optimizer import AdversarialAttack
from demucs_attack.defense.run_defense import add_defense_noise
from demucs_attack.utils.evaluations import compare_audio
from demucs_attack.utils.visualization import plot_all_metrics, create_summary_plot


SOURCE_NAMES = ["drums", "bass", "other", "vocals"]


def load_song_folder(song_dir: Path, device):
    """
    Load mixture + GT stems from a folder.
    """
    mixture, sr = load_audio(song_dir / "mixture.wav", device)

    gt_sources = {}
    for name in SOURCE_NAMES:
        audio, _ = load_audio(song_dir / f"{name}.wav", device)
        gt_sources[name] = audio

    return mixture, gt_sources, sr


def main():

    # get arguments
    parser = argparse.ArgumentParser(
        description="Demucs adversarial attack & defense (folder-based GT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input folder containing song subfolders"
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Configuration file"
    )
    args = parser.parse_args()



    #get input dirs
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    config_path = Path(args.config)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")


    #Start Pipeline
    start_time = time.time()

    print("=" * 60)
    print("DEMUCS ATTACK & DEFENSE PIPELINE (GT STEM MODE)")
    print("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = config["device"]
    print(f"Device: {device}")

    song_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"Found {len(song_dirs)} songs")

    model = load_demucs(config["model"], device)

    all_metrics = {}


    #iterate over each folder song.
    for song_dir in song_dirs:
        print(f"\n▶ Processing: {song_dir.name}")

        song_out = output_dir / song_dir.name
        song_out.mkdir(parents=True, exist_ok=True)

        mixture, gt_sources, sr = load_song_folder(song_dir, device)

        # ======================
        # Baseline separation - this separation is the normal outcome of demucs model.
        # ======================
        baseline_sources = separate(model, mixture, device, with_grad=False)


        """
        save everything in output directory / baseline / sources
        sources: bass, drums, other, vocals
        """
        baseline_dir = song_out / "baseline" / "sources"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        baseline_paths = {}
        for i, name in enumerate(SOURCE_NAMES):
            path = baseline_dir / f"{name}.wav"
            save_audio(path, baseline_sources[i], sr)
            baseline_paths[name] = str(path)


        # ======================
        # Attack - In this part I will preform the attack
        # ======================
        attack = AdversarialAttack(
            model=model,
            epsilon=config["attack"]["epsilon"],
            lr=config["attack"]["lr"],
            device=device,
        )

        sources_for_attack = {"mixture": mixture, **gt_sources}

        perturbation = attack.run(
            mixture,
            sources_for_attack,
            config["attack"]["iterations"],
        )

        perturbed_mixture = mixture + perturbation
        # Save to hear if perturbation is audible
        save_audio(song_out / "perturbed_mixture.wav", perturbed_mixture, sr)
        save_audio(song_out / "original_mixture.wav", mixture, sr)

        #separate song after attack and save results in output/song_name/attack
        attacked_sources = separate(model, perturbed_mixture, device, with_grad=False)

        attack_dir = song_out / "attack" / "sources"
        attack_dir.mkdir(parents=True, exist_ok=True)

        attack_paths = {}
        for i, name in enumerate(SOURCE_NAMES):
            path = attack_dir / f"{name}.wav"
            save_audio(path, attacked_sources[i], sr)
            attack_paths[name] = str(path)

        # ======================
        # Defense
        # ======================
        defended_audio, defense_noise = add_defense_noise(
            perturbed_mixture,
            config["defense"]["epsilon"]
        )

        defended_sources = separate(model, defended_audio, device, with_grad=False)

        defense_dir = song_out / "defense" / "sources"
        defense_dir.mkdir(parents=True, exist_ok=True)

        defense_paths = {}
        for i, name in enumerate(SOURCE_NAMES):
            path = defense_dir / f"{name}.wav"
            save_audio(path, defended_sources[i], sr)
            defense_paths[name] = str(path)

        # ======================
        # Evaluation (GT-based)
        # ======================
        metrics = {
            "ground_truth_vs_baseline": {},
            "ground_truth_vs_attack": {},
            "ground_truth_vs_defense": {},
        }

        for name in SOURCE_NAMES:
            metrics["ground_truth_vs_baseline"][name] = compare_audio(
                song_dir / f"{name}.wav",
                baseline_paths[name]
            )

            metrics["ground_truth_vs_attack"][name] = compare_audio(
                song_dir / f"{name}.wav",
                attack_paths[name]
            )

            metrics["ground_truth_vs_defense"][name] = compare_audio(
                song_dir / f"{name}.wav",
                defense_paths[name]
            )

        # ======================
        # Visualization - plot all the metrics (SDR,SIR,SDN)
        # ======================
        viz_dir = song_out / "visualizations"
        plot_all_metrics(metrics, viz_dir)
        create_summary_plot(metrics, viz_dir)

        all_metrics[song_dir.name] = metrics

        with open(song_out / "results_summary.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # ======================
    # Final summary
    # ======================
    summary_path = output_dir / "overall_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": round(time.time() - start_time, 2),
                "songs": list(all_metrics.keys()),
                "metrics": all_metrics,
            },
            f,
            indent=2,
        )

    print("\nPIPELINE COMPLETE")
    print(f"Results saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
