import os
import torch
import torch.nn.functional as F
import torchaudio
import argparse
from tqdm import tqdm
from demucs.pretrained import get_model
from demucs.apply import apply_model
import sys

def compute_loss(model, perturbed_audio, target_sources, device):
    """
    Compute loss for attack using spectral and time loss combined
    
    Args:
        model: Demucs model instance
        perturbed_audio: Perturbed audio tensor
        target_sources: Dictionary of original source tensors
        device: Computation device (GPU RECOMMENDED)
        
    Returns:
        torch.Tensor: Loss value (negative)
    """
    # Ensure perturbed audio is on the correct device
    perturbed_audio = perturbed_audio.to(device)
    
    # Extract source tensors from dictionary, excluding mixture
    source_tensors = []
    for key, source in target_sources.items():
        if key != 'mixture':
            source_tensors.append(source.to(device))
    
    # Ensure perturbed_audio has batch dimension
    if perturbed_audio.dim() == 2:
        perturbed_audio = perturbed_audio.unsqueeze(0)
    
    # Forward pass through Demucs
    estimated_sources = apply_model(model, perturbed_audio, device=device)
    estimated_sources = estimated_sources.squeeze(0)  # Now shape [4, 2, samples]
    
    # Calculate spectral loss for each source and scale
    def spectral_loss(estimated, target, scales=[4096, 2048, 1024, 512, 256], device=None):
        if device is None:
            device = estimated.device
        
        loss = 0
        for scale in scales:
            # Ensure scale doesn't exceed signal length
            if estimated.shape[-1] < scale:
                continue
                
            # Compute STFT for both signals
            est_spec = torch.stft(
                estimated, 
                n_fft=scale, 
                hop_length=scale//4,
                window=torch.hann_window(scale).to(device),
                return_complex=True
            )
            
            target_spec = torch.stft(
                target, 
                n_fft=scale, 
                hop_length=scale//4,
                window=torch.hann_window(scale).to(device),
                return_complex=True
            )
            
            # Compute L1 loss on magnitude spectrograms
            loss += F.l1_loss(torch.abs(est_spec), torch.abs(target_spec), reduction='mean')
            
        return loss

    def time_loss(estimated, target):
        return F.l1_loss(estimated, target, reduction='mean')
    
    # Compute total loss across all sources
    total_loss = 0
    for i, (est_source, target_source) in enumerate(zip(estimated_sources, source_tensors)):
        # Compute losses for this source
        spec_loss_val = spectral_loss(est_source, target_source, device=device)
        time_loss_val = time_loss(est_source, target_source)
        
        # Combine with weights
        source_loss = 0.8 * spec_loss_val + 0.2 * time_loss_val
        total_loss += source_loss
    
    # For adversarial attack, we want to maximize the loss (minimize quality)
    # So we return negative loss (higher values = better attack)
    return -total_loss

    



def run_adversarial_attack(model, sources, output_dir, iterations=5000, epsilon=0.00035, lr=0.01, device="cuda"):
    """
    Run adversarial attack on the full song to degrade separation quality.
    
    Args:
        model: Demucs model
        sources: Dictionary of source tensors
        output_dir: Directory to save results
        iterations: Number of optimization iterations
        epsilon: L-infinity constraint for the perturbation
        lr: Learning rate for optimization
        device: Computation device
        
    Returns:
        torch.Tensor: Optimized perturbation
    """
    print(f"Starting attack for {iterations} iterations...")
    print(f'epsilon: {epsilon}')
    print(f'lr: {lr}')
    print(f'device: {device}')
    
    # Extract and move audio to specified device
    audio = sources["mixture"].to(device)
    
    # Ensure all sources are on the same device
    device_sources = {}
    for key, tensor in sources.items():
        device_sources[key] = tensor.to(device)
    
    # Detach the audio to ensure we don't compute gradients through it
    audio = audio.detach()
    
    # Create perturbation with gradients enabled on the same device
    perturbation = torch.zeros_like(audio, device=device)
    perturbation.requires_grad_()
    
    # Create optimizer
    optimizer = torch.optim.Adam([perturbation], lr=lr)
    
    # Track metrics

    best_loss = float('inf')
    best_perturbation = None
    
    # Attack loop
    for i in tqdm(range(iterations)):

        optimizer.zero_grad()
        
        # restrict volume
        perturbation.data = torch.clamp(perturbation, -epsilon, epsilon)

        # Create perturbed audio
        perturbed_audio = audio + perturbation

        # forward pass
        loss = compute_loss(model, perturbed_audio, device_sources, device)
        
        # Backpropagate
        loss.backward()
        
        # Update perturbation
        optimizer.step()
        
        # restrist volume again
        perturbation.data = torch.clamp(perturbation, -epsilon, epsilon)
        
        # Calculate metrics
        with torch.no_grad():
                        
            # Save best perturbation
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_perturbation = perturbation.clone().detach()
                print(f"found better perbutation with loss: {best_loss}")
                
            
            if (i + 1) % 1000 == 0:
                
                print(f"Iteration {i+1}/{iterations}, Loss: {loss.item():.4f}")
    
    
    # Return the best perturbation found - make sure it's detached
    return best_perturbation.detach()



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='White-box adversarial attack on Demucs v4')
    
    #input file
    parser.add_argument('--input_file', type=str, required=True, help='Path to input audio file') 
    
    #output file
    parser.add_argument('--output_dir', type=str, default='./attack_results', help='Directory to save results')
    
    #type of demuc's model
    parser.add_argument('--model', type=str, default="htdemucs", help='Demucs model name')
    
    #hyper parameters
    parser.add_argument('--iterations', type=int, default=5000, help='Number of attack iterations')
    parser.add_argument('--epsilon', type=float, default=0.0005, help='L-infinity constraint')
    parser.add_argument('--lr', type=float, default=0.5, help='Learning rate for optimization')
    
    #device (GPU recomonded, cpu will be very slow)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--save_sources', action='store_true', help='Save separated sources to output directory')
    return parser.parse_args()

def main():
    """
    Main function to run the adversarial attack on a whole song.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Extract song name from the input file path
    song_name = os.path.splitext(os.path.basename(args.input_file))[0]
    print(f"Processing song: {song_name}")
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create song-specific output directory
    song_output_dir = os.path.join(args.output_dir, song_name)
    os.makedirs(song_output_dir, exist_ok=True)
    
    # Create subdirectories
    originals_dir = os.path.join(song_output_dir, "separation_prior_attack")
    estimated_dir = os.path.join(song_output_dir, "separation_after_attack")
    comparisons_dir = os.path.join(song_output_dir, "comparisons")
    
    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(estimated_dir, exist_ok=True)
    
    # Set device and print info
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Attack parameters: epsilon={args.epsilon}, lr={args.lr}, iterations={args.iterations}")
    
    try:
        # Load Demucs model
        print(f"Loading Demucs model: {args.model}")
        model = get_model(args.model)
        model.to(device)
        
        # Load mixture and separate into sources
        print(f"Loading mixture from {args.input_file}")
        original_mixture, sample_rate = torchaudio.load(args.input_file)
        
        # Save original mixture
        original_mixture_path = os.path.join(originals_dir, "mixture.wav")
        torchaudio.save(original_mixture_path, original_mixture, sample_rate)
        print(f"Saved original mixture to {original_mixture_path}")
        
        # Separate original sources
        print("Separating original sources...")
        original_mixture_device = original_mixture.to(device)
        if original_mixture_device.dim() == 2:
            original_mixture_batch = original_mixture_device.unsqueeze(0)
        else:
            original_mixture_batch = original_mixture_device
        
        with torch.no_grad():
            original_sources_batch = apply_model(model, original_mixture_batch, device=device)
        
        original_sources_batch = original_sources_batch.squeeze(0)  # [sources, channels, samples]
        
        # Create sources dictionary
        source_names = ['drums', 'bass', 'other', 'vocals']
        sources = {'mixture': original_mixture}
        
        # Save original separated sources
        for i, name in enumerate(source_names):
            sources[name] = original_sources_batch[i].cpu()
            source_path = os.path.join(originals_dir, f"{name}.wav")
            torchaudio.save(source_path, sources[name], sample_rate)
            print(f"Saved original {name} to {source_path}")
        
        # Apply adversarial attack to the whole song
        print(f"Starting adversarial attack on {song_name}")
        perturbation = run_adversarial_attack(
            model=model,
            sources=sources,
            output_dir=song_output_dir,
            iterations=args.iterations,
            epsilon=args.epsilon,
            lr=args.lr,
            device=device
        )
        
        # Create perturbed mixture
        perturbed_mixture = original_mixture + perturbation.cpu()
        
        # Save perturbed mixture
        perturbed_mixture_path = os.path.join(song_output_dir, "perturbed_mixture.wav")
        torchaudio.save(perturbed_mixture_path, perturbed_mixture, sample_rate)
        print(f"Saved perturbed mixture to {perturbed_mixture_path}")
        
        # Save perturbation alone
        perturbation_path = os.path.join(song_output_dir, "final_perturbation.wav")
        torchaudio.save(perturbation_path, perturbation.cpu(), sample_rate)
        print(f"Saved perturbation to {perturbation_path}")
        
        # Separate sources from the perturbed audio
        print("Separating sources from perturbed audio...")
        perturbed_mixture_device = perturbed_mixture.to(device)
        if perturbed_mixture_device.dim() == 2:
            perturbed_mixture_batch = perturbed_mixture_device.unsqueeze(0)
        else:
            perturbed_mixture_batch = perturbed_mixture_device
        
        with torch.no_grad():
            perturbed_sources_batch = apply_model(model, perturbed_mixture_batch, device=device)
        
        perturbed_sources_batch = perturbed_sources_batch.squeeze(0)  # [sources, channels, samples]
        
        # Save perturbed sources
        for i, name in enumerate(source_names):
            source_path = os.path.join(estimated_dir, f"{name}.wav")
            perturbed_source = perturbed_sources_batch[i].cpu()
            torchaudio.save(source_path, perturbed_source, sample_rate)
            print(f"Saved perturbed {name} to {source_path}")
            
            # Calculate difference with original source
            source_diff = torch.mean(torch.abs(sources[name] - perturbed_source)).item()
            print(f"Average difference between original and perturbed {name}: {source_diff}")
        
        # Save perturbed mixture in estimated folder for completeness
        torchaudio.save(os.path.join(estimated_dir, "mixture.wav"), perturbed_mixture, sample_rate)
        

            
           

        print(f"Attack on '{song_name}' completed successfully!")
        print(f"Results saved in: {song_output_dir}")
        print(f"  Original sources: {originals_dir}")
        print(f"  Estimated sources (after attack): {estimated_dir}")
        print(f"  Visualizations: {comparisons_dir}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)