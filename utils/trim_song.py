import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import argparse
import sys

def trim_song(input_file, duration):
    """
    Simple function to trim a song to specified duration from the start.
    
    Args:
        input_file: Path to input audio file
        duration: Duration in seconds to keep
    
    Returns:
        Path to output file
    """
    input_path = Path(input_file)
    
    # Create output folder at project root level (not inside originals)
    output_folder = Path("originals_trimmed")
    output_folder.mkdir(exist_ok=True)
    output_path = output_folder / f"{input_path.stem}.wav"
    
    print(f"Loading: {input_path.name}")
    
    # Load audio file
    audio, sr = librosa.load(input_file, sr=None, mono=False)
    
    # Convert to stereo if mono
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    
    # Calculate samples for duration
    duration_samples = int(duration * sr)
    
    # Trim audio (from start to duration)
    if duration_samples < audio.shape[1]:
        audio = audio[:, :duration_samples]
        print(f"Trimmed to {duration} seconds")
    else:
        print(f"Song is shorter than {duration}s, keeping full length")
    
    # Save as WAV
    sf.write(output_path, audio.T, sr)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    actual_duration = audio.shape[1] / sr
    print(f"✓ Saved: {output_path.name} ({actual_duration:.1f}s, {size_mb:.1f} MB)")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Trim song to specified duration')
    parser.add_argument('input_file', help='Input audio file')
    parser.add_argument('duration', type=float, help='Duration in seconds')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: File '{args.input_file}' not found")
        sys.exit(1)
    
    # Check duration is positive
    if args.duration <= 0:
        print("Error: Duration must be positive")
        sys.exit(1)
    
    # Trim the song
    trim_song(args.input_file, args.duration)

if __name__ == "__main__":
    main()

# Example usage:
"""
# Command line:
python trim_song.py song.mp3 30      # Trim to 30 seconds
python trim_song.py track.wav 45     # Trim to 45 seconds

# In Python:
from trim_song import trim_song
trim_song('song.mp3', 30)  # Trim to 30 seconds
"""