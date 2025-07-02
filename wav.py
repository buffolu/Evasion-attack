import os
import subprocess
from pathlib import Path
import argparse

def extract_mixture_from_stem(input_folder, output_folder='originals', output_format='wav'):
    """
    Extract mixture (full song) from .stem.m4 files.
    
    Args:
        input_folder: Path to folder containing .stem.m4 files
        output_folder: Name of output folder (default: 'originals')
        output_format: Output format ('wav', 'mp3', 'flac')
    """
    input_path = Path(input_folder)
    output_path = input_path / output_folder
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    print(f"Created output folder: {output_path}")
    
    # Find all .stem.m4 files
    stem_files = list(input_path.glob('*.stem.m4'))
    
    if not stem_files:
        print("No .stem.m4 files found!")
        print("Looking for files with different extensions...")
        
        # Try other possible STEM file extensions
        alt_patterns = ['*.stem', '*.stem.mp4', '*.stem.m4a']
        for pattern in alt_patterns:
            alt_files = list(input_path.glob(pattern))
            if alt_files:
                print(f"Found {len(alt_files)} {pattern} files")
                stem_files = alt_files
                break
        
        if not stem_files:
            print("No STEM files found!")
            return
    
    print(f"Found {len(stem_files)} STEM files to process:")
    
    for stem_file in stem_files:
        # Create output filename
        base_name = stem_file.name.replace('.stem.m4', '').replace('.stem', '')
        output_file = output_path / f"{base_name}.{output_format}"
        
        print(f"Extracting mixture: {stem_file.name} -> {output_file.name}")
        
        try:
            # Extract mixture track (usually track 0) using ffmpeg
            cmd = [
                'ffmpeg', 
                '-i', str(stem_file),
                '-map', '0:0',  # Select first audio track (mixture)
                '-y'  # Overwrite output file
            ]
            
            # Add format-specific options
            if output_format == 'wav':
                cmd.extend(['-c:a', 'pcm_s16le', '-ar', '44100'])
            elif output_format == 'mp3':
                cmd.extend(['-c:a', 'libmp3lame', '-b:a', '320k'])
            elif output_format == 'flac':
                cmd.extend(['-c:a', 'flac'])
            
            cmd.append(str(output_file))
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Show file size
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ Extracted successfully ({size_mb:.1f} MB)")
            
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error extracting from {stem_file.name}")
            # Try alternative track mapping
            try:
                print("    Trying alternative track extraction...")
                cmd_alt = [
                    'ffmpeg', 
                    '-i', str(stem_file),
                    '-c:a', 'pcm_s16le' if output_format == 'wav' else 'libmp3lame',
                    '-ar', '44100',
                    '-y',
                    str(output_file)
                ]
                subprocess.run(cmd_alt, check=True, capture_output=True)
                
                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"  ✓ Extracted successfully with alternative method ({size_mb:.1f} MB)")
                
            except subprocess.CalledProcessError:
                print(f"  ✗ Failed to extract from {stem_file.name}")
                
        except FileNotFoundError:
            print("  ✗ Error: ffmpeg not found. Please install ffmpeg first.")
            break
    
    print(f"\nCompleted! Mixture files saved to: {output_path}")

def show_stem_info(stem_file):
    """
    Show information about tracks in a STEM file.
    """
    try:
        result = subprocess.run([
            'ffprobe', 
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            str(stem_file)
        ], capture_output=True, text=True, check=True)
        
        import json
        data = json.loads(result.stdout)
        
        print(f"\nTracks in {stem_file.name}:")
        for i, stream in enumerate(data.get('streams', [])):
            if stream.get('codec_type') == 'audio':
                title = stream.get('tags', {}).get('title', f'Track {i}')
                print(f"  Track {i}: {title}")
                
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        print(f"Cannot analyze {stem_file.name}")

def main():
    parser = argparse.ArgumentParser(description='Extract mixture from STEM files')
    parser.add_argument('input_folder', help='Folder containing .stem.m4 files')
    parser.add_argument('--output', '-o', default='originals', help='Output folder name (default: originals)')
    parser.add_argument('--format', choices=['wav', 'mp3', 'flac'], default='wav',
                       help='Output format (default: wav)')
    parser.add_argument('--info', action='store_true', help='Show track information for first STEM file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        return
    
    if args.info:
        # Show info about first STEM file found
        input_path = Path(args.input_folder)
        stem_files = list(input_path.glob('*.stem.m4'))
        if not stem_files:
            stem_files = list(input_path.glob('*.stem'))
        
        if stem_files:
            show_stem_info(stem_files[0])
        else:
            print("No STEM files found to analyze")
        return
    
    extract_mixture_from_stem(args.input_folder, args.output, args.format)

# Simple function for direct use
def extract_mixtures(folder_path, output_format='wav'):
    """
    Simple function to extract mixtures from STEM files.
    
    Args:
        folder_path: Path to folder containing .stem.m4 files
        output_format: Output format ('wav', 'mp3', 'flac')
    """
    extract_mixture_from_stem(folder_path, output_format=output_format)

if __name__ == "__main__":
    main()

# Example usage:
"""
# Extract mixtures as WAV files:
python stem_extractor.py /path/to/stems

# Extract as MP3 (smaller files):
python stem_extractor.py /path/to/stems --format mp3

# Show track info:
python stem_extractor.py /path/to/stems --info

# In Python:
from stem_extractor import extract_mixtures
extract_mixtures('/path/to/stems')
"""