"""
Start Program - Interactive Song Trimmer
Main program that processes audio files from 'originals' folder.
"""
from pathlib import Path
from utils.trim_song import trim_song
import sys
sys.path.append('attacking_scripts')
from process_folder import process_audio_folder

def find_audio_files(folder_path):
    """Find all audio files in the folder."""
    folder_path = Path(folder_path)
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.mp4', '.aac', '.ogg']
    
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(folder_path.glob(f'*{ext}'))
    
    return sorted(audio_files)

def main():
    
    # Set input folder to "originals"
    input_folder = Path("originals")
    
    # Find audio files
    audio_files = find_audio_files(input_folder)
    
    if not audio_files:
        print("❌ No audio files found in 'originals' folder!")
        print("\nSupported formats: WAV, MP3, FLAC, M4A, MP4, AAC, OGG")
        print("Please add some audio files to the 'originals' folder.")
        return
    
    # Show found files
    print(f"📁 Found {len(audio_files)} audio files in 'originals' folder:")
    for i, file in enumerate(audio_files[:5], 1):  # Show first 5
        print(f"   {i}. {file.name}")
    if len(audio_files) > 5:
        print(f"   ... and {len(audio_files) - 5} more files")
    
    # Ask user if they want to trim
    print(f"\n🎵 Do you want to trim these {len(audio_files)} songs?")
    trimming_chosen = False
    while True:
        choice = input("Enter 'y' for yes, 'n' for no: ").lower().strip()
        if choice in ['y', 'yes']:
            trimming_chosen = True
            break
        elif choice in ['n', 'no']:
            trimming_chosen = False
            break
        else:
            print("❌ Please enter 'y' for yes or 'n' for no.")
    
    # If trimming was chosen, do the trimming process
    if trimming_chosen:
        # Ask for trim duration
        while True:
            try:
                print("\n⏱️  How many seconds would you like to keep from the start of each song?")
                duration_input = input("Enter duration (e.g., 30, 45, 60): ").strip()
                duration = float(duration_input)
                
                if duration <= 0:
                    print("❌ Duration must be positive. Please try again.")
                    continue
                
                print(f"✓ Will trim each song to {duration} seconds")
                break
                
            except ValueError:
                print("❌ Invalid input. Please enter a number (e.g., 30, 45, 60).")
        
        # Create output folder reference (for display purposes)
        output_folder = Path("originals_trimmed")
        print(f"\n📂 Output folder: {output_folder}")
        
        # Process all files
        print(f"\n🎶 Processing {len(audio_files)} songs...")
        print("=" * 50)
        
        successful = 0
        failed = 0
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}]", end=" ")
            result = trim_song(audio_file, duration)  
            if result:
                successful += 1
            else:
                failed += 1
        
        print(f"\n✓ Trimming complete! {successful} successful, {failed} failed")
    else:
        print("👋 No trimming selected.")
    
    # Now run the attack script processing
    print("\n" + "=" * 60)
    print("🚀 Starting attack script processing...")
    
    # Determine which folder to process
    if trimming_chosen:
        target_folder = "originals_trimmed"
        print(f"📂 Processing trimmed files in '{target_folder}' folder")
    else:
        target_folder = "originals"
        print(f"📂 Processing original files in '{target_folder}' folder")
    
    # Run the attack script processing
    try:
        process_audio_folder(
            input_folder=target_folder,
            script_path="attacking_scripts/attack_script.py"
        )
        print("✅ Attack script processing completed!")
    except Exception as e:
        print(f"❌ Error during attack script processing: {e}")
    
    print("\n🎉 All processing complete!")

if __name__ == "__main__":
    main()