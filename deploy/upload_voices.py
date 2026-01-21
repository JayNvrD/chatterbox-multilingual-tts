"""
Voice Upload Script for Chatterbox TTS (Modal)

Uses the Modal client to call the 'upload_voice' function directly,
bypassing the HTTP API for secure volume access.
"""

import modal
import os
from pathlib import Path

# App name locally defined in main.py
APP_NAME = "chatterbox-tts"

# Voices to upload: (filename, voice_name, language_code)
# The files are expected to be in the 'voices/' subdirectory
VOICES = [
    # English
    ("voices/abigail_en_female.wav", "abigail_en_female", "en"),
    ("voices/anaya_en_female.wav", "anaya_en_female", "en"),
    ("voices/john_en_male.wav", "john_en_male", "en"),
    
    # Hindi
    ("voices/priya_hi_female.wav", "priya_hi_female", "hi"),
    ("voices/raj_hi_male.wav", "raj_hi_male", "hi"),
    
    # Spanish
    ("voices/maria_es_female.wav", "maria_es_female", "es"),
    ("voices/carlos_es_male.wav", "carlos_es_male", "es"),
    
    # Arabic
    ("voices/fatima_ar_female.wav", "fatima_ar_female", "ar"),
    ("voices/omar_ar_male.wav", "omar_ar_male", "ar"),
]

def main():
    print(f"Connecting to Modal app: {APP_NAME}...")
    try:
        # Get the remote function
        upload_fn = modal.Function.from_name(APP_NAME, "upload_voice")
    except Exception as e:
        print(f"Error finding function: {e}")
        print("Did you run 'modal deploy main.py'?")
        return

    print("Starting upload...")
    success_count = 0
    
    for filename, voice_name, lang in VOICES:
        # Resolve path relative to this script
        file_path = Path(__file__).parent / filename
        
        if not file_path.exists():
            print(f"❌ File not found: {filename}")
            continue
            
        print(f"Uploading {voice_name} ({lang})...", end=" ", flush=True)
        try:
            # Call remote function
            audio_bytes = file_path.read_bytes()
            upload_fn.remote(voice_name, audio_bytes)
            print("✅")
            success_count += 1
        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\nDone! Uploaded {success_count}/{len(VOICES)} voices.")
    print("Run 'modal run main.py --list-volume' to verify.")

if __name__ == "__main__":
    main()
