import os
import torch
import torchaudio as ta
import time
from mtl_tts import ChatterboxMultilingualTTS

def main():
    print("Testing Chatterbox TTS locally...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("WARNING: Running on CPU will be very slow.")

    # Load model
    print("Loading model from pretrained...")
    # NOTE: This may require HF_TOKEN if gated.
    # We use the token handled in the model code now.
    start_time = time.perf_counter()
    model = ChatterboxMultilingualTTS.from_pretrained(
        device=device,
        use_meanflow=True,  # Test optimized mode
        token=os.getenv("HF_TOKEN")
    )
    print(f"Model loaded in {time.perf_counter() - start_time:.2f}s")

    # Select a voice prompt
    voice_path = "deploy/voices/anaya_en_female.wav"
    if not os.path.exists(voice_path):
        # Fallback to current dir if run from deploy/
        voice_path = "voices/anaya_en_female.wav"
        
    print(f"Using voice prompt: {voice_path}")
    
    text = "Hello! I am running locally on your NVIDIA GPU. The SDPA optimization and absolute import fixes are being verified."
    print(f"Synthesizing: \"{text}\"")
    
    start_time = time.perf_counter()
    wav = model.generate(
        text=text,
        language_id="en",
        audio_prompt_path=voice_path,
        exaggeration=0.5
    )
    duration = time.perf_counter() - start_time
    print(f"Generation finished in {duration:.2f}s")

    # Save output
    output_path = "local_test_output.wav"
    ta.save(output_path, wav, 24000)
    print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    main()
