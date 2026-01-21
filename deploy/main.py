"""
Chatterbox Multilingual TTS API - Optimized Single GPU Container

Uses local chatterbox_mtl package with meanflow S3Gen for ~200ms latency.
FastAPI runs directly on GPU container - no network hop!

Deploy:
    modal deploy main.py

Test:
    modal run main.py --text "Hello world" --language en

API Endpoints:
    POST /stream         - Streaming speech generation (for LiveKit)
    POST /speak          - Blocking speech generation
    GET  /voices         - List available voices
    GET  /languages      - List supported languages
    GET  /health         - Health check
    GET  /docs           - OpenAPI documentation
"""

import modal
import io
import os
import struct
from typing import Optional

# =============================================================================
# Configuration
# =============================================================================

GPU_TYPE = "A10G"  # Options: "T4", "L4", "A10G", "A100"
SCALEDOWN_WINDOW = 60 * 5  # 5 minutes
USE_MEANFLOW = True  # True = ~200ms, False = ~4 seconds

# Supported languages (23)
SUPPORTED_LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
    "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}

# =============================================================================
# WAV Header Utility
# =============================================================================

def create_wav_header(sample_rate: int, channels: int, bits_per_sample: int, data_size: int = 0xFFFFFFFF) -> bytes:
    """Creates a WAV header for streaming."""
    header = io.BytesIO()
    header.write(b'RIFF')
    chunk_size = 36 + data_size if data_size != 0xFFFFFFFF else 0x7FFFFFFF - 36
    header.write(struct.pack('<I', chunk_size))
    header.write(b'WAVE')
    header.write(b'fmt ')
    header.write(struct.pack('<I', 16))
    header.write(struct.pack('<H', 1))
    header.write(struct.pack('<H', channels))
    header.write(struct.pack('<I', sample_rate))
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    header.write(struct.pack('<I', byte_rate))
    block_align = channels * (bits_per_sample // 8)
    header.write(struct.pack('<H', block_align))
    header.write(struct.pack('<H', bits_per_sample))
    header.write(b'data')
    header.write(struct.pack('<I', data_size))
    return header.getvalue()

# =============================================================================
# Modal App & Image
# =============================================================================

app = modal.App("chatterbox-tts")

# Local package will be copied into the image at build time

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.0.0,<2.7.0",
        "torchaudio>=2.0.0,<2.7.0",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "numpy>=1.24.0,<1.26.0",
        "librosa>=0.10.0",
        "transformers>=4.40.0",
        "diffusers>=0.29.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.20.0",
        "resemble-perth>=1.0.1",
        "conformer>=0.3.2",
        "s3tokenizer",
        "pyloudnorm",
        "omegaconf",
        "spacy-pkuseg",
        "pykakasi>=2.3.0",
        "fastapi[standard]",
    )
    # Add local optimized code (mtl_tts.py, models/, etc.)
    .add_local_dir("..", remote_path="/root/chatterbox_mtl", copy=True, ignore=[".git", "deploy", "notebooks", "__pycache__", ".venv", ".agent"])
)

# Volumes for caching and voices
cache_volume = modal.Volume.from_name("chatterbox-cache", create_if_missing=True)
voices_volume = modal.Volume.from_name("chatterbox-voices", create_if_missing=True)

# =============================================================================
# Single GPU Container: TTS Model + FastAPI
# =============================================================================

@app.cls(
    image=image,
    gpu=GPU_TYPE,
    min_containers=1,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={
        "/cache": cache_volume,
        "/voices": voices_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=10)
class TTSService:
    """
    Single GPU container with TTS model and FastAPI endpoints.
    No network hop - FastAPI runs directly on GPU!
    """
    
    @modal.enter()
    def load_model(self):
        """Load model when container starts."""
        import sys
        import time
        
        # Add the copied directory to path
        sys.path.insert(0, "/root/chatterbox_mtl")
        os.environ["HF_HOME"] = "/cache/huggingface"
        self.load_error = None
        
        # Import the optimized module directly from the root
        from mtl_tts import ChatterboxMultilingualTTS
        
        print("=" * 60)
        print("Loading Chatterbox Multilingual TTS")
        print(f"Mode: {'meanflow (~200ms)' if USE_MEANFLOW else 'standard (~4s)'}")
        print("=" * 60)
        
        start = time.perf_counter()
        
        # Check CUDA
        self.cuda_ok = False
        self.device_name = "CPU"
        try:
            import torch
            self.cuda_ok = torch.cuda.is_available()
            if self.cuda_ok:
                self.device_name = torch.cuda.get_device_name(0)
        except Exception as e:
            print(f"CUDA Diagnostic Error: {e}")

        print(f"--- DEVICE DIAGNOSTIC ---")
        print(f"CUDA Available: {self.cuda_ok}")
        print(f"Device Name: {self.device_name}")
        print(f"-------------------------")

        try:
            self.model = ChatterboxMultilingualTTS.from_pretrained(
                device="cuda" if self.cuda_ok else "cpu",
                use_meanflow=USE_MEANFLOW,
            )
            self.sample_rate = self.model.sr
            print(f"✅ Model loaded in {time.perf_counter() - start:.1f}s on {self.device_name}")
        except Exception as e:
            self.load_error = str(e)
            print(f"Model Load Error: {e}")
            # Don't raise, let the API report it
        print(f"✅ Sample rate: {self.sample_rate}")
        print(f"✅ CFM steps: {self.model.n_cfm_timesteps}")
        print("=" * 60)
    
    def _generate_audio(
        self,
        text: str,
        language: str = "en",
        voice: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
    ) -> bytes:
        """Internal method to generate audio. Returns WAV bytes."""
        import torchaudio as ta
        
        # Validate language
        lang_id = language if language in SUPPORTED_LANGUAGES else "en"
        
        # Find voice file if specified
        voice_path = None
        if voice:
            for ext in ['.wav', '.mp3']:
                path = f"/voices/{voice}{ext}"
                if os.path.exists(path):
                    voice_path = path
                    break
        
        # Generate audio
        wav = self.model.generate(
            text=text,
            language_id=lang_id,
            audio_prompt_path=voice_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        ta.save(buffer, wav, self.sample_rate, format="wav")
        buffer.seek(0)
        return buffer.read()
    
    @modal.asgi_app()
    def api(self):
        """FastAPI app running directly on GPU container."""
        from fastapi import FastAPI, Query, HTTPException
        from fastapi.responses import StreamingResponse, JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        import time
        
        api = FastAPI(
            title="Chatterbox Multilingual TTS API",
            description="Fast Multilingual Text-to-Speech (~200ms) with 23 languages",
            version="3.0.0",
        )
        
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @api.get("/health")
        async def health():
            return {
                "status": "ok",
                "model": "chatterbox-multilingual",
                "mode": "meanflow" if USE_MEANFLOW else "standard",
                "sample_rate": self.sample_rate,
                "languages": len(SUPPORTED_LANGUAGES),
            }
        
        @api.get("/languages")
        async def languages():
            return {
                "languages": [{"code": k, "name": v} for k, v in SUPPORTED_LANGUAGES.items()],
                "count": len(SUPPORTED_LANGUAGES),
            }
        
        @api.get("/voices")
        async def voices():
            voices_volume.reload()
            voice_list = []
            if os.path.exists("/voices"):
                for f in os.listdir("/voices"):
                    if f.endswith(('.wav', '.mp3')):
                        name = os.path.splitext(f)[0]
                        parts = name.rsplit('_', 2)
                        lang = parts[-2] if len(parts) >= 3 else "en"
                        gender = parts[-1] if len(parts) >= 3 else "unknown"
                        voice_list.append({
                            "name": name,
                            "language": lang,
                            "gender": gender,
                        })
            return {"voices": voice_list, "count": len(voice_list)}
        
        @api.post("/speak")
        async def speak(
            text: str = Query(..., description="Text to synthesize"),
            language: str = Query("en", description="Language code"),
            voice: str = Query(None, description="Voice name"),
            exaggeration: float = Query(0.5, ge=0.0, le=2.0),
        ):
            """Generate speech (blocking). Returns full WAV file."""
            if not text.strip():
                raise HTTPException(status_code=400, detail="Text cannot be empty")
            
            try:
                start = time.perf_counter()
                audio_bytes = self._generate_audio(
                    text=text,
                    language=language,
                    voice=voice,
                    exaggeration=exaggeration,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                print(f"✅ /speak: {elapsed_ms:.0f}ms for '{text[:50]}...'")
                
                return StreamingResponse(
                    io.BytesIO(audio_bytes),
                    media_type="audio/wav",
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @api.post("/stream")
        async def stream(
            text: str = Query(..., description="Text to synthesize"),
            language: str = Query("en", description="Language code"),
            voice: str = Query(None, description="Voice name"),
            exaggeration: float = Query(0.5, ge=0.0, le=2.0),
            chunk_size: int = Query(4800, description="PCM samples per chunk"),
        ):
            """
            Generate speech with streaming response.
            Yields WAV header + PCM chunks for low-latency playback.
            """
            if not text.strip():
                raise HTTPException(status_code=400, detail="Text cannot be empty")
            
            async def generate_stream():
                import torchaudio as ta
                import io
                
                try:
                    start_time = time.perf_counter()
                    
                    # Validate language
                    lang_id = language if language in SUPPORTED_LANGUAGES else "en"
                    
                    # Find voice file
                    voice_path = None
                    if voice:
                        for ext in ['.wav', '.mp3']:
                            path = f"/voices/{voice}{ext}"
                            if os.path.exists(path):
                                voice_path = path
                                break
                    
                    # Generate audio - SKIP POST-PROCESSING for millisecond TTFB
                    # In true meanflow mode, this takes ~200-300ms on A10G
                    wav = self.model.generate(
                        text=text,
                        language_id=lang_id,
                        audio_prompt_path=voice_path,
                        exaggeration=exaggeration,
                        apply_post_processing=False  # ← Crucial for ms latency!
                    )
                    
                    ttfb_ms = (time.perf_counter() - start_time) * 1000
                    print(f"⚡ TTFB: {ttfb_ms:.2f}ms for '{text[:20]}...'")
                    
                    # Convert to WAV in memory and yield chunks
                    buffer = io.BytesIO()
                    ta.save(buffer, wav, self.sample_rate, format="wav")
                    buffer.seek(0)
                    
                    # Yield first chunk (contains WAV header)
                    chunk = buffer.read(4800)
                    if chunk:
                        yield chunk
                    
                    # Yield remaining data
                    while True:
                        chunk = buffer.read(4800)
                        if not chunk:
                            break
                        yield chunk
                        
                except Exception as e:
                    print(f"❌ Streaming Error: {e}")
                    raise
            
            return StreamingResponse(
                generate_stream(), 
                media_type="audio/wav"
            )
        
        @api.get("/info")
        async def info():
            """Get model info."""
            return {
                "model": "chatterbox-multilingual",
                "sample_rate": self.sample_rate,
                "cfm_steps": self.model.n_cfm_timesteps,
                "use_meanflow": self.model.use_meanflow,
                "languages": len(SUPPORTED_LANGUAGES),
                "gpu": GPU_TYPE,
                "cuda": self.cuda_ok if hasattr(self, "cuda_ok") else False,
                "device": self.device_name if hasattr(self, "device_name") else "unknown",
                "load_error": self.load_error if hasattr(self, "load_error") else "not_loaded",
            }
        
        return api
    
    @modal.method()
    def generate(
        self,
        text: str,
        language: str = "en",
        voice: str = None,
        exaggeration: float = 0.5,
    ) -> bytes:
        """Direct method for programmatic use. Returns WAV bytes."""
        return self._generate_audio(
            text=text,
            language=language,
            voice=voice,
            exaggeration=exaggeration,
        )

# =============================================================================
# Voice Upload Helper
# =============================================================================

@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/voices": voices_volume},
)
def upload_voice(name: str, audio_bytes: bytes):
    """Upload a voice sample to the volume."""
    from pathlib import Path
    Path(f"/voices/{name}.wav").write_bytes(audio_bytes)
    voices_volume.commit()
    return {"uploaded": name}

@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/voices": voices_volume},
)
def list_voices_volume():
    """List voices in the volume."""
    voices = []
    if os.path.exists("/voices"):
        for f in os.listdir("/voices"):
            if f.endswith(('.wav', '.mp3')):
                voices.append(os.path.splitext(f)[0])
    return voices

# =============================================================================
# CLI Entry Point
# =============================================================================

@app.local_entrypoint()
def main(
    text: str = "Hello, this is a fast multilingual TTS test!",
    language: str = "en",
    voice: str = None,
    output: str = "tts_output.wav",
):
    """Test the TTS model."""
    import pathlib
    import time
    
    print(f"Text: {text}")
    print(f"Language: {language}")
    print(f"Mode: {'meanflow (~200ms)' if USE_MEANFLOW else 'standard (~4s)'}")
    
    tts = TTSService()
    
    start = time.perf_counter()
    audio = tts.generate.remote(text=text, language=language, voice=voice)
    elapsed = (time.perf_counter() - start) * 1000
    
    pathlib.Path(output).write_bytes(audio)
    print(f"✅ Generated in {elapsed:.0f}ms")
    print(f"✅ Saved: {output}")
