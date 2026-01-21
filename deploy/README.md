# Chatterbox TTS API - Deploy

Complete deployment package for Chatterbox TTS with LiveKit integration.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Modal deployment (API) |
| `chatterbox_tts.py` | LiveKit TTS plugin |
| `livekit_agent.py` | Example voice agent |
| `upload_voices.py` | Voice upload helper |
| `voices/` | Voice samples (9 ready) |

## Quick Start

### 1. Deploy API to Modal
```bash
pip install modal
modal setup
modal deploy main.py
```

### 2. Upload Voices
```bash
# Edit API_URL in upload_voices.py first!
python upload_voices.py --url YOUR_MODAL_URL
```

### 3. Run LiveKit Agent
```bash
pip install -r requirements_livekit.txt

# Set environment variables
export LIVEKIT_URL=wss://your-project.livekit.cloud
export LIVEKIT_API_KEY=your_key
export LIVEKIT_API_SECRET=your_secret
export CHATTERBOX_API_URL=https://your-modal-url.modal.run

# Run agent
python livekit_agent.py dev
```

## LiveKit Usage

```python
from chatterbox_tts import ChatterboxTTS

# Initialize
tts = ChatterboxTTS(
    api_url="https://your-modal-url.modal.run",
    voice="priya_hi_female",  # Your uploaded voice
    language="hi"
)

# Streaming (for LiveKit)
stream = tts.stream()
stream.push_text("Hello world")
stream.end_input()

async for audio in stream:
    await audio_source.capture_frame(audio.frame)
```

## Available Voices

| Voice ID | Language |
|----------|----------|
| `abigail_en_female` | English |
| `anaya_en_female` | English |
| `john_en_male` | English |
| `priya_hi_female` | Hindi |
| `raj_hi_male` | Hindi |
| `maria_es_female` | Spanish |
| `carlos_es_male` | Spanish |
| `fatima_ar_female` | Arabic |
| `omar_ar_male` | Arabic |

## Cost Estimate

T4 GPU: ~$117/month for 100 calls × 5 min/day
