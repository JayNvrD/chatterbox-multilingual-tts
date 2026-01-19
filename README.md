# Chatterbox Multilingual TTS

**Fast Multilingual Text-to-Speech with 23 Language Support**

~200ms inference latency using meanflow optimization.

## Installation

```bash
pip install -e .
```

Or with dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from chatterbox_mtl import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# Load model (fast mode - ~200ms)
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Generate speech
wav = model.generate("Hello, how are you?", language_id="en")

# Save audio
import torchaudio
torchaudio.save("output.wav", wav, model.sr)
```

## Performance Modes

| Mode | Parameter | Latency | Quality |
|------|-----------|---------|---------|
| **Fast** | `use_meanflow=True` | ~200ms | Good |
| **Quality** | `use_meanflow=False` | ~4 sec | Best |

```python
# Fast mode (default)
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", use_meanflow=True)

# Quality mode
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", use_meanflow=False)

# Custom CFM steps
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", n_cfm_timesteps=4)
```

## Supported Languages (23)

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| ar | Arabic | hi | Hindi | pl | Polish |
| da | Danish | it | Italian | pt | Portuguese |
| de | German | ja | Japanese | ru | Russian |
| el | Greek | ko | Korean | sv | Swedish |
| en | English | ms | Malay | sw | Swahili |
| es | Spanish | nl | Dutch | tr | Turkish |
| fi | Finnish | no | Norwegian | zh | Chinese |
| fr | French | he | Hebrew | | |

## API Reference

### `ChatterboxMultilingualTTS.from_pretrained()`

```python
model = ChatterboxMultilingualTTS.from_pretrained(
    device="cuda",           # "cuda" or "cpu"
    use_meanflow=True,       # Fast mode (2 CFM steps)
    n_cfm_timesteps=None,    # Custom CFM steps (default: 2 or 10)
)
```

### `model.generate()`

```python
wav = model.generate(
    text="Hello world!",
    language_id="en",             # Language code
    audio_prompt_path=None,       # Path to voice sample (optional)
    exaggeration=0.5,             # Emotion intensity (0.0-2.0)
    cfg_weight=0.5,               # Guidance weight (0.0-1.0)
    temperature=0.8,              # Sampling temperature
)
```

## License

MIT License - Based on [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox)
