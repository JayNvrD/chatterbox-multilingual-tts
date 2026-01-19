"""
Chatterbox Multilingual TTS - Fast & Optimized

23 language support with ~200ms inference latency.

Usage:
    from chatterbox_mtl import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    
    # Fast mode (default) - uses meanflow S3Gen
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    wav = model.generate("Hello world!", language_id="en")
    
    # Quality mode - uses standard S3Gen (slower but higher quality)
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", use_meanflow=False)
"""

__version__ = "1.0.0"

from .mtl_tts import (
    ChatterboxMultilingualTTS,
    SUPPORTED_LANGUAGES,
    Conditionals,
    punc_norm,
)

__all__ = [
    "ChatterboxMultilingualTTS",
    "SUPPORTED_LANGUAGES", 
    "Conditionals",
    "punc_norm",
    "__version__",
]
