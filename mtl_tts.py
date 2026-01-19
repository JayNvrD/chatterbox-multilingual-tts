"""
Chatterbox Multilingual TTS - Optimized for Low Latency

This is a modified version of the original ChatterboxMultilingualTTS
that supports meanflow S3Gen for ~200ms inference (vs ~4 seconds).

Key optimizations:
- use_meanflow=True: Uses distilled S3Gen model (2 CFM steps)
- n_cfm_timesteps=2: Reduces diffusion steps for faster inference

Usage:
    from chatterbox_mtl import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    
    # Fast mode (default) - ~200ms
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", use_meanflow=True)
    
    # Quality mode - ~4 seconds
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", use_meanflow=False)
"""

from typing import Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import os

import librosa
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download, hf_hub_download

from models.t3 import T3
from models.t3.modules.t3_config import T3Config
from models.s3tokenizer import S3_SR, drop_invalid_tokens
from models.s3gen import S3GEN_SR, S3Gen
from models.tokenizers import MTLTokenizer
from models.voice_encoder import VoiceEncoder
from models.t3.modules.cond_enc import T3Cond


# HuggingFace repositories
REPO_ID = "ResembleAI/chatterbox"          # Multilingual T3 + standard S3Gen
TURBO_REPO_ID = "ResembleAI/chatterbox-turbo"  # Meanflow S3Gen

# Supported languages (23)
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


def normalize_acronyms(text: str) -> str:
    """Expand acronyms like 'SDPA' to 'S.D.P.A.' with spaces to ensure phoneme separation."""
    import re
    def replace_acronym(match):
        acronym = match.group(0)
        # Expansion to "S. D. P. A. " forces the tokenizer to treat them as individual letters
        return " ".join([f"{c}." for c in acronym]) + " "
    
    # Match words that are 2-5 uppercase letters long
    return re.sub(r'\b[A-Z]{2,5}\b', replace_acronym, text)


def punc_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    if not text:
        return "You need to add some text for me to talk."
    
    # Normalize acronyms
    text = normalize_acronyms(text)

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxMultilingualTTS:
    """
    Multilingual Text-to-Speech with 23 language support.
    
    Supports two modes:
    - Fast mode (use_meanflow=True): ~200ms latency with 2 CFM steps
    - Quality mode (use_meanflow=False): ~4 seconds with 10 CFM steps
    """
    
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR
    
    # Default CFM steps for each mode
    DEFAULT_CFM_STEPS_MEANFLOW = 2   # Fast mode
    DEFAULT_CFM_STEPS_STANDARD = 10  # Quality mode

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Conditionals = None,
        use_meanflow: bool = True,
        n_cfm_timesteps: int = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio (24000)
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.use_meanflow = use_meanflow
        self.watermarker = perth.PerthImplicitWatermarker()
        
        # Set CFM timesteps based on mode
        if n_cfm_timesteps is not None:
            self.n_cfm_timesteps = n_cfm_timesteps
        else:
            self.n_cfm_timesteps = (
                self.DEFAULT_CFM_STEPS_MEANFLOW if use_meanflow 
                else self.DEFAULT_CFM_STEPS_STANDARD
            )

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(
        cls, 
        ckpt_dir, 
        device,
        use_meanflow: bool = True,
        meanflow_weights_path: str = None,
        n_cfm_timesteps: int = None,
    ) -> 'ChatterboxMultilingualTTS':
        """
        Load from local checkpoint directory.
        
        Args:
            ckpt_dir: Path to checkpoint directory with T3 and tokenizer files
            device: Device to load model on ("cuda" or "cpu")
            use_meanflow: If True, use fast meanflow S3Gen (requires meanflow_weights_path)
            meanflow_weights_path: Path to s3gen_meanflow.safetensors
            n_cfm_timesteps: Number of CFM steps (default: 2 for meanflow, 10 for standard)
        """
        ckpt_dir = Path(ckpt_dir)

        # Load Voice Encoder
        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", weights_only=True))
        ve.to(device).eval()

        # Load Multilingual T3
        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        # Load S3Gen (meanflow or standard)
        if use_meanflow:
            assert meanflow_weights_path is not None, (
                "meanflow_weights_path required when use_meanflow=True. "
                "Download from ResembleAI/chatterbox-turbo"
            )
            s3gen = S3Gen(meanflow=True)
            s3gen.load_state_dict(load_safetensors(meanflow_weights_path), strict=True)
        else:
            s3gen = S3Gen(meanflow=False)
            s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", weights_only=True))
        s3gen.to(device).eval()

        # Load tokenizer
        tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))

        # Load default voice conditionals
        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(
            t3, s3gen, ve, tokenizer, device, 
            conds=conds, 
            use_meanflow=use_meanflow,
            n_cfm_timesteps=n_cfm_timesteps
        )

    @classmethod
    def from_pretrained(
        cls, 
        device: str = "cuda",
        use_meanflow: bool = True,
        n_cfm_timesteps: int = None,
        token: Optional[str] = None,
    ) -> 'ChatterboxMultilingualTTS':
        """
        Load from HuggingFace Hub.
        
        Args:
            device: Device to load model on ("cuda" or "cpu")
            use_meanflow: If True, use fast meanflow S3Gen (~200ms). 
                         If False, use standard S3Gen (~4 seconds).
            n_cfm_timesteps: Number of CFM steps (default: 2 for meanflow, 10 for standard)
        
        Returns:
            ChatterboxMultilingualTTS instance
        """
        print(f"Loading Chatterbox Multilingual TTS (meanflow={use_meanflow})...")
        
        # Download multilingual model files
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=[
                    "ve.pt", 
                    "t3_mtl23ls_v2.safetensors", 
                    "s3gen.pt",
                    "grapheme_mtl_merged_expanded_v1.json", 
                    "conds.pt", 
                    "Cangjie5_TC.json"
                ],
                token=token,
            )
        )
        
        # Download meanflow S3Gen if needed
        meanflow_weights_path = None
        if use_meanflow:
            print("Downloading meanflow S3Gen for fast inference...")
            meanflow_weights_path = hf_hub_download(
                repo_id=TURBO_REPO_ID,
                filename="s3gen_meanflow.safetensors",
                token=token,
            )
        
        return cls.from_local(
            ckpt_dir, 
            device, 
            use_meanflow=use_meanflow,
            meanflow_weights_path=meanflow_weights_path,
            n_cfm_timesteps=n_cfm_timesteps,
        )

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        """Prepare voice conditionals from an audio file."""
        # Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text: str,
        language_id: str = "en",
        audio_prompt_path: str = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        apply_post_processing: bool = True,
    ) -> torch.Tensor:
        """
        Generate speech from text.
        
        Args:
            text: Text to synthesize
            language_id: Language code (e.g., "en", "hi", "es")
            audio_prompt_path: Optional path to voice sample for cloning
            exaggeration: Emotion intensity (0.0 to 2.0)
            cfg_weight: Classifier-free guidance weight (0.0 to 1.0)
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty
            min_p: Minimum probability threshold
            top_p: Top-p sampling threshold
            apply_post_processing: If False, skip watermarking and trimming (faster for streaming)
            
        Returns:
            Audio waveform as torch.Tensor with shape (1, samples)
        """
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, (
                "Please `prepare_conditionals` first or specify `audio_prompt_path`"
            )

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(
            text, language_id=language_id.lower() if language_id else None
        ).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            # T3: Text -> Speech Tokens
            t3_start = time.perf_counter()
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            t3_time = (time.perf_counter() - t3_start) * 1000
            # Extract only the conditional batch
            speech_tokens = speech_tokens[0]

            # Clean up invalid tokens
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            # S3Gen: Speech Tokens -> Waveform
            s3_start = time.perf_counter()
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
                n_cfm_timesteps=self.n_cfm_timesteps,  # Use configured CFM steps
            )
            s3_time = (time.perf_counter() - s3_start) * 1000
            print(f"  - T3 inference: {t3_time:.1f}ms")
            print(f"  - S3Gen inference: {s3_time:.1f}ms")
            
            wav = wav.squeeze(0).detach().cpu().numpy()
            
            if apply_post_processing:
                # Proper Fix: Aggressively trim silence/noise from the end
                # top_db=40 is fairly strict but safe for speech
                wav, _ = librosa.effects.trim(wav, top_db=40, frame_length=512, hop_length=128)
                wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
            
        return torch.from_numpy(wav).unsqueeze(0)
