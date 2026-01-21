"""
Chatterbox TTS Plugin for LiveKit Agents (v2 - Modern API)

This module provides a TTS class that integrates with LiveKit agents
using the modern tts.SynthesizeStream API with AudioEmitter for low-latency streaming.

Key improvements over v1:
- Uses modern LiveKit TTS API (AudioEmitter pattern)
- Proper error handling with LiveKit error types
- Uses LiveKit's sentence tokenizer for better text chunking
- Optimized streaming with immediate audio output
- Better connection management

Modal URL Format:
    After `modal deploy main.py`, you get a single base URL:
    https://<workspace>--chatterbox-tts-web-app.modal.run
    
    With routes:
    - POST /stream    - Streaming speech generation
    - POST /speak     - Non-streaming speech generation  
    - GET  /health    - Health check

Usage in LiveKit agent:
    from chatterbox_plugin import ChatterboxTTS
    
    tts = ChatterboxTTS(
        api_url="https://your-workspace--chatterbox-tts-web-app.modal.run",
        voice="default",
        language="en"
    )
    
    # Use in AgentSession
    session = AgentSession(tts=tts, ...)
"""

from __future__ import annotations

import asyncio
import struct
import logging
from dataclasses import dataclass, replace
from typing import Optional

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given


logger = logging.getLogger("chatterbox_tts")


@dataclass
class _TTSOptions:
    """Internal configuration for Chatterbox TTS."""
    api_url: str
    voice: str
    language: str
    exaggeration: float
    cfg_weight: float
    temperature: float
    sample_rate: int
    chunk_size: int

    def get_stream_url(self) -> str:
        return f"{self.api_url}/stream"

    def get_speak_url(self) -> str:
        return f"{self.api_url}/speak"

    def get_health_url(self) -> str:
        return f"{self.api_url}/health"


class ChatterboxTTS(tts.TTS):
    """
    Chatterbox TTS plugin for LiveKit Agents.
    
    Uses modern LiveKit TTS API with AudioEmitter for low-latency streaming.
    Connects to your deployed Chatterbox TTS API (on Modal).
    """
    
    def __init__(
        self,
        api_url: str,
        voice: str = "default",
        language: str = "en",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        sample_rate: int = 24000,
        chunk_size: int = 200,
        http_session: Optional[aiohttp.ClientSession] = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
    ):
        """
        Create a new instance of Chatterbox TTS.

        Args:
            api_url: The base URL for your Modal Chatterbox TTS deployment.
            voice: Voice ID to use. Defaults to "default".
            language: Language code. Defaults to "en".
            exaggeration: Exaggeration level for prosody. Defaults to 0.5.
            cfg_weight: CFG weight for generation. Defaults to 0.5.
            temperature: Temperature for sampling. Defaults to 0.8.
            sample_rate: Audio sample rate in Hz. Defaults to 24000.
            chunk_size: Streaming chunk size in samples. Defaults to 200.
            http_session: Optional aiohttp session to reuse.
            tokenizer: Optional sentence tokenizer. Uses basic tokenizer by default.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=False,  # Chatterbox doesn't support word timestamps
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )
        
        self._opts = _TTSOptions(
            api_url=api_url.rstrip("/"),
            voice=voice,
            language=language,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
        )
        
        self._session = http_session
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.basic.SentenceTokenizer()
        )

    @property
    def provider(self) -> str:
        return "Chatterbox"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def prewarm(self) -> None:
        """Prewarm the TTS by making a health check request."""
        asyncio.create_task(self._do_prewarm())

    async def _do_prewarm(self) -> None:
        """Async prewarm implementation."""
        try:
            async with self._ensure_session().get(
                self._opts.get_health_url(),
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    logger.info("Chatterbox TTS prewarm successful")
                else:
                    logger.warning(f"Chatterbox TTS prewarm returned {resp.status}")
        except Exception as e:
            logger.warning(f"Chatterbox TTS prewarm failed: {e}")

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        exaggeration: NotGivenOr[float] = NOT_GIVEN,
        cfg_weight: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Update TTS options dynamically."""
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language
        if is_given(exaggeration):
            self._opts.exaggeration = exaggeration
        if is_given(cfg_weight):
            self._opts.cfg_weight = cfg_weight
        if is_given(temperature):
            self._opts.temperature = temperature

    def synthesize(
        self, 
        text: str, 
        *, 
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "ChunkedStream":
        """Synthesize speech from text (non-streaming)."""
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, 
        *, 
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "SynthesizeStream":
        """Create a streaming TTS session."""
        return SynthesizeStream(tts=self, conn_options=conn_options)

    async def aclose(self) -> None:
        """Clean up resources."""
        pass  # Session is managed by http_context


class ChunkedStream(tts.ChunkedStream):
    """Non-streaming TTS using the /speak endpoint."""

    def __init__(
        self, 
        *, 
        tts: ChatterboxTTS, 
        input_text: str, 
        conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Execute the synthesis and emit audio."""
        params = {
            "text": self._input_text,
            "language": self._opts.language,
            "exaggeration": self._opts.exaggeration,
        }
        if self._opts.voice and self._opts.voice != "default":
            params["voice"] = self._opts.voice

        try:
            async with self._tts._ensure_session().post(
                self._opts.get_speak_url(),
                params=params,
                timeout=aiohttp.ClientTimeout(
                    total=60, 
                    sock_connect=self._conn_options.timeout
                ),
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise APIStatusError(
                        message=f"Chatterbox API error: {error}",
                        status_code=resp.status,
                        request_id=None,
                        body=None
                    )

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                )
                
                # Start a segment before pushing audio (required by LiveKit)
                output_emitter.start_segment(segment_id=utils.shortuuid())

                # Parse WAV and extract PCM
                data = await resp.read()
                pcm_data = self._extract_pcm_from_wav(data)
                output_emitter.push(pcm_data)
                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, 
                status_code=e.status, 
                request_id=None, 
                body=None
            ) from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

    def _extract_pcm_from_wav(self, wav_data: bytes) -> bytes:
        """Extract raw PCM data from WAV file, skipping header."""
        if len(wav_data) < 44:
            return wav_data
        return wav_data[44:]


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming TTS using the /stream endpoint with AudioEmitter."""

    def __init__(
        self, 
        *, 
        tts: ChatterboxTTS, 
        conn_options: APIConnectOptions
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Execute streaming synthesis with low-latency audio emission."""
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        
        # Start a segment before pushing any audio (required by LiveKit)
        output_emitter.start_segment(segment_id=segment_id)

        # HYBRID OPT: Start TTS on first sentence for low TTFB,
        # but batch remaining text to minimize API calls
        
        # Use sentence tokenizer to detect sentence boundaries
        sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()
        
        async def _input_task() -> None:
            """Collect input text and tokenize into sentences."""
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue
                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _synthesis_task() -> None:
            """
            Hybrid strategy for optimal latency/compute balance:
            1. Start TTS immediately on first sentence (low TTFB)
            2. Batch ALL remaining sentences into one request (1x compute for rest)
            """
            first_sentence_sent = False
            remaining_text = ""
            
            async for ev in sent_tokenizer_stream:
                sentence = ev.token.strip()
                if not sentence:
                    continue
                    
                if not first_sentence_sent:
                    # First sentence: send immediately for low TTFB
                    self._mark_started()
                    logger.debug(f"First sentence (fast path): '{sentence}'")
                    await self._synthesize_and_emit(
                        sentence, 
                        output_emitter,
                        request_id
                    )
                    first_sentence_sent = True
                else:
                    # Remaining sentences: batch them
                    remaining_text += " " + sentence
            
            # Send batched remaining text as single request
            if remaining_text.strip():
                logger.debug(f"Batched remaining ({len(remaining_text)} chars)")
                await self._synthesize_and_emit(
                    remaining_text.strip(),
                    output_emitter,
                    request_id
                )

        try:
            # Run input collection and synthesis concurrently
            input_task = asyncio.create_task(_input_task())
            synthesis_task = asyncio.create_task(_synthesis_task())

            try:
                await asyncio.gather(input_task, synthesis_task)
            finally:
                await sent_tokenizer_stream.aclose()
                await utils.aio.gracefully_cancel(input_task, synthesis_task)

            output_emitter.end_input()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, 
                status_code=e.status, 
                request_id=None, 
                body=None
            ) from None
        except Exception as e:
            logger.exception("Chatterbox streaming error")
            raise APIConnectionError() from e

    async def _synthesize_and_emit(
        self, 
        text: str, 
        output_emitter: tts.AudioEmitter,
        request_id: str
    ) -> None:
        """Synthesize a text chunk and emit audio frames immediately."""
        params = {
            "text": text,
            "language": self._opts.language,
            "exaggeration": self._opts.exaggeration,
            "chunk_size": self._opts.chunk_size,
        }
        if self._opts.voice and self._opts.voice != "default":
            params["voice"] = self._opts.voice

        logger.debug(f"Synthesizing: '{text[:50]}...' ({len(text)} chars)")

        try:
            async with self._tts._ensure_session().post(
                self._opts.get_stream_url(),
                params=params,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Chatterbox API error {response.status}: {error}")
                    raise APIStatusError(
                        message=error,
                        status_code=response.status,
                        request_id=request_id,
                        body=None
                    )

                # Stream audio with minimal buffering for low latency
                header_buffer = b""
                header_parsed = False
                residual_bytes = b""
                
                # Frame settings
                sample_rate = self._opts.sample_rate
                bytes_per_sample = 2  # 16-bit PCM
                frame_duration_ms = 20  # 20ms frames for low latency
                samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
                bytes_per_frame = samples_per_frame * bytes_per_sample

                async for chunk in response.content.iter_chunked(1024):
                    current_data = chunk

                    # Parse WAV header (first 44 bytes)
                    if not header_parsed:
                        header_buffer += current_data
                        if len(header_buffer) >= 44:
                            # Extract sample rate from header if different
                            try:
                                parsed_rate = struct.unpack('<I', header_buffer[24:28])[0]
                                if parsed_rate != sample_rate:
                                    sample_rate = parsed_rate
                                    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
                                    bytes_per_frame = samples_per_frame * bytes_per_sample
                            except Exception:
                                pass  # Use defaults
                            
                            header_parsed = True
                            current_data = header_buffer[44:]
                        else:
                            continue

                    # Emit PCM frames immediately
                    data_to_process = residual_bytes + current_data
                    offset = 0

                    while offset + bytes_per_frame <= len(data_to_process):
                        frame_data = data_to_process[offset:offset + bytes_per_frame]
                        output_emitter.push(frame_data)
                        offset += bytes_per_frame

                    residual_bytes = data_to_process[offset:]

                # Emit any remaining audio (padded if needed)
                if residual_bytes:
                    if len(residual_bytes) < bytes_per_frame:
                        residual_bytes += b'\x00' * (bytes_per_frame - len(residual_bytes))
                    output_emitter.push(residual_bytes)

        except APIStatusError:
            raise
        except Exception as e:
            logger.error(f"Synthesis failed for text: {e}")
            raise


# =============================================================================
# Helper functions
# =============================================================================

async def list_voices(api_url: str) -> list:
    """Get list of available voices from the API."""
    url = f"{api_url.rstrip('/')}/voices"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("voices", [])
            return []


async def list_languages(api_url: str) -> list:
    """Get list of supported languages from the API."""
    url = f"{api_url.rstrip('/')}/languages"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("languages", [])
            return []
