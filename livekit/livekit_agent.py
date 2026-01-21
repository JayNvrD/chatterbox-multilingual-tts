"""
Example LiveKit Agent using Chatterbox TTS

This demonstrates how to use the Chatterbox TTS plugin with LiveKit agents
for real-time voice applications.

Requirements:
    pip install livekit-agents livekit-plugins-deepgram livekit-plugins-openai aiohttp

Environment Variables:
    LIVEKIT_URL=wss://your-project.livekit.cloud
    LIVEKIT_API_KEY=your_api_key
    LIVEKIT_API_SECRET=your_api_secret
    CHATTERBOX_API_URL=https://your-modal-url.modal.run
    OPENAI_API_KEY=your_openai_key (for LLM)
    DEEPGRAM_API_KEY=your_deepgram_key (for STT)

Run:
    python livekit_agent.py dev
"""

import os
import asyncio
from livekit import agents, rtc
from livekit.agents import AgentSession, AgentServer
from livekit.plugins import deepgram, openai

# Import the Chatterbox TTS plugin
from chatterbox_tts import ChatterboxTTS, list_voices


# =============================================================================
# Configuration
# =============================================================================

# Your deployed Chatterbox TTS API URL (from Modal)
CHATTERBOX_API_URL = os.getenv(
    "CHATTERBOX_API_URL", 
    "https://jayant--chatterbox-tts-web-app.modal.run"
)

# Default voice to use
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "priya_hi_female")

# Language for the voice
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "hi")


# =============================================================================
# Agent Server
# =============================================================================

server = AgentServer()


@server.rtc_session()
async def voice_agent(ctx: agents.JobContext):
    """
    A voice agent that uses Chatterbox TTS for speech synthesis.
    """
    
    # Connect to the room
    await ctx.connect()
    
    # Wait for a participant to join
    participant = await ctx.wait_for_participant()
    print(f"Participant joined: {participant.identity}")
    
    # Initialize components
    
    # Speech-to-Text (using Deepgram)
    stt = deepgram.STT()
    
    # Large Language Model (using OpenAI)
    llm = openai.LLM(model="gpt-4o-mini")
    
    # Text-to-Speech (using Chatterbox!)
    tts = ChatterboxTTS(
        api_url=CHATTERBOX_API_URL,
        voice=DEFAULT_VOICE,
        language=DEFAULT_LANGUAGE,
        exaggeration=0.6,  # Slightly more expressive
    )
    
    # Create agent session
    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
    )
    
    # Start the session
    await session.start(ctx.room, participant)
    
    # Send initial greeting
    await session.say(f"Hello! I'm your AI assistant. How can I help you today?")
    
    print("Agent is now listening...")


# =============================================================================
# Standalone TTS Example (without full agent)
# =============================================================================

async def standalone_tts_example():
    """
    Example of using Chatterbox TTS as a standalone component.
    This is useful for testing or simpler use cases.
    """
    
    print("=== Standalone TTS Example ===\n")
    
    # Initialize TTS
    tts = ChatterboxTTS(
        api_url=CHATTERBOX_API_URL,
        voice="john_en_male",
        language="en",
    )
    
    # List available voices
    print("Available voices:")
    voices = await list_voices(CHATTERBOX_API_URL)
    for v in voices:
        lang = v.get("metadata", {}).get("language", "?")
        print(f"  - {v['name']} [{lang}]")
    print()
    
    # Simple synthesis (non-streaming)
    print("Synthesizing: 'Hello from Chatterbox TTS!'")
    audio_bytes = await tts.synthesize("Hello from Chatterbox TTS!")
    
    # Save to file
    with open("test_output.wav", "wb") as f:
        f.write(audio_bytes)
    print(f"Saved to test_output.wav ({len(audio_bytes)} bytes)\n")
    
    # Streaming synthesis
    print("Streaming synthesis...")
    stream = tts.stream()
    stream.push_text("This is a streaming test. ")
    stream.push_text("The audio is generated in real-time. ")
    stream.push_text("Pretty cool, right?")
    stream.end_input()
    
    frame_count = 0
    async for audio in stream:
        frame_count += 1
        if audio.is_final:
            print(f"Received {frame_count} audio frames (final)")
    
    print("\nDone!")


# =============================================================================
# Multi-language Example
# =============================================================================

async def multilingual_example():
    """
    Example showing multilingual TTS capabilities.
    """
    
    print("=== Multilingual TTS Example ===\n")
    
    # Define voices and texts for each language
    examples = [
        {
            "voice": "john_en_male",
            "language": "en",
            "text": "Hello! This is English speech.",
            "output": "english_output.wav",
        },
        {
            "voice": "priya_hi_female",
            "language": "hi",
            "text": "नमस्ते! यह हिंदी में बोल रहा है।",
            "output": "hindi_output.wav",
        },
        {
            "voice": "maria_es_female",
            "language": "es",
            "text": "¡Hola! Esto es español.",
            "output": "spanish_output.wav",
        },
        {
            "voice": "omar_ar_male",
            "language": "ar",
            "text": "مرحبا! هذه اللغة العربية.",
            "output": "arabic_output.wav",
        },
    ]
    
    for ex in examples:
        print(f"Generating {ex['language'].upper()}: {ex['text'][:30]}...")
        
        tts = ChatterboxTTS(
            api_url=CHATTERBOX_API_URL,
            voice=ex["voice"],
            language=ex["language"],
        )
        
        try:
            audio = await tts.synthesize(ex["text"])
            with open(ex["output"], "wb") as f:
                f.write(audio)
            print(f"  ✓ Saved to {ex['output']}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\nDone!")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run standalone test
        asyncio.run(standalone_tts_example())
    elif len(sys.argv) > 1 and sys.argv[1] == "multilingual":
        # Run multilingual test
        asyncio.run(multilingual_example())
    else:
        # Run the LiveKit agent
        agents.cli.run_app(server)
