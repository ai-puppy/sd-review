# System Design: 24/7 Voice AI Intake System for Law Firms

## Table of Contents
1. [Problem Statement & Requirements](#1-problem-statement--requirements)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Real-Time Voice Pipeline](#3-real-time-voice-pipeline)
4. [Conversational AI Engine](#4-conversational-ai-engine)
5. [Case Qualification & Scoring](#5-case-qualification--scoring)
6. [Attorney-Client Privilege Protection](#6-attorney-client-privilege-protection)
7. [CMS Integration](#7-cms-integration)
8. [Human Handoff System](#8-human-handoff-system)
9. [Multi-Language & Accent Handling](#9-multi-language--accent-handling)
10. [Scale & Performance](#10-scale--performance)
11. [Monitoring & Quality Assurance](#11-monitoring--quality-assurance)
12. [Interview Discussion Points](#12-interview-discussion-points)

---

## 1. Problem Statement & Requirements

### The Business Problem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY LAW FIRMS NEED AI INTAKE                              │
└─────────────────────────────────────────────────────────────────────────────┘

CURRENT STATE (Without AI Intake):
──────────────────────────────────
• Potential client calls at 2 AM after car accident
• Goes to voicemail → calls competitor instead
• Firm loses $50,000+ case

• Intake staff overwhelmed during business hours
• 40% of calls go unanswered or abandoned
• No consistent qualification process

• Spanish-speaking client calls
• Receptionist doesn't speak Spanish
• Client hangs up, calls bilingual competitor

DESIRED STATE (With AI Intake):
───────────────────────────────
• 24/7/365 availability - never miss a lead
• Consistent qualification every time
• Multilingual support
• Instant case scoring and prioritization
• Seamless CRM/CMS sync
• Human handoff when needed
• All conversations properly privileged
```

### Functional Requirements

| Requirement | Description |
|-------------|-------------|
| **24/7 Availability** | Handle calls any time, including holidays |
| **Natural Conversation** | Feel like talking to a person, not an IVR |
| **Intake Interview** | Gather: contact info, incident details, injuries, at-fault party, witnesses |
| **Case Scoring** | Real-time qualification score (1-100) based on case viability |
| **Language Support** | English, Spanish minimum; expandable to others |
| **Accent Handling** | Robust ASR across regional accents |
| **Privilege Protection** | Ensure recordings maintain attorney-client privilege |
| **CMS Sync** | Auto-create leads in Clio, Filevine, Litify, etc. |
| **Human Handoff** | Warm transfer to attorney when needed |
| **Callback Scheduling** | Schedule callbacks if client prefers |
| **SMS Follow-up** | Send confirmation texts with next steps |

### Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| **Latency** | < 500ms response time for conversational flow |
| **Accuracy** | > 95% speech recognition accuracy |
| **Availability** | 99.99% uptime (< 52 min downtime/year) |
| **Scalability** | Handle 1000+ concurrent calls across all tenants |
| **Compliance** | TCPA, state bar rules, HIPAA (for medical info) |
| **Security** | Encrypted calls, secure transcript storage |

### Latency Budget Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONVERSATIONAL AI LATENCY BUDGET                          │
└─────────────────────────────────────────────────────────────────────────────┘

Human expectation: Response within 500-1000ms feels "natural"
                   > 2000ms feels like the system is "thinking" or broken

Target: < 800ms end-to-end (leaves buffer for network variability)

┌──────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│  User speaks    STT        LLM         TTS        Audio      Total       │
│  ──────────    ────       ────        ────       ─────      ─────       │
│                                                                           │
│  "I was in    →  150ms  →  300ms  →  150ms  →  100ms  =   700ms        │
│   a car          (Deepgram) (GPT-4    (ElevenLabs) (streaming)           │
│   accident"                 streaming)                                    │
│                                                                           │
│  ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←      │
│                                                                           │
│  "I'm so sorry to hear that. Are you okay? Can you tell me..."          │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘

Key techniques to hit latency targets:
• Streaming STT (transcribe as user speaks)
• Streaming LLM (start generating before full transcript)
• Streaming TTS (start speaking before full response)
• Sentence-level pipelining
• Regional deployment (minimize network hops)
```

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VOICE AI INTAKE SYSTEM ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

                              PSTN / SIP
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TELEPHONY LAYER                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Twilio    │  │   Telnyx    │  │  Bandwidth  │  │   Plivo     │        │
│  │  (Primary)  │  │  (Backup)   │  │  (Backup)   │  │  (Intl)     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MEDIA STREAM ROUTER                               │   │
│  │  • WebSocket connection management                                   │   │
│  │  • Audio stream multiplexing                                        │   │
│  │  • Tenant routing (phone number → tenant)                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VOICE PROCESSING LAYER                                  │
│                                                                              │
│  ┌──────────────────┐         ┌──────────────────┐                         │
│  │   SPEECH-TO-TEXT │         │   TEXT-TO-SPEECH │                         │
│  │                  │         │                  │                         │
│  │  • Deepgram      │         │  • ElevenLabs    │                         │
│  │  • Streaming     │         │  • Low-latency   │                         │
│  │  • Multi-lang    │         │  • Voice cloning │                         │
│  │  • Diarization   │         │  • Streaming     │                         │
│  └────────┬─────────┘         └────────▲─────────┘                         │
│           │                            │                                    │
│           ▼                            │                                    │
│  ┌─────────────────────────────────────┴────────────────────────────────┐  │
│  │                    CONVERSATION ORCHESTRATOR                          │  │
│  │                                                                       │  │
│  │  • Turn management (who's speaking)                                  │  │
│  │  • Interrupt handling                                                │  │
│  │  • Silence detection                                                 │  │
│  │  • Barge-in support                                                  │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI LAYER                                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CONVERSATIONAL AI ENGINE                          │   │
│  │                                                                      │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │   │
│  │  │    Intent     │  │    Dialog     │  │   Response    │           │   │
│  │  │   Classifier  │  │    Manager    │  │   Generator   │           │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘           │   │
│  │                                                                      │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │   │
│  │  │    Entity     │  │     Case      │  │   Sentiment   │           │   │
│  │  │   Extractor   │  │    Scorer     │  │   Analyzer    │           │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  LLM Backend: GPT-4 Turbo (streaming) | Claude 3.5 Sonnet (backup)         │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA & INTEGRATION LAYER                             │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Transcript │  │    CMS      │  │   Callback  │  │    SMS      │        │
│  │   Storage   │  │ Integration │  │  Scheduler  │  │   Service   │        │
│  │  (Encrypted)│  │  (Clio,etc) │  │             │  │  (Twilio)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │   Human     │  │   Quality   │  │   Audit     │                         │
│  │  Handoff    │  │  Analytics  │  │    Log      │                         │
│  │   Queue     │  │             │  │             │                         │
│  └─────────────┘  └─────────────┘  └─────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Call Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INTAKE CALL FLOW                                     │
└─────────────────────────────────────────────────────────────────────────────┘

  Potential Client                          AI Intake Agent
       │                                          │
       │  ──── Dials law firm number ────────▶   │
       │                                          │
       │  ◀──── "Thank you for calling          │
       │         Smith & Associates.             │
       │         I'm Alex, a virtual             │
       │         assistant. How can I help?" ───│
       │                                          │
       │  ──── "I was in a car accident         │
       │        yesterday and I think I          │
       │        need a lawyer" ─────────────▶   │
       │                                          │
       │                                    ┌─────┴─────┐
       │                                    │  Intent:  │
       │                                    │  PI Intake│
       │                                    │  Score: 75│
       │                                    └─────┬─────┘
       │                                          │
       │  ◀──── "I'm so sorry to hear that.      │
       │         I'd like to help you speak       │
       │         with one of our attorneys.       │
       │         First, can I get your name?" ───│
       │                                          │
       │  ──── "Maria Garcia" ──────────────▶   │
       │                                          │
       │  ◀──── "Thank you, Maria. And a         │
       │         phone number where we can        │
       │         reach you?" ────────────────────│
       │                                          │
       │  ──── "555-123-4567" ──────────────▶   │
       │                                          │
       │  ◀──── "Got it. Can you tell me         │
       │         more about the accident?         │
       │         When and where did it happen?" ─│
       │                                          │
       │  ──── "Yesterday around 5pm on          │
       │        Highway 101. The other            │
       │        driver ran a red light" ────▶    │
       │                                          │
       │                                    ┌─────┴─────┐
       │                                    │  Score: 85│
       │                                    │  Clear    │
       │                                    │  liability│
       │                                    └─────┬─────┘
       │                                          │
       │  ◀──── "Were you injured? Did you       │
       │         seek medical treatment?" ────────│
       │                                          │
       │  ──── "Yes, I went to the ER.           │
       │        They said I have whiplash        │
       │        and a concussion" ───────────▶   │
       │                                          │
       │                                    ┌─────┴─────┐
       │                                    │  Score: 92│
       │                                    │  Document-│
       │                                    │  ed injury│
       │                                    │  HIGH PRI │
       │                                    └─────┬─────┘
       │                                          │
       │  ◀──── "I've captured all the details.  │
       │         This sounds like a strong case.  │
       │         Would you like me to connect     │
       │         you with an attorney now, or     │
       │         schedule a callback?" ───────────│
       │                                          │
       │  ──── "I can talk now" ─────────────▶   │
       │                                          │
       │  ◀──── "Perfect. I'm transferring you   │
       │         to Attorney Johnson who          │
       │         specializes in car accidents.    │
       │         One moment please..." ───────────│
       │                                          │
       │  ════════ WARM TRANSFER ═══════════════ │
       │                                          │
       │  Attorney joins with full context:      │
       │  • Caller: Maria Garcia                 │
       │  • Case: MVA, clear liability           │
       │  • Injuries: Whiplash, concussion       │
       │  • Score: 92/100                        │
       │                                          │
```

---

## 3. Real-Time Voice Pipeline

### Streaming Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REAL-TIME STREAMING PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘

        Caller                                              AI Agent
          │                                                    │
          │  Audio Stream (8kHz μ-law or 16kHz PCM)           │
          │ ─────────────────────────────────────────────────▶│
          │                                                    │
          │     ┌──────────────────────────────────────────────┤
          │     │                                              │
          │     ▼                                              │
          │  ┌──────────────────────────────────────────────┐ │
          │  │              AUDIO BUFFER                     │ │
          │  │  • 20ms chunks                               │ │
          │  │  • Jitter buffer (40-100ms)                  │ │
          │  │  • Packet loss concealment                   │ │
          │  └──────────────────┬───────────────────────────┘ │
          │                     │                              │
          │                     ▼                              │
          │  ┌──────────────────────────────────────────────┐ │
          │  │           VOICE ACTIVITY DETECTION           │ │
          │  │  • Detect speech vs silence                  │ │
          │  │  • Determine end-of-utterance               │ │
          │  │  • Handle background noise                   │ │
          │  └──────────────────┬───────────────────────────┘ │
          │                     │                              │
          │                     ▼                              │
          │  ┌──────────────────────────────────────────────┐ │
          │  │         STREAMING STT (Deepgram)             │ │
          │  │                                              │ │
          │  │  Audio chunks ──▶ Interim transcripts       │ │
          │  │                   (low confidence)           │ │
          │  │                          │                   │ │
          │  │                          ▼                   │ │
          │  │                   Final transcript          │ │
          │  │                   (high confidence)          │ │
          │  │                   + word timestamps          │ │
          │  └──────────────────┬───────────────────────────┘ │
          │                     │                              │
          │                     ▼                              │
          │  ┌──────────────────────────────────────────────┐ │
          │  │              LLM (Streaming)                  │ │
          │  │                                              │ │
          │  │  • Start generating on interim transcript   │ │
          │  │  • Adjust if final transcript differs       │ │
          │  │  • Stream tokens as generated               │ │
          │  └──────────────────┬───────────────────────────┘ │
          │                     │                              │
          │                     ▼                              │
          │  ┌──────────────────────────────────────────────┐ │
          │  │         STREAMING TTS (ElevenLabs)           │ │
          │  │                                              │ │
          │  │  • Start synthesis on first sentence        │ │
          │  │  • Buffer sentences for smooth playback     │ │
          │  │  • Handle interruptions (barge-in)          │ │
          │  └──────────────────┬───────────────────────────┘ │
          │                     │                              │
          │  ◀──────────────────┘                              │
          │    Audio Stream (response)                        │
          │                                                    │
```

### Speech-to-Text Implementation

```python
import asyncio
from deepgram import Deepgram
from dataclasses import dataclass
from typing import AsyncIterator, Optional
import json

@dataclass
class TranscriptSegment:
    text: str
    is_final: bool
    confidence: float
    start_time: float
    end_time: float
    words: list  # Word-level timestamps for alignment
    speaker: Optional[int] = None  # For diarization

class StreamingSTT:
    """
    Real-time speech-to-text with streaming transcription.
    """
    
    def __init__(self, deepgram_api_key: str):
        self.dg = Deepgram(deepgram_api_key)
        self.current_transcript = ""
        self.final_segments = []
        
    async def create_stream(
        self, 
        language: str = "en-US",
        model: str = "nova-2",  # Best accuracy/speed balance
        sample_rate: int = 8000,  # Telephony standard
    ) -> 'DeepgramStream':
        """
        Create a streaming transcription connection.
        """
        options = {
            "model": model,
            "language": language,
            "smart_format": True,  # Punctuation, formatting
            "interim_results": True,  # Get partial transcripts
            "utterance_end_ms": 1000,  # Silence threshold
            "vad_events": True,  # Voice activity detection
            "diarize": True,  # Speaker identification
            "punctuate": True,
            "sample_rate": sample_rate,
            "channels": 1,
            "encoding": "mulaw" if sample_rate == 8000 else "linear16",
        }
        
        # Create live transcription connection
        connection = await self.dg.transcription.live(options)
        
        return DeepgramStream(connection, self)
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptSegment]:
        """
        Transcribe streaming audio and yield transcript segments.
        """
        stream = await self.create_stream()
        
        async def send_audio():
            async for chunk in audio_stream:
                await stream.send(chunk)
            await stream.finish()
        
        # Start sending audio in background
        send_task = asyncio.create_task(send_audio())
        
        try:
            async for segment in stream.receive():
                yield segment
        finally:
            send_task.cancel()


class DeepgramStream:
    """
    Wrapper for Deepgram streaming connection.
    """
    
    def __init__(self, connection, parent: StreamingSTT):
        self.connection = connection
        self.parent = parent
        self.transcript_queue = asyncio.Queue()
        
        # Set up event handlers
        connection.registerHandler(
            connection.event.TRANSCRIPT_RECEIVED,
            self._handle_transcript
        )
        connection.registerHandler(
            connection.event.UTTERANCE_END,
            self._handle_utterance_end
        )
    
    async def _handle_transcript(self, data):
        """Handle incoming transcript data."""
        transcript = data.get("channel", {}).get("alternatives", [{}])[0]
        
        segment = TranscriptSegment(
            text=transcript.get("transcript", ""),
            is_final=data.get("is_final", False),
            confidence=transcript.get("confidence", 0),
            start_time=data.get("start", 0),
            end_time=data.get("start", 0) + data.get("duration", 0),
            words=transcript.get("words", []),
            speaker=data.get("channel", {}).get("speaker"),
        )
        
        await self.transcript_queue.put(segment)
    
    async def _handle_utterance_end(self, data):
        """Handle end of utterance (silence detected)."""
        # Signal that user has stopped speaking
        await self.transcript_queue.put(
            TranscriptSegment(
                text="",
                is_final=True,
                confidence=1.0,
                start_time=0,
                end_time=0,
                words=[],
            )
        )
    
    async def send(self, audio_chunk: bytes):
        """Send audio chunk to Deepgram."""
        await self.connection.send(audio_chunk)
    
    async def receive(self) -> AsyncIterator[TranscriptSegment]:
        """Receive transcript segments."""
        while True:
            segment = await self.transcript_queue.get()
            yield segment
            if segment.is_final and not segment.text:
                break
    
    async def finish(self):
        """Signal end of audio stream."""
        await self.connection.finish()
```

### Text-to-Speech Implementation

```python
from elevenlabs import ElevenLabs, Voice, VoiceSettings
from typing import AsyncIterator
import asyncio

class StreamingTTS:
    """
    Low-latency text-to-speech with streaming output.
    """
    
    # Pre-configured voices for intake
    VOICES = {
        "professional_female": "EXAVITQu4vr4xnSDxMaL",  # Rachel
        "professional_male": "VR6AewLTigWG4xSOukaG",    # Arnold
        "friendly_female": "21m00Tcm4TlvDq8ikWAM",      # Rachel
        "friendly_male": "yoZ06aMxZJJ28mfd3POQ",        # Sam
    }
    
    def __init__(self, api_key: str):
        self.client = ElevenLabs(api_key=api_key)
        self.voice_settings = VoiceSettings(
            stability=0.75,      # More consistent
            similarity_boost=0.75,
            style=0.5,           # Balanced
            use_speaker_boost=True,
        )
    
    async def synthesize_streaming(
        self,
        text: str,
        voice: str = "professional_female",
        model: str = "eleven_turbo_v2",  # Lowest latency
    ) -> AsyncIterator[bytes]:
        """
        Stream synthesized audio as it's generated.
        
        Key optimization: Start playing audio before full synthesis.
        """
        voice_id = self.VOICES.get(voice, voice)
        
        # Use streaming API
        audio_stream = self.client.text_to_speech.convert_as_stream(
            text=text,
            voice_id=voice_id,
            model_id=model,
            voice_settings=self.voice_settings,
            output_format="pcm_16000",  # Raw PCM for low latency
        )
        
        async for chunk in audio_stream:
            yield chunk
    
    async def synthesize_sentence_by_sentence(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        """
        Synthesize sentences as they're generated by LLM.
        
        This enables speaking the first sentence while
        later sentences are still being generated.
        """
        sentence_buffer = ""
        sentence_endings = {'.', '!', '?', ':', ';'}
        
        async for token in text_stream:
            sentence_buffer += token
            
            # Check if we have a complete sentence
            for ending in sentence_endings:
                if ending in sentence_buffer:
                    # Split at sentence boundary
                    parts = sentence_buffer.split(ending, 1)
                    sentence = parts[0] + ending
                    sentence_buffer = parts[1] if len(parts) > 1 else ""
                    
                    # Synthesize and yield this sentence
                    if sentence.strip():
                        async for audio_chunk in self.synthesize_streaming(sentence):
                            yield audio_chunk
                    break
        
        # Don't forget remaining text
        if sentence_buffer.strip():
            async for audio_chunk in self.synthesize_streaming(sentence_buffer):
                yield audio_chunk
```

### Conversation Orchestrator

```python
import asyncio
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable

class ConversationState(Enum):
    LISTENING = "listening"      # User is speaking
    PROCESSING = "processing"    # Generating response
    SPEAKING = "speaking"        # AI is speaking
    INTERRUPTED = "interrupted"  # User interrupted AI

@dataclass
class TurnContext:
    state: ConversationState
    user_transcript: str
    ai_response: str
    turn_number: int
    start_time: float
    end_time: Optional[float] = None

class ConversationOrchestrator:
    """
    Manages turn-taking and interruption handling.
    
    Key challenges:
    - Detect when user has finished speaking
    - Handle user interrupting AI (barge-in)
    - Manage overlapping speech
    """
    
    def __init__(
        self,
        stt: StreamingSTT,
        tts: StreamingTTS,
        ai_engine: 'ConversationalAI',
        on_turn_complete: Optional[Callable] = None,
    ):
        self.stt = stt
        self.tts = tts
        self.ai = ai_engine
        self.on_turn_complete = on_turn_complete
        
        self.state = ConversationState.LISTENING
        self.current_turn = 0
        self.is_ai_speaking = False
        self.interrupt_event = asyncio.Event()
    
    async def handle_audio_stream(
        self,
        inbound_audio: AsyncIterator[bytes],
        outbound_audio: asyncio.Queue,
    ):
        """
        Main loop handling bidirectional audio.
        """
        async for segment in self.stt.transcribe_stream(inbound_audio):
            # Check for user interruption while AI is speaking
            if self.is_ai_speaking and segment.text.strip():
                await self._handle_interruption()
            
            # Accumulate transcript until user stops speaking
            if segment.is_final and segment.text.strip():
                # User finished speaking
                user_input = segment.text
                
                self.state = ConversationState.PROCESSING
                self.current_turn += 1
                
                # Generate and speak response
                await self._generate_and_speak(
                    user_input,
                    outbound_audio,
                )
                
                self.state = ConversationState.LISTENING
    
    async def _generate_and_speak(
        self,
        user_input: str,
        outbound_audio: asyncio.Queue,
    ):
        """
        Generate AI response and stream to TTS.
        """
        self.is_ai_speaking = True
        self.interrupt_event.clear()
        
        try:
            # Get streaming response from LLM
            response_stream = self.ai.generate_response_streaming(user_input)
            
            # Synthesize and send audio
            async for audio_chunk in self.tts.synthesize_sentence_by_sentence(response_stream):
                # Check for interruption
                if self.interrupt_event.is_set():
                    break
                
                await outbound_audio.put(audio_chunk)
        
        finally:
            self.is_ai_speaking = False
    
    async def _handle_interruption(self):
        """
        Handle user interrupting the AI.
        
        Strategies:
        1. Immediate stop: Stop speaking immediately
        2. Graceful stop: Finish current sentence
        3. Acknowledge: "Go ahead, I'm listening"
        """
        self.state = ConversationState.INTERRUPTED
        self.interrupt_event.set()
        
        # Log interruption for analysis
        await self._log_event("user_interrupted", {
            "turn": self.current_turn,
            "ai_was_speaking": True,
        })


class BargeInDetector:
    """
    Detect user attempting to interrupt AI speech.
    
    Challenge: Distinguish between:
    - User intentionally interrupting
    - User saying "uh-huh" (backchannel)
    - Background noise
    - Echo from AI's own speech
    """
    
    # Backchannel phrases that don't indicate interruption
    BACKCHANNELS = {
        "uh-huh", "mm-hmm", "yeah", "okay", "right",
        "yes", "sure", "got it", "i see", "oh"
    }
    
    def __init__(self, echo_cancellation_enabled: bool = True):
        self.echo_cancellation = echo_cancellation_enabled
        self.energy_threshold = 0.02  # Minimum energy to consider speech
    
    def is_interruption(
        self,
        transcript: str,
        audio_energy: float,
        ai_is_speaking: bool,
    ) -> bool:
        """
        Determine if user speech is an interruption.
        """
        if not ai_is_speaking:
            return False
        
        # Filter low-energy audio (noise)
        if audio_energy < self.energy_threshold:
            return False
        
        # Filter backchannel responses
        normalized = transcript.lower().strip()
        if normalized in self.BACKCHANNELS:
            return False
        
        # Substantial speech while AI is talking = interruption
        if len(transcript.split()) >= 2:
            return True
        
        return False
```

---

## 4. Conversational AI Engine

### Dialog Manager

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json

class IntakeStage(Enum):
    GREETING = "greeting"
    IDENTIFY_NEED = "identify_need"
    COLLECT_CONTACT = "collect_contact"
    COLLECT_INCIDENT = "collect_incident"
    COLLECT_INJURIES = "collect_injuries"
    COLLECT_LIABILITY = "collect_liability"
    COLLECT_ADDITIONAL = "collect_additional"
    QUALIFY_CASE = "qualify_case"
    SCHEDULE_OR_TRANSFER = "schedule_or_transfer"
    CLOSING = "closing"

@dataclass
class IntakeData:
    """Structured data collected during intake."""
    # Contact info
    caller_name: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    preferred_contact_method: Optional[str] = None
    
    # Incident details
    case_type: Optional[str] = None  # "auto_accident", "slip_fall", "medical_mal", etc.
    incident_date: Optional[str] = None
    incident_location: Optional[str] = None
    incident_description: Optional[str] = None
    
    # Parties
    at_fault_party: Optional[str] = None
    at_fault_insurance: Optional[str] = None
    witnesses: List[str] = field(default_factory=list)
    police_report_filed: Optional[bool] = None
    
    # Injuries & treatment
    injuries: List[str] = field(default_factory=list)
    medical_treatment: List[str] = field(default_factory=list)
    ongoing_treatment: Optional[bool] = None
    
    # Case qualification
    statute_of_limitations_ok: Optional[bool] = None
    clear_liability: Optional[bool] = None
    documented_damages: Optional[bool] = None
    
    # Scoring
    case_score: int = 0
    priority: str = "normal"  # "low", "normal", "high", "urgent"
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ConversationalAI:
    """
    LLM-powered conversational agent for legal intake.
    """
    
    SYSTEM_PROMPT = """You are Alex, a professional and empathetic virtual intake specialist for {firm_name}, a personal injury law firm.

Your role is to:
1. Warmly greet callers and understand their legal needs
2. Collect essential information for case evaluation
3. Show empathy - these callers have often been through traumatic experiences
4. Qualify cases based on collected information
5. Either transfer to an attorney or schedule a callback

CONVERSATION GUIDELINES:
- Be warm, professional, and empathetic
- Use simple language (avoid legal jargon)
- Ask one question at a time
- Acknowledge emotions ("I'm sorry to hear that")
- Confirm important information by repeating it back
- Never provide legal advice - only collect information
- If caller seems distressed, offer to slow down or take a break

INFORMATION TO COLLECT:
1. Contact: Name, phone, email
2. Incident: Type, date, location, description
3. Injuries: What injuries, medical treatment received
4. Liability: Who was at fault, insurance info, police report
5. Witnesses: Names and contact info if available

CURRENT STAGE: {current_stage}
COLLECTED DATA: {collected_data}
CASE SCORE: {case_score}/100

Based on the conversation so far, continue the intake naturally. If you have enough information for the current stage, move to the next stage.

Remember: You are NOT an attorney. Do not give legal advice. Your job is to collect information and connect callers with attorneys."""

    def __init__(
        self,
        llm_client,
        firm_config: dict,
    ):
        self.llm = llm_client
        self.firm_name = firm_config.get("name", "our law firm")
        self.firm_practice_areas = firm_config.get("practice_areas", ["personal injury"])
        
        # Conversation state
        self.stage = IntakeStage.GREETING
        self.intake_data = IntakeData()
        self.conversation_history: List[Dict[str, str]] = []
        self.turn_count = 0
    
    async def generate_response_streaming(
        self,
        user_input: str,
    ) -> AsyncIterator[str]:
        """
        Generate streaming response to user input.
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Extract entities from user input
        entities = await self._extract_entities(user_input)
        self._update_intake_data(entities)
        
        # Update case score
        self.intake_data.case_score = self._calculate_case_score()
        
        # Determine if we should move to next stage
        self._maybe_advance_stage()
        
        # Build prompt
        system_prompt = self.SYSTEM_PROMPT.format(
            firm_name=self.firm_name,
            current_stage=self.stage.value,
            collected_data=json.dumps(self.intake_data.to_dict(), indent=2),
            case_score=self.intake_data.case_score,
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history
        ]
        
        # Generate streaming response
        full_response = ""
        async for token in self.llm.chat_streaming(messages):
            full_response += token
            yield token
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })
        
        self.turn_count += 1
    
    async def _extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract structured entities from user text.
        
        Uses LLM for robust extraction across varied phrasings.
        """
        extraction_prompt = f"""Extract the following information from the text if present. Return JSON only.

Text: "{text}"

Extract:
- name: person's name if mentioned
- phone: phone number if mentioned
- email: email if mentioned  
- date: any date mentioned (incident date)
- location: any location/address mentioned
- injuries: list of injuries mentioned
- at_fault: who was at fault if mentioned
- case_type: type of case (auto accident, slip and fall, etc.)

Return only valid JSON. Use null for fields not found."""

        response = await self.llm.chat([
            {"role": "user", "content": extraction_prompt}
        ], response_format={"type": "json_object"})
        
        try:
            return json.loads(response)
        except:
            return {}
    
    def _update_intake_data(self, entities: Dict[str, Any]):
        """Update intake data with extracted entities."""
        if entities.get("name"):
            self.intake_data.caller_name = entities["name"]
        if entities.get("phone"):
            self.intake_data.phone_number = entities["phone"]
        if entities.get("email"):
            self.intake_data.email = entities["email"]
        if entities.get("date"):
            self.intake_data.incident_date = entities["date"]
        if entities.get("location"):
            self.intake_data.incident_location = entities["location"]
        if entities.get("injuries"):
            self.intake_data.injuries.extend(entities["injuries"])
        if entities.get("at_fault"):
            self.intake_data.at_fault_party = entities["at_fault"]
        if entities.get("case_type"):
            self.intake_data.case_type = entities["case_type"]
    
    def _maybe_advance_stage(self):
        """Check if we should move to the next intake stage."""
        stage_requirements = {
            IntakeStage.GREETING: True,  # Always advance
            IntakeStage.IDENTIFY_NEED: self.intake_data.case_type is not None,
            IntakeStage.COLLECT_CONTACT: (
                self.intake_data.caller_name is not None and
                self.intake_data.phone_number is not None
            ),
            IntakeStage.COLLECT_INCIDENT: (
                self.intake_data.incident_date is not None and
                self.intake_data.incident_description is not None
            ),
            IntakeStage.COLLECT_INJURIES: len(self.intake_data.injuries) > 0,
            IntakeStage.COLLECT_LIABILITY: (
                self.intake_data.at_fault_party is not None
            ),
        }
        
        # Define stage order
        stage_order = list(IntakeStage)
        current_index = stage_order.index(self.stage)
        
        # Check if current stage is complete
        if stage_requirements.get(self.stage, False):
            if current_index < len(stage_order) - 1:
                self.stage = stage_order[current_index + 1]
    
    def _calculate_case_score(self) -> int:
        """
        Calculate case qualification score (0-100).
        
        Factors:
        - Clear liability (+30)
        - Documented injuries (+25)
        - Medical treatment (+20)
        - Within statute of limitations (+15)
        - Complete contact info (+10)
        """
        score = 0
        
        # Liability indicators
        liability_keywords = ["ran red light", "rear-ended", "hit me", "their fault", "drunk"]
        if self.intake_data.incident_description:
            if any(kw in self.intake_data.incident_description.lower() for kw in liability_keywords):
                score += 30
                self.intake_data.clear_liability = True
        
        # Injuries documented
        if len(self.intake_data.injuries) > 0:
            score += 25
            self.intake_data.documented_damages = True
        
        # Medical treatment
        treatment_keywords = ["hospital", "er", "emergency", "doctor", "surgery", "mri"]
        if self.intake_data.medical_treatment:
            for treatment in self.intake_data.medical_treatment:
                if any(kw in treatment.lower() for kw in treatment_keywords):
                    score += 20
                    break
        
        # Statute of limitations (simplified - would need actual date calc)
        if self.intake_data.incident_date:
            score += 15
            self.intake_data.statute_of_limitations_ok = True
        
        # Contact info complete
        if self.intake_data.caller_name and self.intake_data.phone_number:
            score += 10
        
        # Set priority based on score
        if score >= 80:
            self.intake_data.priority = "urgent"
        elif score >= 60:
            self.intake_data.priority = "high"
        elif score >= 40:
            self.intake_data.priority = "normal"
        else:
            self.intake_data.priority = "low"
        
        return min(score, 100)
```

### Intent Classification

```python
class IntentClassifier:
    """
    Classify caller intent to route appropriately.
    """
    
    INTENTS = {
        "new_case_pi": "Personal injury case inquiry",
        "new_case_other": "Other legal matter inquiry",
        "existing_client": "Existing client calling about their case",
        "billing_question": "Billing or payment question",
        "speak_to_attorney": "Wants to speak with specific attorney",
        "schedule_appointment": "Wants to schedule consultation",
        "general_question": "General question about the firm",
        "wrong_number": "Called wrong number",
        "spam": "Spam or sales call",
    }
    
    async def classify(self, transcript: str, context: dict = None) -> dict:
        """
        Classify intent from transcript.
        """
        prompt = f"""Classify the caller's intent from this transcript.

Transcript: "{transcript}"

Possible intents:
{json.dumps(self.INTENTS, indent=2)}

Respond with JSON:
{{
    "intent": "intent_key",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        response = await self.llm.chat([
            {"role": "user", "content": prompt}
        ], response_format={"type": "json_object"})
        
        return json.loads(response)
    
    def get_routing(self, intent: str) -> dict:
        """
        Get routing configuration for intent.
        """
        routing = {
            "new_case_pi": {
                "action": "intake",
                "priority": "high",
                "queue": "pi_intake",
            },
            "new_case_other": {
                "action": "intake",
                "priority": "normal",
                "queue": "general_intake",
            },
            "existing_client": {
                "action": "lookup_and_transfer",
                "queue": "existing_clients",
            },
            "billing_question": {
                "action": "transfer",
                "queue": "billing",
            },
            "speak_to_attorney": {
                "action": "transfer",
                "queue": "attorney_direct",
            },
            "spam": {
                "action": "end_call",
                "message": "Thank you for calling, goodbye.",
            },
        }
        
        return routing.get(intent, {"action": "intake", "queue": "general"})
```

---

## 5. Case Qualification & Scoring

### Scoring Model

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class CaseType(Enum):
    AUTO_ACCIDENT = "auto_accident"
    TRUCK_ACCIDENT = "truck_accident"
    MOTORCYCLE = "motorcycle"
    PEDESTRIAN = "pedestrian"
    SLIP_FALL = "slip_fall"
    MEDICAL_MALPRACTICE = "medical_malpractice"
    PRODUCT_LIABILITY = "product_liability"
    WORKERS_COMP = "workers_comp"
    WRONGFUL_DEATH = "wrongful_death"

@dataclass
class CaseScoreFactors:
    """Factors that influence case score."""
    
    # Liability (0-30 points)
    liability_clear: bool = False          # +20
    liability_documented: bool = False     # +10 (police report, witnesses)
    
    # Damages (0-35 points)
    injury_severity: int = 0               # 1-10 scale, multiply by 2
    medical_treatment: bool = False        # +5
    ongoing_treatment: bool = False        # +5
    lost_wages: bool = False               # +5
    
    # Case quality (0-20 points)
    within_sol: bool = True                # -100 if false (disqualify)
    insurance_coverage: bool = False       # +10
    no_prior_attorney: bool = True         # +5
    client_cooperation: int = 5            # 1-5 scale
    
    # Firm fit (0-15 points)
    in_practice_area: bool = True          # +10
    in_jurisdiction: bool = True           # +5


class CaseQualificationEngine:
    """
    Score and qualify intake cases.
    
    Used to:
    - Prioritize leads for attorney review
    - Route to appropriate practice group
    - Trigger immediate vs scheduled callbacks
    """
    
    # Severity scores by injury type
    INJURY_SEVERITY = {
        "death": 10,
        "traumatic brain injury": 9,
        "spinal cord injury": 9,
        "amputation": 9,
        "paralysis": 9,
        "multiple fractures": 8,
        "internal bleeding": 8,
        "surgery required": 7,
        "broken bones": 6,
        "concussion": 5,
        "herniated disc": 5,
        "torn ligament": 5,
        "whiplash": 4,
        "soft tissue injury": 3,
        "bruises": 2,
        "minor cuts": 1,
    }
    
    # Multipliers by case type (some cases are higher value)
    CASE_TYPE_MULTIPLIERS = {
        CaseType.WRONGFUL_DEATH: 1.5,
        CaseType.TRUCK_ACCIDENT: 1.3,
        CaseType.MEDICAL_MALPRACTICE: 1.3,
        CaseType.PRODUCT_LIABILITY: 1.2,
        CaseType.AUTO_ACCIDENT: 1.0,
        CaseType.SLIP_FALL: 0.9,
        CaseType.WORKERS_COMP: 0.8,
    }
    
    def __init__(self, firm_config: dict):
        self.practice_areas = firm_config.get("practice_areas", [])
        self.jurisdictions = firm_config.get("jurisdictions", [])
        self.sol_years = firm_config.get("statute_of_limitations", {})
    
    def score_case(self, intake_data: IntakeData) -> CaseScore:
        """
        Calculate comprehensive case score.
        """
        factors = self._extract_factors(intake_data)
        
        # Calculate component scores
        liability_score = self._score_liability(factors)
        damages_score = self._score_damages(factors, intake_data)
        quality_score = self._score_quality(factors)
        fit_score = self._score_firm_fit(factors, intake_data)
        
        # Base score
        base_score = liability_score + damages_score + quality_score + fit_score
        
        # Apply case type multiplier
        case_type = self._determine_case_type(intake_data)
        multiplier = self.CASE_TYPE_MULTIPLIERS.get(case_type, 1.0)
        
        final_score = min(int(base_score * multiplier), 100)
        
        # Determine priority and recommendation
        priority = self._determine_priority(final_score, factors)
        recommendation = self._generate_recommendation(final_score, factors, intake_data)
        
        return CaseScore(
            score=final_score,
            priority=priority,
            recommendation=recommendation,
            factors=factors,
            component_scores={
                "liability": liability_score,
                "damages": damages_score,
                "quality": quality_score,
                "fit": fit_score,
            },
            case_type=case_type,
        )
    
    def _score_liability(self, factors: CaseScoreFactors) -> int:
        score = 0
        if factors.liability_clear:
            score += 20
        if factors.liability_documented:
            score += 10
        return score
    
    def _score_damages(self, factors: CaseScoreFactors, intake: IntakeData) -> int:
        score = 0
        
        # Injury severity
        max_severity = 0
        for injury in intake.injuries:
            injury_lower = injury.lower()
            for injury_type, severity in self.INJURY_SEVERITY.items():
                if injury_type in injury_lower:
                    max_severity = max(max_severity, severity)
        
        score += max_severity * 2  # 0-20 points
        
        # Treatment factors
        if factors.medical_treatment:
            score += 5
        if factors.ongoing_treatment:
            score += 5
        if factors.lost_wages:
            score += 5
        
        return min(score, 35)
    
    def _score_quality(self, factors: CaseScoreFactors) -> int:
        # SOL check - disqualify if expired
        if not factors.within_sol:
            return -1000  # Will make total negative
        
        score = 0
        if factors.insurance_coverage:
            score += 10
        if factors.no_prior_attorney:
            score += 5
        score += factors.client_cooperation  # 1-5 points
        
        return score
    
    def _score_firm_fit(self, factors: CaseScoreFactors, intake: IntakeData) -> int:
        score = 0
        if factors.in_practice_area:
            score += 10
        if factors.in_jurisdiction:
            score += 5
        return score
    
    def _determine_priority(self, score: int, factors: CaseScoreFactors) -> str:
        """Determine lead priority for routing."""
        if score < 0:  # SOL expired
            return "disqualified"
        if score >= 85:
            return "urgent"  # Immediate attorney callback
        if score >= 70:
            return "high"    # Same-day callback
        if score >= 50:
            return "normal"  # Next business day
        return "low"         # May not be worth pursuing
    
    def _generate_recommendation(
        self,
        score: int,
        factors: CaseScoreFactors,
        intake: IntakeData
    ) -> str:
        """Generate human-readable recommendation."""
        if score < 0:
            return f"DISQUALIFIED: Statute of limitations likely expired. Incident date: {intake.incident_date}"
        
        if score >= 85:
            return f"HIGH VALUE LEAD: Clear liability with documented injuries. Recommend immediate attorney callback."
        
        if score >= 70:
            return f"STRONG LEAD: Good case factors. Schedule same-day consultation."
        
        if score >= 50:
            issues = []
            if not factors.liability_clear:
                issues.append("liability unclear")
            if not factors.medical_treatment:
                issues.append("no documented treatment")
            
            return f"MODERATE LEAD: Potential case but {', '.join(issues)}. Needs attorney evaluation."
        
        return f"LOW PRIORITY: Limited case value indicators. Consider declining or referral."


@dataclass
class CaseScore:
    score: int
    priority: str
    recommendation: str
    factors: CaseScoreFactors
    component_scores: dict
    case_type: CaseType
```

### Real-Time Score Display

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LIVE INTAKE DASHBOARD                                     │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  CALL IN PROGRESS - Maria Garcia                    Duration: 4:32      │
  │  Phone: (555) 123-4567                             AI Agent: Alex       │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  CASE SCORE: ████████████████████░░░░  85/100  [URGENT]                │
  │                                                                         │
  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
  │  │ LIABILITY: 30   │  │ DAMAGES: 25     │  │ QUALITY: 20     │        │
  │  │ ■■■■■■■■■■      │  │ ■■■■■■■■░░      │  │ ■■■■■■■■■■      │        │
  │  │ Clear fault     │  │ ER visit        │  │ Fresh case      │        │
  │  │ Police report   │  │ Whiplash        │  │ Good SOL        │        │
  │  └─────────────────┘  │ Concussion      │  └─────────────────┘        │
  │                       └─────────────────┘                              │
  │                                                                         │
  │  COLLECTED INFORMATION:                                                │
  │  ✓ Name: Maria Garcia                                                  │
  │  ✓ Phone: (555) 123-4567                                              │
  │  ✓ Incident: Auto accident, 01/14/2024, Highway 101                   │
  │  ✓ Injuries: Whiplash, concussion (ER documented)                     │
  │  ✓ Liability: Other driver ran red light                              │
  │  ○ Insurance: Not yet collected                                        │
  │  ○ Witnesses: Not yet asked                                           │
  │                                                                         │
  │  RECOMMENDATION:                                                        │
  │  ┌───────────────────────────────────────────────────────────────────┐ │
  │  │ ⚡ HIGH VALUE LEAD: Clear liability with documented injuries.      │ │
  │  │    Recommend immediate attorney callback.                          │ │
  │  └───────────────────────────────────────────────────────────────────┘ │
  │                                                                         │
  │  [👤 Take Over]  [📞 Transfer Now]  [📅 Schedule Callback]  [📝 Notes] │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Attorney-Client Privilege Protection

### The Privilege Challenge

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                ATTORNEY-CLIENT PRIVILEGE IN AI INTAKE                        │
└─────────────────────────────────────────────────────────────────────────────┘

THE LEGAL QUESTION:
───────────────────
Is a conversation with an AI intake bot privileged?

TRADITIONAL PRIVILEGE REQUIREMENTS:
1. Communication between client and attorney (or agent)
2. Made in confidence
3. For purpose of seeking legal advice
4. Client intends it to be confidential

THE CHALLENGE WITH AI:
──────────────────────
• Is an AI "agent" of the attorney for privilege purposes?
• Are cloud-stored transcripts "confidential"?
• Does third-party (OpenAI, Deepgram) access waive privilege?
• What about when humans review transcripts for QA?

PROTECTIVE MEASURES WE IMPLEMENT:
─────────────────────────────────
1. CLEAR DISCLOSURE
   "This call is being handled by a virtual assistant on behalf of 
    Smith & Associates law firm. This call may be recorded and is 
    confidential attorney-client communication."

2. UNDER ATTORNEY SUPERVISION
   All AI interactions are under supervision of licensed attorney
   AI is acting as agent of the firm

3. CONFIDENTIAL STORAGE
   • End-to-end encryption
   • Per-tenant isolation
   • Access controls
   • Audit logging

4. THIRD-PARTY AGREEMENTS
   • BAAs with all processors
   • Data processing agreements
   • No training on client data
   • Regional data residency

5. MINIMIZE EXPOSURE
   • Stream processing (don't store raw audio long-term)
   • PII redaction for analytics
   • Separate privileged vs non-privileged data
```

### Privilege-Preserving Architecture

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import hashlib

class PrivilegeStatus(Enum):
    PRIVILEGED = "privileged"              # Full attorney-client privilege
    POTENTIALLY_PRIVILEGED = "potentially" # Needs attorney review
    NOT_PRIVILEGED = "not_privileged"      # Public info, no privilege claim
    WAIVED = "waived"                      # Privilege was waived

@dataclass
class PrivilegedDocument:
    """
    Wrapper for privileged content with metadata.
    """
    content_hash: str           # Hash of content (for integrity)
    privilege_status: PrivilegeStatus
    privilege_holder: str       # Client who holds privilege
    supervising_attorney: str   # Attorney responsible
    created_at: str
    access_log: list           # Who accessed this document
    
    # Content stored separately with encryption
    encrypted_content_ref: str  # Reference to encrypted storage


class PrivilegeProtectedStorage:
    """
    Storage layer that enforces privilege protections.
    """
    
    def __init__(
        self,
        encryption_service,
        audit_logger,
        tenant_context,
    ):
        self.encryption = encryption_service
        self.audit = audit_logger
        self.tenant = tenant_context
    
    async def store_transcript(
        self,
        transcript: str,
        call_metadata: dict,
        privilege_status: PrivilegeStatus = PrivilegeStatus.PRIVILEGED,
    ) -> str:
        """
        Store call transcript with privilege protections.
        """
        # 1. Encrypt with tenant-specific key
        encrypted = await self.encryption.encrypt(
            plaintext=transcript.encode(),
            context={
                "tenant_id": self.tenant.tenant_id,
                "document_type": "call_transcript",
                "privilege_status": privilege_status.value,
            }
        )
        
        # 2. Store encrypted content
        content_ref = await self._store_encrypted(encrypted)
        
        # 3. Create privilege metadata record
        doc = PrivilegedDocument(
            content_hash=hashlib.sha256(transcript.encode()).hexdigest(),
            privilege_status=privilege_status,
            privilege_holder=call_metadata.get("caller_name", "Unknown"),
            supervising_attorney=self.tenant.supervising_attorney,
            created_at=datetime.utcnow().isoformat(),
            access_log=[],
            encrypted_content_ref=content_ref,
        )
        
        # 4. Store metadata (searchable, but not the content)
        doc_id = await self._store_metadata(doc)
        
        # 5. Audit log
        await self.audit.log(
            action="store_privileged_transcript",
            resource_type="transcript",
            resource_id=doc_id,
            metadata={
                "privilege_status": privilege_status.value,
                "content_hash": doc.content_hash,
            }
        )
        
        return doc_id
    
    async def retrieve_transcript(
        self,
        doc_id: str,
        accessor_id: str,
        access_reason: str,
    ) -> Optional[str]:
        """
        Retrieve transcript with access controls and logging.
        """
        # 1. Get metadata
        doc = await self._get_metadata(doc_id)
        if not doc:
            return None
        
        # 2. Check access authorization
        if not await self._check_access(accessor_id, doc):
            await self.audit.log(
                action="access_denied_privileged",
                resource_id=doc_id,
                user_id=accessor_id,
            )
            raise PrivilegeAccessDenied(
                f"User {accessor_id} not authorized to access privileged document"
            )
        
        # 3. Log access
        await self.audit.log(
            action="access_privileged_transcript",
            resource_type="transcript",
            resource_id=doc_id,
            user_id=accessor_id,
            metadata={
                "access_reason": access_reason,
                "privilege_status": doc.privilege_status.value,
            }
        )
        
        # 4. Retrieve and decrypt
        encrypted = await self._get_encrypted(doc.encrypted_content_ref)
        plaintext = await self.encryption.decrypt(encrypted)
        
        return plaintext.decode()
    
    async def _check_access(self, accessor_id: str, doc: PrivilegedDocument) -> bool:
        """
        Check if accessor is authorized to view privileged content.
        
        Authorized users:
        - Supervising attorney
        - Attorneys at the firm
        - Paralegals assigned to the case
        - The client themselves
        """
        user = await self._get_user(accessor_id)
        
        # Supervising attorney always has access
        if accessor_id == doc.supervising_attorney:
            return True
        
        # Licensed attorneys at the firm
        if user.role == "attorney" and user.tenant_id == self.tenant.tenant_id:
            return True
        
        # Paralegals with case assignment
        if user.role == "paralegal":
            case = await self._get_case_for_transcript(doc)
            if case and accessor_id in case.assigned_staff:
                return True
        
        return False


class ThirdPartyDataProtection:
    """
    Protect data sent to third-party services.
    """
    
    def __init__(self):
        # Services that process audio/text
        self.processors = {
            "deepgram": {
                "data_retention": "none",  # Streaming, no retention
                "baa_signed": True,
                "region": "us",
            },
            "elevenlabs": {
                "data_retention": "none",
                "baa_signed": True,
                "region": "us",
            },
            "openai": {
                "data_retention": "30_days",  # API data retention
                "baa_signed": True,
                "region": "us",
                "opt_out_training": True,  # Opted out of training
            },
        }
    
    def verify_processor_compliance(self, processor: str) -> bool:
        """Verify processor meets our requirements."""
        config = self.processors.get(processor)
        if not config:
            return False
        
        # Must have BAA for HIPAA compliance
        if not config.get("baa_signed"):
            return False
        
        # Must have opted out of training
        if config.get("opt_out_training") is False:
            return False
        
        return True
    
    def redact_for_analytics(self, transcript: str) -> str:
        """
        Redact PII for non-privileged analytics use.
        
        Used for:
        - Aggregate quality metrics
        - Training internal models
        - Performance analysis
        """
        import re
        
        # Phone numbers
        transcript = re.sub(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            '[PHONE]',
            transcript
        )
        
        # Email addresses
        transcript = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            transcript
        )
        
        # SSN
        transcript = re.sub(
            r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            '[SSN]',
            transcript
        )
        
        # Names (using NER would be better in production)
        # This is simplified - would use spaCy or similar
        
        return transcript
```

### Disclosure Script

```python
PRIVILEGE_DISCLOSURE = """
Thank you for calling {firm_name}. 

Before we begin, I want to let you know that I am a virtual legal assistant 
working under the supervision of our attorneys. 

This call may be recorded for quality and training purposes. All information 
you share will be kept confidential as part of an attorney-client relationship 
with our firm.

If at any point you would prefer to speak with a human, just let me know and 
I'll connect you right away.

How can I help you today?
"""

class IntakeGreeting:
    """
    Handle intake greeting with proper disclosures.
    """
    
    async def get_greeting(self, tenant_config: dict, caller_info: dict = None) -> str:
        """
        Generate appropriate greeting with disclosures.
        """
        firm_name = tenant_config.get("firm_name", "our law firm")
        
        # Check if this is a return caller
        if caller_info and caller_info.get("is_existing_client"):
            return self._existing_client_greeting(firm_name, caller_info)
        
        # New caller - full disclosure
        return PRIVILEGE_DISCLOSURE.format(firm_name=firm_name)
    
    def _existing_client_greeting(self, firm_name: str, caller_info: dict) -> str:
        return f"""
Welcome back to {firm_name}. I see you've called us before.

Just as a reminder, I'm a virtual assistant and this call may be recorded.

How can I help you today, {caller_info.get('name', 'there')}?
"""
```

---

## 7. CMS Integration

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CMS INTEGRATION LAYER                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                         Intake Completed
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INTEGRATION ROUTER                                      │
│                                                                              │
│  Tenant Config:                                                             │
│  • cms_type: "clio"                                                         │
│  • api_credentials: encrypted                                               │
│  • field_mappings: {...}                                                    │
│  • auto_create_matter: true                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │    Clio     │     │  Filevine   │     │   Litify    │
    │   Adapter   │     │   Adapter   │     │   Adapter   │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Clio API   │     │ Filevine API│     │ Salesforce  │
    │  (REST)     │     │  (REST)     │     │  (REST)     │
    └─────────────┘     └─────────────┘     └─────────────┘

Created Records:
• Contact/Lead
• Matter/Case (optional)
• Activity/Note (call summary)
• Document (transcript)
• Task (follow-up)
```

### CMS Adapter Implementation

```python
from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class CMSContact:
    first_name: str
    last_name: str
    phone: str
    email: Optional[str] = None
    address: Optional[str] = None
    source: str = "ai_intake"
    
@dataclass
class CMSMatter:
    contact_id: str
    matter_type: str  # "personal_injury", etc.
    description: str
    status: str = "intake"
    assigned_attorney: Optional[str] = None
    
@dataclass
class CMSActivity:
    matter_id: str
    activity_type: str  # "phone_call", "intake"
    subject: str
    description: str
    duration_minutes: int
    

class CMSAdapter(ABC):
    """
    Abstract base for CMS integrations.
    """
    
    @abstractmethod
    async def create_contact(self, contact: CMSContact) -> str:
        """Create or update contact, return ID."""
        pass
    
    @abstractmethod
    async def create_matter(self, matter: CMSMatter) -> str:
        """Create matter/case, return ID."""
        pass
    
    @abstractmethod
    async def add_activity(self, activity: CMSActivity) -> str:
        """Add activity/note to matter."""
        pass
    
    @abstractmethod
    async def upload_document(self, matter_id: str, document: bytes, filename: str) -> str:
        """Upload document to matter."""
        pass


class ClioAdapter(CMSAdapter):
    """
    Clio CMS integration.
    https://app.clio.com/api/v4/documentation
    """
    
    def __init__(self, api_token: str, base_url: str = "https://app.clio.com/api/v4"):
        self.api_token = api_token
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
    
    async def create_contact(self, contact: CMSContact) -> str:
        """Create contact in Clio."""
        # Check if contact exists
        existing = await self._find_contact_by_phone(contact.phone)
        if existing:
            return existing["id"]
        
        payload = {
            "data": {
                "first_name": contact.first_name,
                "last_name": contact.last_name,
                "type": "Person",
                "phone_numbers": [
                    {"name": "Mobile", "number": contact.phone, "default_number": True}
                ],
            }
        }
        
        if contact.email:
            payload["data"]["email_addresses"] = [
                {"name": "Work", "address": contact.email, "default_email": True}
            ]
        
        response = await self._post("/contacts.json", payload)
        return response["data"]["id"]
    
    async def create_matter(self, matter: CMSMatter) -> str:
        """Create matter in Clio."""
        payload = {
            "data": {
                "client": {"id": matter.contact_id},
                "description": matter.description,
                "status": "Open",
                "practice_area": {"name": self._map_practice_area(matter.matter_type)},
            }
        }
        
        if matter.assigned_attorney:
            payload["data"]["responsible_attorney"] = {"id": matter.assigned_attorney}
        
        response = await self._post("/matters.json", payload)
        return response["data"]["id"]
    
    async def add_activity(self, activity: CMSActivity) -> str:
        """Add activity to matter in Clio."""
        payload = {
            "data": {
                "matter": {"id": activity.matter_id},
                "type": "PhoneCall" if activity.activity_type == "phone_call" else "Note",
                "subject": activity.subject,
                "detail": activity.description,
                "date": datetime.utcnow().isoformat(),
                "quantity": activity.duration_minutes * 60,  # Clio uses seconds
            }
        }
        
        response = await self._post("/activities.json", payload)
        return response["data"]["id"]
    
    async def upload_document(self, matter_id: str, document: bytes, filename: str) -> str:
        """Upload document to matter."""
        # First create document record
        doc_payload = {
            "data": {
                "name": filename,
                "parent": {"id": matter_id, "type": "Matter"},
            }
        }
        
        doc_response = await self._post("/documents.json", doc_payload)
        doc_id = doc_response["data"]["id"]
        
        # Then upload content
        await self._upload_file(f"/documents/{doc_id}/upload", document, filename)
        
        return doc_id
    
    def _map_practice_area(self, matter_type: str) -> str:
        """Map our matter types to Clio practice areas."""
        mapping = {
            "auto_accident": "Personal Injury",
            "slip_fall": "Personal Injury",
            "medical_malpractice": "Medical Malpractice",
            "workers_comp": "Workers' Compensation",
        }
        return mapping.get(matter_type, "Personal Injury")


class FilevineAdapter(CMSAdapter):
    """
    Filevine CMS integration.
    """
    
    def __init__(self, org_id: str, api_key: str):
        self.org_id = org_id
        self.api_key = api_key
        self.base_url = f"https://app.filevine.com/api/v2/org/{org_id}"
    
    async def create_contact(self, contact: CMSContact) -> str:
        payload = {
            "fullName": f"{contact.first_name} {contact.last_name}",
            "personTypes": ["Client"],
            "phones": [{"number": contact.phone, "type": "Mobile"}],
        }
        
        if contact.email:
            payload["emails"] = [{"address": contact.email, "type": "Personal"}]
        
        response = await self._post("/contacts", payload)
        return response["contactId"]
    
    async def create_matter(self, matter: CMSMatter) -> str:
        # Filevine calls them "Projects"
        payload = {
            "projectTypeId": self._get_project_type(matter.matter_type),
            "projectName": matter.description[:100],
            "contacts": [
                {"contactId": matter.contact_id, "role": "Client"}
            ],
        }
        
        response = await self._post("/projects", payload)
        return response["projectId"]
    
    # ... similar implementations for other methods


class CMSIntegrationManager:
    """
    Manage CMS integrations across tenants.
    """
    
    ADAPTERS = {
        "clio": ClioAdapter,
        "filevine": FilevineAdapter,
        "litify": LitifyAdapter,
        "mycase": MyCaseAdapter,
        "practicepanther": PracticePantherAdapter,
    }
    
    def __init__(self, tenant_config: dict):
        self.tenant_config = tenant_config
        self.cms_type = tenant_config.get("cms_type")
        self.adapter = self._create_adapter()
    
    def _create_adapter(self) -> Optional[CMSAdapter]:
        if not self.cms_type:
            return None
        
        adapter_class = self.ADAPTERS.get(self.cms_type)
        if not adapter_class:
            raise ValueError(f"Unknown CMS type: {self.cms_type}")
        
        credentials = self._decrypt_credentials()
        return adapter_class(**credentials)
    
    async def sync_intake(self, intake_data: IntakeData, call_metadata: dict) -> dict:
        """
        Sync completed intake to CMS.
        
        Returns IDs of created records.
        """
        if not self.adapter:
            return {"error": "No CMS configured"}
        
        result = {}
        
        try:
            # 1. Create/update contact
            name_parts = (intake_data.caller_name or "Unknown").split(" ", 1)
            contact = CMSContact(
                first_name=name_parts[0],
                last_name=name_parts[1] if len(name_parts) > 1 else "",
                phone=intake_data.phone_number,
                email=intake_data.email,
                source="ai_intake",
            )
            
            contact_id = await self.adapter.create_contact(contact)
            result["contact_id"] = contact_id
            
            # 2. Create matter if configured
            if self.tenant_config.get("auto_create_matter", False):
                matter = CMSMatter(
                    contact_id=contact_id,
                    matter_type=intake_data.case_type or "personal_injury",
                    description=self._generate_matter_description(intake_data),
                )
                
                matter_id = await self.adapter.create_matter(matter)
                result["matter_id"] = matter_id
            else:
                matter_id = None
            
            # 3. Add call activity
            if matter_id:
                activity = CMSActivity(
                    matter_id=matter_id,
                    activity_type="phone_call",
                    subject="AI Intake Call",
                    description=self._generate_activity_summary(intake_data, call_metadata),
                    duration_minutes=call_metadata.get("duration_minutes", 0),
                )
                
                activity_id = await self.adapter.add_activity(activity)
                result["activity_id"] = activity_id
            
            # 4. Upload transcript if configured
            if matter_id and self.tenant_config.get("upload_transcript", True):
                transcript = call_metadata.get("transcript", "")
                if transcript:
                    doc_id = await self.adapter.upload_document(
                        matter_id=matter_id,
                        document=transcript.encode(),
                        filename=f"intake_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    )
                    result["transcript_doc_id"] = doc_id
            
            return result
            
        except Exception as e:
            # Log error but don't fail the intake
            await self._log_sync_error(e, intake_data)
            return {"error": str(e)}
    
    def _generate_matter_description(self, intake: IntakeData) -> str:
        """Generate matter description from intake data."""
        parts = []
        
        if intake.case_type:
            parts.append(f"Case Type: {intake.case_type.replace('_', ' ').title()}")
        
        if intake.incident_date:
            parts.append(f"Incident Date: {intake.incident_date}")
        
        if intake.incident_location:
            parts.append(f"Location: {intake.incident_location}")
        
        if intake.injuries:
            parts.append(f"Injuries: {', '.join(intake.injuries)}")
        
        return "\n".join(parts)
```

---

## 8. Human Handoff System

### Handoff Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HUMAN HANDOFF SYSTEM                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                    Handoff Triggers
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │   Caller    │  │   AI Can't  │  │   High      │
  │   Requests  │  │   Handle    │  │   Score     │
  │   Human     │  │   Query     │  │   Case      │
  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
         │                │                │
         └────────────────┼────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                      HANDOFF ORCHESTRATOR                                │
  │                                                                          │
  │  1. Determine handoff type (immediate, scheduled, queue)                │
  │  2. Find available agent/attorney                                       │
  │  3. Prepare context summary                                             │
  │  4. Execute handoff                                                     │
  └─────────────────────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │   WARM      │  │   QUEUE     │  │  SCHEDULED  │
  │  TRANSFER   │  │             │  │  CALLBACK   │
  │             │  │             │  │             │
  │ • Keep      │  │ • Hold with │  │ • Book slot │
  │   caller on │  │   music     │  │ • Confirm   │
  │ • Conference│  │ • Estimate  │  │   with SMS  │
  │   in agent  │  │   wait time │  │ • Reminder  │
  │ • Context   │  │ • Periodic  │  │   call      │
  │   whisper   │  │   updates   │  │             │
  └─────────────┘  └─────────────┘  └─────────────┘
```

### Handoff Implementation

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
import asyncio

class HandoffType(Enum):
    IMMEDIATE = "immediate"       # Warm transfer now
    QUEUE = "queue"              # Hold for next available
    SCHEDULED = "scheduled"      # Schedule callback
    VOICEMAIL = "voicemail"      # Leave message
    URGENT = "urgent"            # Page on-call attorney

class HandoffReason(Enum):
    CALLER_REQUEST = "caller_request"
    HIGH_VALUE_CASE = "high_value_case"
    COMPLEX_QUESTION = "complex_question"
    EXISTING_CLIENT = "existing_client"
    EMOTIONAL_DISTRESS = "emotional_distress"
    LANGUAGE_BARRIER = "language_barrier"
    AI_CONFUSION = "ai_confusion"
    QUALITY_CHECK = "quality_check"

@dataclass
class HandoffContext:
    """Context passed to human agent during handoff."""
    caller_name: str
    phone_number: str
    call_duration: int
    
    # Intake summary
    case_type: Optional[str]
    incident_summary: str
    injuries: List[str]
    case_score: int
    priority: str
    
    # Handoff details
    handoff_reason: HandoffReason
    ai_confidence: float
    
    # Conversation context
    key_points: List[str]
    questions_asked: List[str]
    caller_concerns: List[str]
    
    # Full transcript (for agent review)
    transcript: str


class HandoffOrchestrator:
    """
    Manage handoffs from AI to human agents.
    """
    
    def __init__(
        self,
        telephony_client,
        agent_queue: 'AgentQueue',
        scheduler: 'CallbackScheduler',
        notification_service,
    ):
        self.telephony = telephony_client
        self.agent_queue = agent_queue
        self.scheduler = scheduler
        self.notifications = notification_service
    
    async def initiate_handoff(
        self,
        call_id: str,
        reason: HandoffReason,
        intake_data: IntakeData,
        conversation_history: List[dict],
    ) -> HandoffResult:
        """
        Initiate handoff to human.
        """
        # 1. Prepare context
        context = self._prepare_context(intake_data, conversation_history, reason)
        
        # 2. Determine handoff type based on availability and priority
        handoff_type = await self._determine_handoff_type(context)
        
        # 3. Execute handoff
        if handoff_type == HandoffType.IMMEDIATE:
            return await self._warm_transfer(call_id, context)
        
        elif handoff_type == HandoffType.QUEUE:
            return await self._queue_transfer(call_id, context)
        
        elif handoff_type == HandoffType.SCHEDULED:
            return await self._schedule_callback(call_id, context)
        
        elif handoff_type == HandoffType.URGENT:
            return await self._urgent_escalation(call_id, context)
        
        else:
            return await self._voicemail_handoff(call_id, context)
    
    async def _determine_handoff_type(self, context: HandoffContext) -> HandoffType:
        """
        Determine best handoff type based on context and availability.
        """
        # Check agent availability
        available_agents = await self.agent_queue.get_available_agents(
            skill=context.case_type
        )
        
        # Urgent cases: page on-call if no one available
        if context.priority == "urgent" and not available_agents:
            return HandoffType.URGENT
        
        # High value with available agent: immediate transfer
        if context.case_score >= 80 and available_agents:
            return HandoffType.IMMEDIATE
        
        # Available agent with short wait: queue
        if available_agents:
            wait_time = await self.agent_queue.estimate_wait(context.case_type)
            if wait_time < 120:  # Less than 2 minutes
                return HandoffType.QUEUE
        
        # Check business hours for callback
        if self._is_business_hours():
            return HandoffType.SCHEDULED
        
        # After hours: voicemail
        return HandoffType.VOICEMAIL
    
    async def _warm_transfer(
        self,
        call_id: str,
        context: HandoffContext,
    ) -> HandoffResult:
        """
        Execute warm transfer to available agent.
        
        Warm transfer: AI stays on briefly to introduce,
        then drops off.
        """
        # 1. Find best available agent
        agent = await self.agent_queue.get_best_agent(
            skill=context.case_type,
            priority=context.priority,
        )
        
        # 2. Notify agent (screen pop)
        await self.notifications.send_screen_pop(
            agent_id=agent.id,
            context=context,
        )
        
        # 3. Speak handoff message to caller
        handoff_message = self._generate_handoff_message(agent, "immediate")
        await self.telephony.speak(call_id, handoff_message)
        
        # 4. Conference in agent
        await self.telephony.conference(
            call_id=call_id,
            target=agent.phone,
            whisper=self._generate_whisper(context),  # Only agent hears
        )
        
        # 5. Wait for agent to accept
        agent_accepted = await self._wait_for_agent_accept(agent.id, timeout=30)
        
        if agent_accepted:
            # 6. Drop AI from call
            await asyncio.sleep(2)  # Brief overlap for smooth transition
            await self.telephony.drop_from_conference(call_id, "ai")
            
            return HandoffResult(
                success=True,
                handoff_type=HandoffType.IMMEDIATE,
                agent_id=agent.id,
            )
        else:
            # Agent didn't accept, fall back to queue
            return await self._queue_transfer(call_id, context)
    
    async def _queue_transfer(
        self,
        call_id: str,
        context: HandoffContext,
    ) -> HandoffResult:
        """
        Put caller in queue for next available agent.
        """
        # 1. Estimate wait time
        wait_time = await self.agent_queue.estimate_wait(context.case_type)
        
        # 2. Offer queue or callback
        offer_message = f"""
        I'd like to connect you with one of our attorneys. 
        The estimated wait time is about {wait_time // 60} minutes.
        
        Would you like to hold, or should I have someone call you back?
        """
        
        await self.telephony.speak(call_id, offer_message)
        
        # 3. Wait for response (simplified - would use STT)
        response = await self._get_caller_response(call_id)
        
        if "call back" in response.lower() or "callback" in response.lower():
            return await self._schedule_callback(call_id, context)
        
        # 4. Add to queue
        queue_position = await self.agent_queue.enqueue(
            call_id=call_id,
            context=context,
        )
        
        # 5. Play hold message and music
        await self.telephony.play_hold_music(
            call_id,
            initial_message=f"You are number {queue_position} in queue. Please hold.",
            periodic_updates=True,
        )
        
        return HandoffResult(
            success=True,
            handoff_type=HandoffType.QUEUE,
            queue_position=queue_position,
            estimated_wait=wait_time,
        )
    
    async def _schedule_callback(
        self,
        call_id: str,
        context: HandoffContext,
    ) -> HandoffResult:
        """
        Schedule a callback from an attorney.
        """
        # 1. Get available slots
        slots = await self.scheduler.get_available_slots(
            case_type=context.case_type,
            priority=context.priority,
        )
        
        # 2. Offer slots to caller
        slot_message = self._format_slot_options(slots[:3])
        await self.telephony.speak(call_id, slot_message)
        
        # 3. Get caller's preference (simplified)
        selected_slot = await self._get_slot_selection(call_id, slots)
        
        # 4. Book the callback
        booking = await self.scheduler.book_callback(
            slot=selected_slot,
            caller_phone=context.phone_number,
            caller_name=context.caller_name,
            context=context,
        )
        
        # 5. Confirm with caller
        confirmation = f"""
        I've scheduled a callback for {selected_slot.formatted_time}.
        
        You'll receive a text message with the confirmation details.
        
        Is there anything else I can help you with before we end the call?
        """
        
        await self.telephony.speak(call_id, confirmation)
        
        # 6. Send SMS confirmation
        await self.notifications.send_sms(
            to=context.phone_number,
            message=f"Callback confirmed with {self.firm_name} for {selected_slot.formatted_time}. "
                    f"Call from: {self.firm_phone}"
        )
        
        return HandoffResult(
            success=True,
            handoff_type=HandoffType.SCHEDULED,
            scheduled_time=selected_slot.datetime,
            booking_id=booking.id,
        )
    
    def _generate_whisper(self, context: HandoffContext) -> str:
        """
        Generate whisper message for agent (caller doesn't hear).
        """
        return f"""
        Incoming transfer from AI intake.
        Caller: {context.caller_name}
        Case type: {context.case_type}
        Score: {context.case_score} out of 100
        Priority: {context.priority}
        
        Key points:
        {chr(10).join('• ' + p for p in context.key_points[:3])}
        
        Press 1 to accept.
        """


class AgentQueue:
    """
    Manage queue of callers waiting for agents.
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def get_available_agents(self, skill: str = None) -> List[Agent]:
        """Get list of available agents, optionally filtered by skill."""
        agents = await self._get_all_agents()
        
        available = [
            a for a in agents 
            if a.status == "available" 
            and (skill is None or skill in a.skills)
        ]
        
        return available
    
    async def get_best_agent(self, skill: str, priority: str) -> Agent:
        """
        Get best available agent for this call.
        
        Considers:
        - Skill match
        - Current workload
        - Performance metrics
        - Round-robin fairness
        """
        available = await self.get_available_agents(skill)
        
        if not available:
            raise NoAgentAvailable()
        
        # Score agents
        scored = []
        for agent in available:
            score = 0
            
            # Skill match bonus
            if skill in agent.primary_skills:
                score += 10
            elif skill in agent.skills:
                score += 5
            
            # Lower workload = higher score
            score += (10 - agent.current_calls) * 2
            
            # Higher performance = higher score
            score += agent.performance_score / 10
            
            # Fairness: penalize recently assigned
            minutes_since_last = (datetime.now() - agent.last_assigned).seconds / 60
            score += min(minutes_since_last, 10)
            
            scored.append((agent, score))
        
        # Return highest scored agent
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
    
    async def estimate_wait(self, skill: str = None) -> int:
        """Estimate wait time in seconds."""
        queue_length = await self._get_queue_length(skill)
        avg_handle_time = await self._get_avg_handle_time(skill)
        available_agents = len(await self.get_available_agents(skill))
        
        if available_agents == 0:
            return 600  # 10 minute max estimate
        
        return int((queue_length * avg_handle_time) / available_agents)
```

---

## 9. Multi-Language & Accent Handling

### Language Support Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-LANGUAGE SUPPORT                                    │
└─────────────────────────────────────────────────────────────────────────────┘

        Incoming Call
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LANGUAGE DETECTION                                        │
│                                                                              │
│  Option 1: IVR Prompt                                                       │
│  "For English, press 1. Para español, oprima el 2."                        │
│                                                                              │
│  Option 2: Automatic Detection                                              │
│  • Analyze first few seconds of speech                                      │
│  • Detect language with high confidence                                     │
│  • Confirm with caller: "I detected Spanish. Is that correct?"             │
│                                                                              │
│  Option 3: Phone Number Routing                                             │
│  • Different numbers for different languages                                │
│  • 1-800-XXX-XXXX (English)                                                │
│  • 1-800-XXX-YYYY (Spanish)                                                │
└─────────────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LANGUAGE-SPECIFIC PIPELINE                                │
│                                                                              │
│  ┌─────────────────┐                    ┌─────────────────┐                │
│  │  English STT    │                    │   Spanish STT   │                │
│  │  • Deepgram     │                    │  • Deepgram     │                │
│  │  • Model: nova-2│                    │  • Model: nova-2│                │
│  │  • Lang: en-US  │                    │  • Lang: es-US  │                │
│  └────────┬────────┘                    └────────┬────────┘                │
│           │                                      │                          │
│           ▼                                      ▼                          │
│  ┌─────────────────┐                    ┌─────────────────┐                │
│  │  English LLM    │                    │  Spanish LLM    │                │
│  │  • System prompt│                    │  • System prompt│                │
│  │    in English   │                    │    in Spanish   │                │
│  │  • Responses in │                    │  • Responses in │                │
│  │    English      │                    │    Spanish      │                │
│  └────────┬────────┘                    └────────┬────────┘                │
│           │                                      │                          │
│           ▼                                      ▼                          │
│  ┌─────────────────┐                    ┌─────────────────┐                │
│  │  English TTS    │                    │   Spanish TTS   │                │
│  │  • Voice: Rachel│                    │  • Voice: Sofia │                │
│  │  • Accent: US   │                    │  • Accent: MX   │                │
│  └─────────────────┘                    └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Language Detection & Handling

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class SupportedLanguage(Enum):
    ENGLISH = "en"
    SPANISH = "es"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    VIETNAMESE = "vi"
    KOREAN = "ko"

@dataclass
class LanguageConfig:
    code: str
    stt_model: str
    stt_language: str
    tts_voice_id: str
    tts_language: str
    system_prompt_template: str
    greeting: str

LANGUAGE_CONFIGS = {
    SupportedLanguage.ENGLISH: LanguageConfig(
        code="en",
        stt_model="nova-2",
        stt_language="en-US",
        tts_voice_id="EXAVITQu4vr4xnSDxMaL",  # Rachel
        tts_language="en-US",
        system_prompt_template="prompts/intake_english.txt",
        greeting="Thank you for calling {firm_name}. I'm Alex, a virtual assistant. How can I help you today?",
    ),
    SupportedLanguage.SPANISH: LanguageConfig(
        code="es",
        stt_model="nova-2",
        stt_language="es-US",  # US Spanish for legal terms
        tts_voice_id="XrExE9yKIg1WjnnlVkGX",  # Sofia
        tts_language="es-MX",
        system_prompt_template="prompts/intake_spanish.txt",
        greeting="Gracias por llamar a {firm_name}. Soy Alex, un asistente virtual. ¿En qué puedo ayudarle hoy?",
    ),
}

class LanguageDetector:
    """
    Detect caller's language from speech.
    """
    
    def __init__(self, deepgram_client):
        self.deepgram = deepgram_client
    
    async def detect_language(
        self, 
        audio_sample: bytes,
        duration_seconds: float = 3.0,
    ) -> tuple[SupportedLanguage, float]:
        """
        Detect language from audio sample.
        
        Returns: (detected_language, confidence)
        """
        # Use Deepgram's language detection
        response = await self.deepgram.transcription.prerecorded(
            audio_sample,
            {
                "detect_language": True,
                "model": "nova-2",
            }
        )
        
        detected = response["results"]["channels"][0]["detected_language"]
        confidence = response["results"]["channels"][0]["language_confidence"]
        
        # Map to supported language
        language_map = {
            "en": SupportedLanguage.ENGLISH,
            "es": SupportedLanguage.SPANISH,
            "pt": SupportedLanguage.PORTUGUESE,
            "zh": SupportedLanguage.CHINESE,
        }
        
        language = language_map.get(detected, SupportedLanguage.ENGLISH)
        
        return language, confidence
    
    async def detect_from_ivr(self, dtmf_digit: str) -> SupportedLanguage:
        """
        Get language from IVR selection.
        """
        ivr_map = {
            "1": SupportedLanguage.ENGLISH,
            "2": SupportedLanguage.SPANISH,
        }
        
        return ivr_map.get(dtmf_digit, SupportedLanguage.ENGLISH)


class AccentAdapter:
    """
    Handle various accents and dialects within a language.
    """
    
    # Regional accent configurations for STT
    ACCENT_MODELS = {
        "en-US": {"model": "nova-2", "keywords": []},
        "en-GB": {"model": "nova-2", "keywords": []},
        "en-AU": {"model": "nova-2", "keywords": []},
        "en-IN": {"model": "nova-2", "keywords": []},  # Indian English
        
        "es-US": {"model": "nova-2", "keywords": []},   # US Spanish
        "es-MX": {"model": "nova-2", "keywords": []},   # Mexican Spanish
        "es-ES": {"model": "nova-2", "keywords": []},   # Spain Spanish
    }
    
    # Legal terminology by language (improves STT accuracy)
    LEGAL_KEYWORDS = {
        "en": [
            "personal injury", "negligence", "liability", "damages",
            "deposition", "plaintiff", "defendant", "settlement",
            "statute of limitations", "contingency", "retainer",
        ],
        "es": [
            "lesiones personales", "negligencia", "responsabilidad",
            "daños y perjuicios", "demandante", "demandado",
            "acuerdo", "abogado", "compensación",
        ],
    }
    
    def get_stt_config(self, language: SupportedLanguage, accent: str = None) -> dict:
        """
        Get optimized STT configuration for language/accent.
        """
        base_config = LANGUAGE_CONFIGS[language]
        
        config = {
            "model": base_config.stt_model,
            "language": accent or base_config.stt_language,
            "smart_format": True,
            "punctuate": True,
            "diarize": True,
            "keywords": self.LEGAL_KEYWORDS.get(base_config.code, []),
        }
        
        return config


class MultilingualIntake:
    """
    Handle intake in multiple languages.
    """
    
    def __init__(
        self,
        language_detector: LanguageDetector,
        stt_factory,
        tts_factory,
        llm_client,
    ):
        self.detector = language_detector
        self.stt_factory = stt_factory
        self.tts_factory = tts_factory
        self.llm = llm_client
        
        self.current_language: Optional[SupportedLanguage] = None
        self.language_config: Optional[LanguageConfig] = None
    
    async def initialize_language(self, method: str = "auto", audio: bytes = None) -> SupportedLanguage:
        """
        Initialize language for the call.
        """
        if method == "auto" and audio:
            language, confidence = await self.detector.detect_language(audio)
            
            if confidence < 0.8:
                # Low confidence - ask caller
                language = await self._ask_language_preference()
            
        elif method == "ivr":
            # Play IVR and get selection
            language = await self._play_language_ivr()
        
        else:
            language = SupportedLanguage.ENGLISH
        
        # Set up language-specific components
        self.current_language = language
        self.language_config = LANGUAGE_CONFIGS[language]
        
        self.stt = self.stt_factory.create(self.language_config)
        self.tts = self.tts_factory.create(self.language_config)
        
        return language
    
    async def _play_language_ivr(self) -> SupportedLanguage:
        """
        Play language selection IVR.
        """
        # Play in both languages
        ivr_prompt = """
        For English, press 1.
        Para español, oprima el número 2.
        """
        
        # Synthesize and play
        await self.telephony.play_and_collect_digit(ivr_prompt, valid_digits="12")
        
        digit = await self.telephony.get_digit(timeout=10)
        
        return await self.detector.detect_from_ivr(digit)
    
    async def get_localized_prompt(self, prompt_key: str, **kwargs) -> str:
        """
        Get prompt in current language.
        """
        prompts = {
            SupportedLanguage.ENGLISH: {
                "greeting": "Thank you for calling {firm_name}. How can I help you?",
                "ask_name": "May I have your name please?",
                "ask_phone": "And what's the best phone number to reach you?",
                "ask_incident": "Can you tell me what happened?",
                "empathy": "I'm sorry to hear that. That must be difficult.",
                "transfer": "I'm going to connect you with one of our attorneys.",
            },
            SupportedLanguage.SPANISH: {
                "greeting": "Gracias por llamar a {firm_name}. ¿En qué puedo ayudarle?",
                "ask_name": "¿Me puede dar su nombre por favor?",
                "ask_phone": "¿Y cuál es el mejor número de teléfono para contactarle?",
                "ask_incident": "¿Me puede contar qué pasó?",
                "empathy": "Lamento mucho escuchar eso. Debe ser muy difícil.",
                "transfer": "Voy a conectarle con uno de nuestros abogados.",
            },
        }
        
        prompt_template = prompts[self.current_language].get(prompt_key, "")
        return prompt_template.format(**kwargs)
```

---

## 10. Scale & Performance

### Performance Targets

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| End-to-end latency | < 800ms | > 1500ms |
| STT latency | < 200ms | > 400ms |
| LLM response start | < 400ms | > 800ms |
| TTS first byte | < 200ms | > 400ms |
| Concurrent calls | 1000+ | N/A |
| Call success rate | > 99.5% | < 98% |
| Uptime | 99.99% | < 99.9% |

### Scaling Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HORIZONTAL SCALING ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────┘

                         Load Balancer (AWS ALB)
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VOICE GATEWAY CLUSTER                                   │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Gateway 1  │  │  Gateway 2  │  │  Gateway 3  │  │  Gateway N  │        │
│  │  (50 calls) │  │  (50 calls) │  │  (50 calls) │  │  (50 calls) │        │
│  │             │  │             │  │             │  │             │        │
│  │ • WebSocket │  │ • WebSocket │  │ • WebSocket │  │ • WebSocket │        │
│  │ • Media     │  │ • Media     │  │ • Media     │  │ • Media     │        │
│  │   handling  │  │   handling  │  │   handling  │  │   handling  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                              │
│  Auto-scaling: Add gateway when avg calls > 40 per instance                 │
│  Scale down: Remove gateway when avg calls < 20                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AI PROCESSING CLUSTER                                   │
│                                                                              │
│  Connection pooling to external APIs:                                        │
│  • Deepgram: 100 concurrent streams per pod                                 │
│  • OpenAI: 500 RPM per pod                                                  │
│  • ElevenLabs: 100 concurrent streams per pod                               │
│                                                                              │
│  Caching:                                                                    │
│  • Common phrases TTS cache (Redis)                                         │
│  • Entity extraction cache                                                  │
│  • Firm configuration cache                                                 │
└─────────────────────────────────────────────────────────────────────────────┘


Regional Deployment:
────────────────────
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  US-East    │     │  US-West    │     │  EU-West    │
│  (Primary)  │     │  (Backup)   │     │  (GDPR)     │
│             │     │             │     │             │
│ Latency to  │     │ Latency to  │     │ Latency to  │
│ NYC: 10ms   │     │ LA: 10ms    │     │ London: 15ms│
└─────────────┘     └─────────────┘     └─────────────┘

Route calls based on caller's location for minimum latency.
```

### Latency Optimization Techniques

```python
class LatencyOptimizer:
    """
    Techniques to minimize end-to-end latency.
    """
    
    # 1. Speculative execution
    async def speculative_response(
        self,
        partial_transcript: str,
        confidence: float,
    ):
        """
        Start generating response before transcript is final.
        
        If confidence > 0.8 and partial looks complete,
        start LLM generation speculatively.
        """
        if confidence > 0.8 and self._looks_complete(partial_transcript):
            # Start generating
            speculative_task = asyncio.create_task(
                self.ai.generate_response(partial_transcript)
            )
            
            # Wait for final transcript
            final_transcript = await self.wait_for_final()
            
            if final_transcript == partial_transcript:
                # Speculation was correct, use the response
                return await speculative_task
            else:
                # Speculation was wrong, cancel and regenerate
                speculative_task.cancel()
                return await self.ai.generate_response(final_transcript)
    
    # 2. TTS caching for common phrases
    async def get_tts_cached(self, text: str) -> bytes:
        """
        Cache commonly used phrases.
        """
        cache_key = f"tts:{hashlib.md5(text.encode()).hexdigest()}"
        
        cached = await self.redis.get(cache_key)
        if cached:
            return cached
        
        audio = await self.tts.synthesize(text)
        
        # Cache for 24 hours
        await self.redis.set(cache_key, audio, ex=86400)
        
        return audio
    
    # Common phrases to pre-cache
    CACHED_PHRASES = [
        "Thank you for calling.",
        "Can you tell me more about that?",
        "I understand.",
        "Let me connect you with an attorney.",
        "One moment please.",
        "Is there anything else?",
    ]
    
    # 3. Connection pooling
    def create_connection_pool(self):
        """
        Maintain persistent connections to reduce latency.
        """
        self.deepgram_pool = ConnectionPool(
            create_connection=self._create_deepgram_stream,
            max_size=100,
            min_size=10,
            max_idle_time=300,
        )
        
        self.openai_pool = AsyncHTTPConnectionPool(
            base_url="https://api.openai.com",
            max_connections=50,
            keepalive_timeout=60,
        )
    
    # 4. Sentence-level streaming
    async def stream_response_sentences(
        self,
        llm_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        """
        Synthesize and stream sentence by sentence.
        
        Don't wait for full response - start TTS on first sentence.
        """
        buffer = ""
        
        async for token in llm_stream:
            buffer += token
            
            # Check for sentence end
            for punct in ['. ', '! ', '? ']:
                if punct in buffer:
                    sentence, buffer = buffer.split(punct, 1)
                    sentence += punct.strip()
                    
                    # Synthesize this sentence
                    async for audio in self.tts.synthesize_streaming(sentence):
                        yield audio
```

---

## 11. Monitoring & Quality Assurance

### Key Metrics Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VOICE AI INTAKE DASHBOARD                                 │
└─────────────────────────────────────────────────────────────────────────────┘

  REAL-TIME METRICS                              QUALITY SCORES
  ─────────────────                              ──────────────
  Active Calls:     127                          Intent Accuracy:    94.2%
  Calls Today:      1,847                        Entity Extraction:  91.8%
  Avg Duration:     4:32                         Caller Satisfaction: 4.3/5
  AI Handle Rate:   78%                          Handoff Success:    98.1%
  
  LATENCY (P95)                                  CONVERSION
  ────────────                                   ──────────
  STT:              180ms  ████████░░            Calls → Leads:      67%
  LLM:              340ms  ██████████░           Leads → Consults:   42%
  TTS:              165ms  ████████░░            Consults → Clients: 31%
  End-to-End:       720ms  █████████░░
  
  CALL OUTCOMES (Last 24h)                       TOP ISSUES
  ─────────────────────────                      ──────────
  ┌──────────────────────────────────────┐       1. "Complex question" (12%)
  │ ████████████████████     AI Complete │       2. "Caller request" (8%)
  │ ████████░░░░░░░░░░░░     Transferred │       3. "Low confidence" (5%)
  │ ███░░░░░░░░░░░░░░░░░     Scheduled   │       4. "Language issue" (3%)
  │ █░░░░░░░░░░░░░░░░░░░     Abandoned   │
  └──────────────────────────────────────┘
```

### Quality Assurance System

```python
class QualityMonitor:
    """
    Monitor and improve intake call quality.
    """
    
    async def evaluate_call(self, call_record: CallRecord) -> QualityScore:
        """
        Evaluate completed call quality.
        """
        scores = {}
        
        # 1. Intent accuracy (did AI understand correctly?)
        scores['intent'] = await self._evaluate_intent(
            call_record.initial_transcript,
            call_record.classified_intent,
            call_record.final_outcome,
        )
        
        # 2. Entity extraction accuracy
        scores['entities'] = await self._evaluate_entities(
            call_record.transcript,
            call_record.extracted_entities,
            call_record.verified_entities,  # If human verified
        )
        
        # 3. Conversation flow
        scores['flow'] = self._evaluate_flow(
            call_record.turns,
            call_record.interruptions,
            call_record.clarifications_needed,
        )
        
        # 4. Outcome appropriateness
        scores['outcome'] = self._evaluate_outcome(
            call_record.case_score,
            call_record.handoff_decision,
            call_record.final_outcome,
        )
        
        # 5. Caller satisfaction (if survey completed)
        if call_record.post_call_survey:
            scores['satisfaction'] = call_record.post_call_survey.score
        
        overall = sum(scores.values()) / len(scores)
        
        return QualityScore(
            overall=overall,
            components=scores,
            flags=self._identify_flags(scores, call_record),
        )
    
    def _evaluate_flow(
        self,
        turns: List[ConversationTurn],
        interruptions: int,
        clarifications: int,
    ) -> float:
        """
        Evaluate conversation flow quality.
        """
        score = 100.0
        
        # Penalize excessive interruptions
        if interruptions > 3:
            score -= (interruptions - 3) * 5
        
        # Penalize excessive clarifications
        if clarifications > 2:
            score -= (clarifications - 2) * 10
        
        # Check response latency
        for turn in turns:
            if turn.ai_response_time > 2000:  # > 2 seconds
                score -= 5
        
        # Check for repetition (AI repeating itself)
        ai_responses = [t.ai_response for t in turns]
        unique_ratio = len(set(ai_responses)) / len(ai_responses)
        if unique_ratio < 0.9:
            score -= 10
        
        return max(0, score) / 100
    
    def _identify_flags(
        self,
        scores: dict,
        call_record: CallRecord,
    ) -> List[QualityFlag]:
        """
        Identify issues for human review.
        """
        flags = []
        
        # Low score components
        for component, score in scores.items():
            if score < 0.7:
                flags.append(QualityFlag(
                    type="low_score",
                    component=component,
                    score=score,
                ))
        
        # Caller requested human
        if call_record.caller_requested_human:
            flags.append(QualityFlag(
                type="caller_request",
                reason=call_record.handoff_reason,
            ))
        
        # Long call duration (might indicate issues)
        if call_record.duration_seconds > 600:  # 10+ minutes
            flags.append(QualityFlag(
                type="long_duration",
                duration=call_record.duration_seconds,
            ))
        
        # Negative sentiment detected
        if call_record.caller_sentiment < 0.3:
            flags.append(QualityFlag(
                type="negative_sentiment",
                sentiment=call_record.caller_sentiment,
            ))
        
        return flags


class ContinuousImprovement:
    """
    Use call data to improve the system.
    """
    
    async def analyze_failure_patterns(self, time_period: str = "7d") -> dict:
        """
        Analyze common failure patterns.
        """
        # Get flagged calls
        flagged_calls = await self.db.get_flagged_calls(time_period)
        
        patterns = {
            "intent_failures": [],
            "entity_failures": [],
            "handoff_reasons": {},
            "low_confidence_phrases": [],
        }
        
        for call in flagged_calls:
            # Analyze intent failures
            if call.intent_score < 0.7:
                patterns["intent_failures"].append({
                    "transcript": call.initial_transcript,
                    "classified": call.classified_intent,
                    "actual": call.actual_intent,
                })
            
            # Track handoff reasons
            if call.was_handed_off:
                reason = call.handoff_reason
                patterns["handoff_reasons"][reason] = \
                    patterns["handoff_reasons"].get(reason, 0) + 1
            
            # Find low-confidence transcriptions
            for segment in call.transcript_segments:
                if segment.confidence < 0.8:
                    patterns["low_confidence_phrases"].append(
                        segment.text
                    )
        
        return patterns
    
    async def generate_improvement_report(self) -> str:
        """
        Generate actionable improvement recommendations.
        """
        patterns = await self.analyze_failure_patterns()
        
        recommendations = []
        
        # Intent classification improvements
        if patterns["intent_failures"]:
            recommendations.append({
                "area": "Intent Classification",
                "issue": f"{len(patterns['intent_failures'])} misclassifications",
                "action": "Add these examples to training data",
                "examples": patterns["intent_failures"][:5],
            })
        
        # STT improvements
        if patterns["low_confidence_phrases"]:
            # Find common words with low confidence
            common_words = self._find_common_low_confidence_words(
                patterns["low_confidence_phrases"]
            )
            recommendations.append({
                "area": "Speech Recognition",
                "issue": f"Low confidence on legal terms",
                "action": "Add to custom vocabulary / keywords",
                "words": common_words[:10],
            })
        
        return recommendations
```

---

## 12. Interview Discussion Points

### Questions They'll Ask

**Q: How do you handle the latency requirements for conversational AI?**

> **A:** Multiple techniques: (1) Streaming at every stage - STT streams transcripts as spoken, LLM streams tokens, TTS starts on first sentence; (2) Speculative execution - start generating response on high-confidence partial transcript; (3) Connection pooling to external APIs; (4) Regional deployment to minimize network hops; (5) Caching common TTS phrases. Target is <800ms end-to-end, with each component budgeted: STT <200ms, LLM start <400ms, TTS first byte <200ms.

**Q: How do you ensure attorney-client privilege for AI-handled calls?**

> **A:** Several layers: (1) Clear disclosure at call start that it's a virtual assistant under attorney supervision; (2) All conversations treated as privileged attorney-client communication; (3) Encrypted storage with per-tenant keys; (4) Third-party agreements (BAAs) with all processors; (5) Opt-out of training for all LLM providers; (6) Access controls and audit logging. The AI acts as an agent of the firm under attorney supervision.

**Q: What happens when the AI can't handle a question?**

> **A:** Graceful handoff system: (1) Detect confidence dropping or complex question; (2) Determine best handoff type - immediate transfer if agents available, queue with wait estimate, or scheduled callback; (3) Prepare context summary for human; (4) Execute warm transfer with "whisper" to agent so they have context; (5) Log for quality improvement. Key is making handoff seamless - caller shouldn't feel abandoned.

**Q: How do you handle different accents and languages?**

> **A:** Language detection at call start (IVR or automatic), then language-specific pipeline: separate STT models optimized for each language, system prompts in that language, TTS voices matching the language. For accents, we use STT providers with strong accent coverage and add legal terminology as custom vocabulary to improve recognition. Spanish is priority #2 after English for US law firms.

**Q: How do you measure and improve quality?**

> **A:** Multi-faceted: (1) Automated scoring on every call - intent accuracy, entity extraction, conversation flow, outcome appropriateness; (2) Post-call surveys for caller satisfaction; (3) Human QA review of flagged calls; (4) Continuous analysis of failure patterns; (5) A/B testing of prompts and voices. Dashboard shows real-time metrics. Goal is >95% intent accuracy and >90% AI-handled rate without handoff.

### Questions to Ask Them

1. **"What's your current AI-to-human handoff rate, and what are the main reasons callers get transferred?"**

2. **"How do you handle after-hours calls - do you have 24/7 human backup or rely entirely on AI with callbacks?"**

3. **"Have you encountered any bar association concerns about AI handling intake? How did you address them?"**

4. **"What's your approach to training the AI on firm-specific information - do you fine-tune per tenant or use context injection?"**

5. **"How do you handle the scenario where two competing law firms both use your platform and a client calls the wrong one?"**

---

## Quick Reference: Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **STT Provider** | Deepgram Nova-2 | Best accuracy + low latency + streaming |
| **LLM** | GPT-4 Turbo (streaming) | Best reasoning + function calling |
| **TTS Provider** | ElevenLabs Turbo | Most natural + low latency |
| **Telephony** | Twilio Media Streams | Reliable + WebSocket support |
| **Call Recording** | Per-tenant encrypted storage | Privilege protection |
| **Language** | Auto-detect + IVR confirm | Best UX + accuracy |
| **Handoff** | Warm transfer with whisper | Context preserved |

---

*This system design covers a production-grade voice AI intake system. Focus on latency optimization and privilege protection as key differentiators for Eve Legal's use case.*
