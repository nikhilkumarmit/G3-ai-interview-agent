import os
import time
import asyncio
from typing import Optional, Callable, Dict, Any, List
from loguru import logger
import config
from src.voice.latency import latency_tracker

"""
=============================================================================
VAD CONFIGURATION & THINKING PAUSE NOTE (Requirement #4 & PDF Checkpoint):
-----------------------------------------------------------------------------
Candidates preparing for high-stakes HR rounds frequently take pauses of 
1.0 - 1.5 seconds to mentally formulate their answers into structured formats 
(such as the STAR framework: Situation, Task, Action, Result).

Premature turn-ending during these thinking pauses leads to severe frustration 
and inaccurate grading. We therefore configure Silero VAD with:
- `stop_secs`: 1.2 seconds (Generous silence buffer to protect candidate thinking pauses)
- `confidence`: 0.5 (Optimal threshold balancing speech clarity against ambient noise)
- `min_volume`: 0.6 (Prevents background breathing or keyboard clicks from resetting silence)
=============================================================================
"""

class VoicePipelineConfig:
    def __init__(
        self,
        stop_secs: float = config.VAD_STOP_SECS,
        confidence: float = config.VAD_CONFIDENCE,
        min_volume: float = config.VAD_MIN_VOLUME,
        allow_interruptions: bool = True
    ):
        self.stop_secs = stop_secs
        self.confidence = confidence
        self.min_volume = min_volume
        self.allow_interruptions = allow_interruptions

class VoicePipelineSession:
    """
    Manages an active audio session with VAD, Speech-To-Text, Interruption Handling,
    LLM reasoning via function tools, and Text-To-Speech output.
    """

    def __init__(
        self,
        session_id: str,
        on_candidate_transcript: Optional[Callable[[str, float, float], Any]] = None,
        on_agent_speaking: Optional[Callable[[str], Any]] = None,
        on_interruption: Optional[Callable[[], Any]] = None
    ):
        self.session_id = session_id
        self.config = VoicePipelineConfig()
        self.on_candidate_transcript = on_candidate_transcript
        self.on_agent_speaking = on_agent_speaking
        self.on_interruption = on_interruption

        # State tracking
        self.is_candidate_speaking = False
        self.is_agent_speaking = False
        self.speech_start_time: float = 0.0
        self.speech_end_time: float = 0.0
        self.interruption_count: int = 0
        self.current_agent_utterance: str = ""

    def on_vad_speech_start(self, timestamp: Optional[float] = None):
        """Called when VAD detects candidate starts speaking."""
        now = timestamp or time.time()
        self.is_candidate_speaking = True
        self.speech_start_time = now

        # Interruption handling: If agent was talking, stop agent immediately
        if self.is_agent_speaking:
            self.interruption_count += 1
            self.is_agent_speaking = False
            logger.info(f"[INTERRUPTION] Candidate interrupted agent in session {self.session_id}! Halting TTS playback.")
            if self.on_interruption:
                self.on_interruption()

    def on_vad_speech_stop(self, timestamp: Optional[float] = None) -> float:
        """Called when candidate ceases speaking for > stop_secs (1.2s thinking pause cushion)."""
        now = timestamp or time.time()
        self.is_candidate_speaking = False
        self.speech_end_time = now
        latency_tracker.mark_candidate_finished(now)
        logger.debug(f"[VAD] Candidate speech stopped at {now:.3f} (cushion: {self.config.stop_secs}s)")
        return now

    async def process_spoken_transcript(self, transcript_text: str) -> Dict[str, Any]:
        """
        Processes recognized candidate speech, measures turn latency,
        and triggers agent voice response.
        """
        clean_text = transcript_text.strip()
        if not clean_text:
            return {"status": "empty"}

        # Member 1 -> Member 2 Agreed Interface:
        # on_candidate_transcript(text: str, start_time: float, end_time: float)
        start_t = self.speech_start_time or (time.time() - 2.0)
        end_t = self.speech_end_time or time.time()

        if self.on_candidate_transcript:
            result = self.on_candidate_transcript(clean_text, start_t, end_t)
        else:
            from src.interview.tools import interview_manager
            result = interview_manager.process_candidate_answer(self.session_id, clean_text)

        # Mark first agent audio latency
        latency_ms = latency_tracker.mark_first_audio_output()

        agent_text = result.get("agent_response") or result.get("question_text", "")
        self.is_agent_speaking = True
        self.current_agent_utterance = agent_text

        if self.on_agent_speaking:
            self.on_agent_speaking(agent_text)

        return {
            "session_id": self.session_id,
            "candidate_transcript": clean_text,
            "agent_response": agent_text,
            "latency_ms": latency_ms,
            "stage": result.get("stage", ""),
            "is_completed": result.get("is_completed", False),
            "is_followup": result.get("is_followup", False),
            "interruption_count": self.interruption_count
        }

    def trigger_interruption(self):
        """Simulate or handle manual interruption from browser UI."""
        if self.is_agent_speaking:
            self.interruption_count += 1
            self.is_agent_speaking = False
            logger.info(f"[INTERRUPT] Manual interruption triggered for session {self.session_id}")
            if self.on_interruption:
                self.on_interruption()
            return True
        return False
