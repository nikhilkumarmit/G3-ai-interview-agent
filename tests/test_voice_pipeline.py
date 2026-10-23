import pytest
import time
import asyncio
from src.voice.pipeline import VoicePipelineSession, VoicePipelineConfig
from src.voice.latency import LatencyTracker
from src.voice.accent_evaluator import calculate_word_error_rate, evaluate_accented_speech_fairness
import config

def test_vad_thinking_pause_setting():
    """Verify that VAD stop_secs is configured with >= 1.0s buffer for candidate thinking pauses."""
    session = VoicePipelineSession(session_id="test_vad")
    assert session.config.stop_secs >= 1.0, "VAD stop_secs must be at least 1.0s to accommodate thinking pauses"
    assert session.config.stop_secs == config.VAD_STOP_SECS

def test_latency_tracker():
    """Verify candidate-stop-to-first-agent-audio latency measurement."""
    tracker = LatencyTracker()
    tracker.reset()

    t0 = time.time()
    tracker.mark_candidate_finished(t0)
    
    # Simulate 250ms processing time
    t1 = t0 + 0.250
    latency_ms = tracker.mark_first_audio_output(t1)

    assert latency_ms is not None
    assert 240 <= latency_ms <= 260
    
    summary = tracker.get_summary_metrics()
    assert summary["count"] == 1
    assert 240 <= summary["avg_ms"] <= 260

def test_interruption_handling():
    """Verify that candidate speech during agent utterance interrupts the agent immediately."""
    interrupted_called = False

    def on_interrupt():
        nonlocal interrupted_called
        interrupted_called = True

    session = VoicePipelineSession(
        session_id="test_interrupt",
        on_interruption=on_interrupt
    )

    # Agent starts speaking
    session.is_agent_speaking = True
    assert session.is_agent_speaking is True

    # Candidate starts speaking (VAD speech start)
    session.on_vad_speech_start()

    assert session.is_agent_speaking is False, "Agent speaking flag should be immediately set to False on interruption"
    assert interrupted_called is True, "Interruption callback should be fired"
    assert session.interruption_count == 1

def test_accented_speech_wer_and_fairness():
    """Verify WER computation and accent bias evaluation report."""
    res = evaluate_accented_speech_fairness()
    assert res["wer"] > 0.0
    assert "fairness_analysis" in res
    assert "PgBouncer" in res["fairness_analysis"] or "PostgreSQL" in res["fairness_analysis"]
    print(f"\nAccented speech WER: {res['wer_percentage']}")
