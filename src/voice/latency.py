import time
from typing import Optional, List, Dict
from loguru import logger

class TurnLatencyRecord:
    def __init__(self, turn_number: int, stop_timestamp: float, first_audio_timestamp: float, latency_ms: float):
        self.turn_number = turn_number
        self.stop_timestamp = stop_timestamp
        self.first_audio_timestamp = first_audio_timestamp
        self.latency_ms = latency_ms

    def to_dict(self) -> Dict[str, float]:
        return {
            "turn_number": self.turn_number,
            "latency_ms": round(self.latency_ms, 2),
            "stop_timestamp": self.stop_timestamp,
            "first_audio_timestamp": self.first_audio_timestamp
        }

class LatencyTracker:
    """
    Measures candidate-stop-to-first-agent-audio latency per turn.
    Fulfills Requirement #3: Print latency for every turn.
    """

    def __init__(self):
        self.candidate_stopped_time: Optional[float] = None
        self.turn_records: List[TurnLatencyRecord] = []
        self._current_turn: int = 1

    def mark_candidate_finished(self, timestamp: Optional[float] = None) -> float:
        """Mark when candidate finishes speaking / VAD triggers turn end."""
        self.candidate_stopped_time = timestamp or time.time()
        logger.debug(f"[LATENCY] Candidate stopped speaking at {self.candidate_stopped_time:.3f}")
        return self.candidate_stopped_time

    def mark_first_audio_output(self, timestamp: Optional[float] = None) -> Optional[float]:
        """
        Mark when first audio chunk from agent TTS plays to candidate.
        Calculates and logs the turn latency in milliseconds.
        """
        now = timestamp or time.time()
        if self.candidate_stopped_time is not None:
            latency_ms = (now - self.candidate_stopped_time) * 1000.0
            record = TurnLatencyRecord(
                turn_number=self._current_turn,
                stop_timestamp=self.candidate_stopped_time,
                first_audio_timestamp=now,
                latency_ms=latency_ms
            )
            self.turn_records.append(record)
            self._current_turn += 1
            
            # Print turn latency per Requirement #3
            print(f"\n⚡ [LATENCY METRIC] Turn #{record.turn_number} Candidate Stop -> First Agent Audio: {latency_ms:.1f} ms")
            logger.info(f"[LATENCY] Turn #{record.turn_number} latency: {latency_ms:.1f} ms")
            
            self.candidate_stopped_time = None
            return latency_ms
        return None

    def get_summary_metrics(self) -> Dict[str, float]:
        """Returns aggregate latency metrics across the interview."""
        if not self.turn_records:
            return {"count": 0, "avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

        latencies = [r.latency_ms for r in self.turn_records]
        return {
            "count": len(latencies),
            "avg_ms": round(sum(latencies) / len(latencies), 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2)
        }

    def reset(self):
        self.candidate_stopped_time = None
        self.turn_records.clear()
        self._current_turn = 1

latency_tracker = LatencyTracker()
