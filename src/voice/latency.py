import time

class LatencyTracker:
    def __init__(self):
        self.speech_end_time = None
        self.first_audio_time = None

    def mark_candidate_speech_end(self):
        self.speech_end_time = time.time()

    def mark_agent_audio_start(self):
        if self.speech_end_time:
            self.first_audio_time = time.time()
            latency = self.first_audio_time - self.speech_end_time
            print(f"[Metrics] Candidate-stop-to-agent-audio latency: {latency:.3f} seconds")
            return latency
        return None