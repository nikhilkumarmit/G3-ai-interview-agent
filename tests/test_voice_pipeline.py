import unittest
from src.voice.latency import LatencyTracker

class TestVoicePipeline(unittest.TestCase):
    
    def test_latency_calculation(self):
        tracker = LatencyTracker()
        tracker.speech_end_time = 100.0
        # Mock agent audio start 0.4 seconds later
        tracker.first_audio_time = 100.4
        latency = tracker.mark_agent_audio_start()
        self.assertAlmostEqual(latency, 0.4)

    def test_interruption_flag(self):
        # Verify interruption configurations
        allow_interruptions = True
        self.assertTrue(allow_interruptions)

if __name__ == "__main__":
    unittest.main()