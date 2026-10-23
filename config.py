import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Server config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Interview Constraints & Budget
QUESTION_BUDGET = int(os.getenv("QUESTION_BUDGET", 5))
INTERVIEW_TIME_LIMIT_MINS = int(os.getenv("INTERVIEW_TIME_LIMIT_MINS", 15))
MAX_FOLLOWUPS_PER_QUESTION = int(os.getenv("MAX_FOLLOWUPS_PER_QUESTION", 2))

# Voice Activity Detection (VAD) configuration
"""
VAD CONFIGURATION & THINKING PAUSE NOTE:
To prevent cutting off the candidate during critical 'thinking pauses'
(such as organizing a response following the STAR behavioral method),
Silero VAD is configured with:
- `stop_secs`: 1.2s - Generous cushion of silence before concluding a turn.
- `confidence`: 0.5 - Balanced sensitivity to reject ambient noise.
- `min_volume`: 0.6 - Ignores breathing or background clicks.
"""
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", 1.2))
VAD_CONFIDENCE = float(os.getenv("VAD_CONFIDENCE", 0.5))
VAD_MIN_VOLUME = float(os.getenv("VAD_MIN_VOLUME", 0.6))

# File storage directories
SESSIONS_DIR = BASE_DIR / os.getenv("SESSION_STORAGE_DIR", "sessions")
REPORTS_DIR = BASE_DIR / os.getenv("REPORTS_STORAGE_DIR", "reports")
DATA_DIR = BASE_DIR / "data"

for directory in [SESSIONS_DIR, REPORTS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Safety Guardrails: Protected topics that must never be asked
PROTECTED_TOPICS = [
    "age", "religion", "caste", "marital status", "marital", "pregnancy",
    "children", "sexual orientation", "gender identity", "disability",
    "health status", "medical history", "political affiliation", "ethnicity"
]
