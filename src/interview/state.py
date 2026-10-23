import time
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class InterviewStage(str, Enum):
    INTRO = "intro"
    RESUME_QUESTIONS = "resume_questions"
    BEHAVIOURAL = "behavioural"
    ROLE_QUESTIONS = "role_questions"
    CANDIDATE_QUESTIONS = "candidate_questions"
    CLOSING = "closing"

class QuestionType(str, Enum):
    INTRO = "intro"
    RESUME = "resume_grounded"
    BEHAVIOURAL = "behavioural_star"
    ROLE_JD = "role_jd_link"
    CANDIDATE_INQUIRY = "candidate_inquiry"
    FOLLOWUP = "follow_up"
    CLOSING = "closing"

class InterviewQuestion(BaseModel):
    id: str
    text: str
    stage: InterviewStage
    question_type: QuestionType
    grounding_source: str = ""  # Specific resume project/skill or JD requirement
    asked_at: float = Field(default_factory=time.time)
    follow_ups: List[str] = Field(default_factory=list)
    answers: List[str] = Field(default_factory=list)

class TranscriptTurn(BaseModel):
    turn_id: str
    speaker: str  # "agent" or "candidate"
    stage: str
    text: str
    start_time: float
    end_time: float
    latency_ms: Optional[float] = None
    question_id: Optional[str] = None
    is_followup: bool = False

class InterviewState(BaseModel):
    session_id: str
    candidate_name: str = "Candidate"
    target_role: str = "Software Engineer"
    stage: InterviewStage = InterviewStage.INTRO
    
    # State counters & constraints
    questions_asked: int = 0
    question_budget: int = 5
    time_limit_mins: int = 15
    start_time: float = Field(default_factory=time.time)
    last_turn_time: float = Field(default_factory=time.time)
    
    # Active question tracking
    current_question: Optional[InterviewQuestion] = None
    current_followup_count: int = 0
    max_followups: int = 2
    
    # Turn log & notes
    turns: List[TranscriptTurn] = Field(default_factory=list)
    questions_history: List[InterviewQuestion] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    is_completed: bool = False
    closing_notes: str = ""
    
    # Guardrails
    safety_flags_triggered: List[str] = Field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def time_left_seconds(self) -> float:
        remaining = (self.time_limit_mins * 60) - self.elapsed_seconds
        return max(0.0, remaining)

    def is_time_expired(self) -> bool:
        return self.time_left_seconds <= 0

    def should_wrap_up(self) -> bool:
        """Returns True if question budget is met or time is running low (< 60s)."""
        return (self.questions_asked >= self.question_budget) or (self.time_left_seconds < 60)

    def advance_stage(self) -> InterviewStage:
        """Sequential stage progression."""
        transitions = {
            InterviewStage.INTRO: InterviewStage.RESUME_QUESTIONS,
            InterviewStage.RESUME_QUESTIONS: InterviewStage.BEHAVIOURAL,
            InterviewStage.BEHAVIOURAL: InterviewStage.ROLE_QUESTIONS,
            InterviewStage.ROLE_QUESTIONS: InterviewStage.CANDIDATE_QUESTIONS,
            InterviewStage.CANDIDATE_QUESTIONS: InterviewStage.CLOSING,
            InterviewStage.CLOSING: InterviewStage.CLOSING
        }
        self.stage = transitions.get(self.stage, InterviewStage.CLOSING)
        return self.stage

    def record_turn(
        self,
        speaker: str,
        text: str,
        start_time: float,
        end_time: float,
        latency_ms: Optional[float] = None,
        is_followup: bool = False
    ) -> TranscriptTurn:
        turn = TranscriptTurn(
            turn_id=f"turn_{len(self.turns) + 1}",
            speaker=speaker,
            stage=self.stage.value,
            text=text,
            start_time=start_time,
            end_time=end_time,
            latency_ms=latency_ms,
            question_id=self.current_question.id if self.current_question else None,
            is_followup=is_followup
        )
        self.turns.append(turn)
        self.last_turn_time = end_time
        return turn
