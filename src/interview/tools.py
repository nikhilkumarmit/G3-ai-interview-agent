import time
from typing import Dict, Any, List, Optional
from loguru import logger

from src.interview.state import (
    InterviewState,
    InterviewStage,
    InterviewQuestion,
    QuestionType
)
from src.interview.question_engine import QuestionEngine

INTERVIEW_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "next_question",
            "description": "Moves to the next grounded interview question based on resume and JD analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_text": {
                        "type": "string",
                        "description": "The exact question text formulated for the candidate."
                    },
                    "grounding_source": {
                        "type": "string",
                        "description": "The specific resume project, skill, or JD requirement grounded in this question."
                    },
                    "stage": {
                        "type": "string",
                        "description": "The current interview stage."
                    }
                },
                "required": ["question_text", "grounding_source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_answer",
            "description": "Records candidate answer, checks for vagueness, and manages follow-up limit (max 2).",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_response": {
                        "type": "string",
                        "description": "The transcript text of the candidate's answer."
                    },
                    "is_vague": {
                        "type": "boolean",
                        "description": "Whether the answer lacks concrete detail or STAR structure."
                    },
                    "followup_needed": {
                        "type": "boolean",
                        "description": "Whether to ask a follow-up question."
                    }
                },
                "required": ["candidate_response", "is_vague"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "end_interview",
            "description": "Gracefully concludes the interview session when budget is reached or candidate is done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Reason for ending (e.g. 'budget_reached', 'closing_completed', 'candidate_requested')"
                    }
                }
            }
        }
    }
]

class InterviewManager:
    """
    Coordinates session state, question engine, and tool execution.
    Maintains active interview sessions in-memory while persisting to disk via SessionStore.
    """

    def __init__(self):
        self.active_sessions: Dict[str, InterviewState] = {}
        self.question_engines: Dict[str, QuestionEngine] = {}
        self.planned_questions: Dict[str, List[InterviewQuestion]] = {}
        self.question_pointers: Dict[str, int] = {}

    def register_session(self, state: InterviewState, question_engine: QuestionEngine):
        session_id = state.session_id
        self.active_sessions[session_id] = state
        self.question_engines[session_id] = question_engine
        
        # Pre-plan grounded questions
        plan = question_engine.generate_interview_plan()
        self.planned_questions[session_id] = plan
        self.question_pointers[session_id] = 0

        # Set first question
        if plan:
            state.current_question = plan[0]
            state.stage = plan[0].stage

    def get_session(self, session_id: str) -> Optional[InterviewState]:
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        # Automatic recovery from SessionStore on server reload/restart
        try:
            from src.persistence.session_store import session_store
            data = session_store.load_session(session_id)
            if data:
                data.pop("_last_persisted_at", None)
                state = InterviewState(**data)
                self.active_sessions[session_id] = state

                # Create fallback engine if missing
                if session_id not in self.question_engines:
                    from src.ingestion.resume_loader import CandidateSummary, IngestionResult
                    summary = CandidateSummary(
                        name=state.candidate_name,
                        target_role=state.target_role,
                        skills=[],
                        projects=[]
                    )
                    dummy_ingestion = IngestionResult(
                        resume_chunks=[],
                        jd_chunks=[],
                        candidate_summary=summary,
                        raw_resume_text="",
                        raw_jd_text=""
                    )
                    self.question_engines[session_id] = QuestionEngine(dummy_ingestion)

                return state
        except Exception as e:
            logger.debug(f"Could not restore session {session_id} from disk: {e}")

        return None

    # Tool Execution Handlers
    def execute_next_question(self, session_id: str) -> Dict[str, Any]:
        """Tool: next_question()"""
        state = self.active_sessions.get(session_id)
        if not state:
            return {"error": f"Session {session_id} not found"}

        plan = self.planned_questions.get(session_id, [])
        ptr = self.question_pointers.get(session_id, 0) + 1

        if ptr < len(plan) and not state.should_wrap_up():
            next_q = plan[ptr]
            self.question_pointers[session_id] = ptr
            state.current_question = next_q
            state.stage = next_q.stage
            state.current_followup_count = 0
            state.questions_asked += 1
            
            logger.info(f"Session {session_id} advanced to question {ptr+1}: {next_q.id} (Stage: {next_q.stage.value})")
            return {
                "action": "ask_question",
                "question_id": next_q.id,
                "question_text": next_q.text,
                "stage": next_q.stage.value,
                "grounding_source": next_q.grounding_source,
                "is_followup": False
            }
        else:
            return self.execute_end_interview(session_id, reason="budget_completed")

    def execute_save_answer(self, session_id: str, candidate_text: str) -> Dict[str, Any]:
        """Tool: save_answer()"""
        state = self.active_sessions.get(session_id)
        engine = self.question_engines.get(session_id)
        if not state or not engine:
            return {"error": "Invalid session"}

        # Safety check on candidate response
        safety_flag = engine.check_safety_violation(candidate_text)
        if safety_flag:
            state.safety_flags_triggered.append(safety_flag)
            logger.warning(f"Safety flag caught in session {session_id}: {safety_flag}")

        current_q = state.current_question
        if current_q:
            current_q.answers.append(candidate_text)

        # Candidate Questions Stage Handling
        if state.stage == InterviewStage.CANDIDATE_QUESTIONS:
            ans_to_candidate = engine.get_candidate_questions_response(candidate_text)
            closing = engine.generate_closing_statement(state.candidate_name)
            combined_response = f"{ans_to_candidate}\n\n{closing}"
            state.stage = InterviewStage.CLOSING
            state.is_completed = True
            return {
                "action": "end_interview",
                "agent_response": combined_response,
                "stage": "closing",
                "is_completed": True
            }

        # Evaluate completeness & decide follow-up
        is_vague, followup_text = engine.evaluate_answer_completeness(current_q, candidate_text)

        if is_vague and state.current_followup_count < state.max_followups and not state.should_wrap_up():
            state.current_followup_count += 1
            if current_q:
                current_q.follow_ups.append(followup_text)
            logger.info(f"Triggering follow-up #{state.current_followup_count} for session {session_id}")
            return {
                "action": "ask_followup",
                "agent_response": followup_text,
                "stage": state.stage.value,
                "followup_index": state.current_followup_count,
                "is_followup": True
            }
        else:
            # Advance to next core question
            return self.execute_next_question(session_id)

    def execute_end_interview(self, session_id: str, reason: str = "closing") -> Dict[str, Any]:
        """Tool: end_interview()"""
        state = self.active_sessions.get(session_id)
        engine = self.question_engines.get(session_id)
        if not state:
            return {"error": "Invalid session"}

        closing_msg = (
            engine.generate_closing_statement(state.candidate_name)
            if engine
            else f"Thank you {state.candidate_name}. That concludes our session. Your evaluation report is ready."
        )

        state.stage = InterviewStage.CLOSING
        state.is_completed = True
        state.closing_notes = f"Interview ended. Reason: {reason}."
        logger.info(f"Interview ended for session {session_id}. Reason: {reason}")

        return {
            "action": "end_interview",
            "agent_response": closing_msg,
            "stage": InterviewStage.CLOSING.value,
            "is_completed": True,
            "reason": reason
        }

    def process_candidate_answer(self, session_id: str, transcript: str, latency_ms: Optional[float] = None) -> Dict[str, Any]:
        """
        Agreed Interface (Member 2 -> Member 1):
        process_candidate_answer(session_id: str, transcript: str) -> dict
        Processes transcript turn, updates state machine, and returns next response.
        """
        state = self.active_sessions.get(session_id)
        if not state:
            return {"error": f"Session {session_id} not registered"}

        now = time.time()
        # Record candidate transcript turn
        state.record_turn(
            speaker="candidate",
            text=transcript,
            start_time=state.last_turn_time,
            end_time=now,
            latency_ms=latency_ms
        )

        # Check for wrap up condition
        if state.should_wrap_up() and state.stage != InterviewStage.CANDIDATE_QUESTIONS and state.stage != InterviewStage.CLOSING:
            # Move to candidate questions before closing
            plan = self.planned_questions.get(session_id, [])
            inquiry_q = next((q for q in plan if q.stage == InterviewStage.CANDIDATE_QUESTIONS), None)
            if inquiry_q:
                state.stage = InterviewStage.CANDIDATE_QUESTIONS
                state.current_question = inquiry_q
                state.record_turn("agent", inquiry_q.text, now, now + 1.0)
                return {
                    "action": "ask_question",
                    "agent_response": inquiry_q.text,
                    "stage": "candidate_questions",
                    "is_completed": False
                }

        # Process through tools
        result = self.execute_save_answer(session_id, transcript)
        
        # Record agent response turn
        response_text = result.get("agent_response") or result.get("question_text", "")
        if response_text:
            state.record_turn(
                speaker="agent",
                text=response_text,
                start_time=now,
                end_time=time.time(),
                is_followup=result.get("is_followup", False)
            )

        return result

# Global manager instance
interview_manager = InterviewManager()
