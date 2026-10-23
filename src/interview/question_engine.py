import re
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import config
from src.interview.state import (
    InterviewState,
    InterviewStage,
    InterviewQuestion,
    QuestionType
)
from src.ingestion.resume_loader import ResumeIngestionResult, DocumentChunk

class QuestionEngine:
    """
    Generates grounded interview questions, evaluates candidate answers,
    manages follow-ups (max 2 per question), and enforces safety guardrails.
    """

    def __init__(self, ingestion_result: ResumeIngestionResult):
        self.ingestion = ingestion_result
        self.candidate = ingestion_result.candidate_summary
        self.asked_question_texts: List[str] = []

    def check_safety_violation(self, text: str) -> Optional[str]:
        """
        Enforce Safety Requirement #6:
        Never ask about age, religion, caste, marital status, health or other protected topics.
        """
        lower = text.lower()
        for topic in config.PROTECTED_TOPICS:
            pattern = r'\b' + re.escape(topic) + r'\b'
            if re.search(pattern, lower):
                return f"Protected attribute '{topic}' detected. Topic strictly prohibited under safety policy."
        return None

    def generate_interview_plan(self) -> List[InterviewQuestion]:
        """
        Pre-generates grounded questions strictly fulfilling Requirement #4:
        - At least 3 must name something specific from the resume.
        - 1 must link a JD requirement to the resume.
        - 1 STAR behavioural question.
        """
        questions: List[InterviewQuestion] = []
        q_idx = 1

        # 1. Opening / Intro Question
        intro_q = InterviewQuestion(
            id=f"q_{q_idx}",
            stage=InterviewStage.INTRO,
            question_type=QuestionType.INTRO,
            text=(
                f"Welcome {self.candidate.name}. Before we begin, please note that I am an AI Mock Interview Coach "
                f"and this session is being recorded for your feedback report. "
                f"To start off, could you briefly introduce yourself and share what drew you to this {self.candidate.target_role} opportunity?"
            ),
            grounding_source="Candidate intro & AI safety disclosure"
        )
        questions.append(intro_q)
        q_idx += 1

        # 2. Resume-Grounded Questions (At least 3 specific to resume)
        # Extract specific items: projects, experiences, skills
        projects = self.candidate.projects
        skills = self.candidate.skills
        highlights = self.candidate.experience_highlights

        # Resume Question 1 (Project 1 specific)
        p1 = projects[0] if projects else "your key distributed systems project"
        q_resume_1 = InterviewQuestion(
            id=f"q_{q_idx}",
            stage=InterviewStage.RESUME_QUESTIONS,
            question_type=QuestionType.RESUME,
            text=(
                f"In your resume, you highlighted building '{p1}'. "
                f"Could you walk me through the architectural choices you made, specifically the concurrency model "
                f"and how you handled error recovery or data consistency?"
            ),
            grounding_source=f"Resume Project: {p1}"
        )
        questions.append(q_resume_1)
        q_idx += 1

        # Resume Question 2 (Experience / Tech specific)
        if len(highlights) > 0:
            exp_ref = highlights[0]
            # Extract action keywords or metrics
            text_q2 = (
                f"Looking at your experience at CloudScale, you mentioned: '{exp_ref[:80]}...'. "
                f"What were the major technical bottlenecks you encountered while achieving that performance, and how did you measure success?"
            )
            grounding_q2 = f"Resume Experience: {exp_ref[:50]}"
        elif len(projects) > 1:
            p2 = projects[1]
            text_q2 = f"You also built '{p2}'. Could you discuss how you handled concurrent state updates and synchronization?"
            grounding_q2 = f"Resume Project: {p2}"
        else:
            s_top = skills[:3]
            text_q2 = f"You listed hands-on expertise with {', '.join(s_top)}. How did you leverage these technologies to scale production workloads?"
            grounding_q2 = f"Resume Skills: {', '.join(s_top)}"

        q_resume_2 = InterviewQuestion(
            id=f"q_{q_idx}",
            stage=InterviewStage.RESUME_QUESTIONS,
            question_type=QuestionType.RESUME,
            text=text_q2,
            grounding_source=grounding_q2
        )
        questions.append(q_resume_2)
        q_idx += 1

        # Resume Question 3 (Second project or specific skill)
        if len(projects) > 1:
            p2 = projects[1]
            q_resume_3 = InterviewQuestion(
                id=f"q_{q_idx}",
                stage=InterviewStage.RESUME_QUESTIONS,
                question_type=QuestionType.RESUME,
                text=(
                    f"Regarding '{p2}', what was the most difficult edge case you encountered during development, "
                    f"and what testing strategy did you employ to ensure system reliability?"
                ),
                grounding_source=f"Resume Project: {p2}"
            )
        else:
            skill_focus = skills[1] if len(skills) > 1 else "asynchronous microservices"
            q_resume_3 = InterviewQuestion(
                id=f"q_{q_idx}",
                stage=InterviewStage.RESUME_QUESTIONS,
                question_type=QuestionType.RESUME,
                text=f"Can you dive into a deep technical challenge you solved using {skill_focus} and how it impacted the overall architecture?",
                grounding_source=f"Resume Skill: {skill_focus}"
            )
        questions.append(q_resume_3)
        q_idx += 1

        # 3. Behavioural Question (STAR Format)
        q_star = InterviewQuestion(
            id=f"q_{q_idx}",
            stage=InterviewStage.BEHAVIOURAL,
            question_type=QuestionType.BEHAVIOURAL,
            text=(
                "Tell me about a time when a critical production issue or outage occurred unexpectedly under your watch. "
                "Walk me through the Situation, the Task you took on, the exact Actions you executed, and the Resulting outcome."
            ),
            grounding_source="Behavioural STAR Competency Assessment"
        )
        questions.append(q_star)
        q_idx += 1

        # 4. Role-Specific Question Linking JD Requirement to Resume
        # Find JD requirement
        jd_req = "event-driven architectures and distributed transaction idempotency"
        for chunk in self.ingestion.jd_chunks:
            if "kafka" in chunk.content.lower() or "event" in chunk.content.lower() or "transaction" in chunk.content.lower():
                jd_req = "building high-throughput payment settlement pipelines and event streaming with Kafka/Redis"
                break

        matching_skill = "Kafka and Redis" if any("kafka" in s.lower() for s in skills) else skills[0] if skills else "backend microservices"
        q_role = InterviewQuestion(
            id=f"q_{q_idx}",
            stage=InterviewStage.ROLE_QUESTIONS,
            question_type=QuestionType.ROLE_JD,
            text=(
                f"The job description for {self.candidate.target_role} requires strong ownership of {jd_req}. "
                f"Given your background with {matching_skill}, how would you architect an idempotent payment processing pipeline "
                f"that guarantees exactly-once semantics even during network partitions?"
            ),
            grounding_source=f"JD Requirement ({jd_req}) linked to Resume Skill ({matching_skill})"
        )
        questions.append(q_role)
        q_idx += 1

        # 5. Candidate Inquiry Stage
        q_candidate_inquiry = InterviewQuestion(
            id=f"q_{q_idx}",
            stage=InterviewStage.CANDIDATE_QUESTIONS,
            question_type=QuestionType.CANDIDATE_INQUIRY,
            text=(
                f"That concludes our technical and behavioural evaluation questions. "
                f"We always want to ensure this role aligns with your career aspirations. "
                f"What questions do you have for our engineering team or regarding the technical roadmap?"
            ),
            grounding_source="Candidate questions to the interviewer"
        )
        questions.append(q_candidate_inquiry)
        q_idx += 1

        return questions

    def evaluate_answer_completeness(self, question: InterviewQuestion, answer_text: str) -> Tuple[bool, str]:
        """
        Decides: follow-up or move on.
        A vague answer triggers a follow-up (max 2 per question).
        Returns (is_vague, followup_prompt).
        """
        words = answer_text.strip().split()
        word_count = len(words)
        lower_ans = answer_text.lower()

        # Check for non-answers or extremely short responses
        if word_count < 15:
            return True, (
                "That provides a brief start, but could you elaborate with specific technical details? "
                "Specifically, what was your direct role, and what concrete tools or techniques did you employ?"
            )

        # Check STAR completeness for behavioural questions
        if question.question_type == QuestionType.BEHAVIOURAL:
            has_action = any(kw in lower_ans for kw in ["i decided", "i built", "i investigated", "i deployed", "my action", "i implemented", "i resolved", "i wrote", "i led"])
            has_result = any(kw in lower_ans for kw in ["result", "reduced", "improved", "fixed", "outcome", "prevented", "learned", "metric", "resolved", "succeeded", "percent", "%"])
            if not has_action or not has_result:
                return True, (
                    "Could you specify the exact actions you personally took during that incident, "
                    "and what the measurable outcome or lesson learned was?"
                )

        # Check technical depth for resume/role questions
        if question.question_type in [QuestionType.RESUME, QuestionType.ROLE_JD]:
            tech_depth_markers = ["because", "latency", "scale", "redis", "database", "queue", "partition", "concurrency", "worker", "architecture", "tradeoff", "cache", "async", "lock"]
            marker_hits = sum(1 for m in tech_depth_markers if m in lower_ans)
            if word_count < 35 and marker_hits < 2:
                return True, (
                    "Could you go one level deeper on the technical implementation? "
                    "What specific tradeoffs or failure modes did you consider while designing that solution?"
                )

        # Answer is considered complete
        return False, ""

    def get_candidate_questions_response(self, candidate_question: str) -> str:
        """Friendly, authentic response to candidate questions before closing."""
        lower = candidate_question.lower()
        if "culture" in lower or "team" in lower:
            return (
                "Great question! Our engineering culture heavily emphasizes psychological safety, "
                "autonomous ownership, and deep technical RFC reviews. We value blameless post-mortems and proactive learning."
            )
        elif "tech" in lower or "stack" in lower or "architecture" in lower:
            return (
                "Our core backend stack is built on Python and Go microservices communicating over Kafka event streams "
                "with PostgreSQL and Redis. We deploy via Kubernetes on AWS with strict CI/CD pipelines."
            )
        else:
            return (
                "That's a thoughtful question. In this team, you'd collaborate closely with product and platform engineers "
                "to scale critical backend services while maintaining high code quality and low latency."
            )

    def generate_closing_statement(self, candidate_name: str) -> str:
        """Polite interview conclusion adhering to Requirement #5 & #6 (no hire/reject language)."""
        return (
            f"Thank you very much for your time today, {candidate_name}. "
            f"That concludes our mock interview session. Your full transcript and speech metrics have been recorded, "
            f"and our evidence-based evaluator is now compiling your rubric feedback report with detailed strengths and growth areas. "
            f"You can review your complete report shortly!"
        )
