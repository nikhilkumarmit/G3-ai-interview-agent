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

        # 2. Resume-Grounded Questions (At least 3 specific to the uploaded resume)
        projects = self.candidate.projects
        skills = self.candidate.skills
        highlights = self.candidate.experience_highlights

        # Resume Question 1 (Grounding on candidate's primary project or experience)
        if len(projects) > 0:
            p1 = projects[0]
            text_q1 = (
                f"In your resume, you highlighted your work on '{p1}'. "
                f"Could you walk me through the architectural and design decisions you made, "
                f"the most critical technical challenges you faced, and how you evaluated its performance?"
            )
            grounding_q1 = f"Resume Project: {p1}"
        elif len(highlights) > 0:
            exp1 = highlights[0]
            text_q1 = (
                f"In your resume, you detailed your experience: '{exp1[:90]}...'. "
                f"Could you walk me through your specific role in that initiative and the concrete technical challenges you solved?"
            )
            grounding_q1 = f"Resume Experience: {exp1[:50]}"
        elif len(skills) >= 2:
            s_pair = f"{skills[0]} and {skills[1]}"
            text_q1 = (
                f"Your resume emphasizes strong proficiency in {s_pair}. "
                f"Can you share a complex, real-world project where you applied these technologies and the key tradeoffs you considered?"
            )
            grounding_q1 = f"Resume Skills: {s_pair}"
        else:
            text_q1 = (
                "Could you walk me through the most technically demanding project or system you have built, "
                "focusing on your direct ownership and architectural decisions?"
            )
            grounding_q1 = "Resume Background Overview"

        q_resume_1 = InterviewQuestion(
            id=f"q_{q_idx}",
            stage=InterviewStage.RESUME_QUESTIONS,
            question_type=QuestionType.RESUME,
            text=text_q1,
            grounding_source=grounding_q1
        )
        questions.append(q_resume_1)
        q_idx += 1

        # Resume Question 2 (Grounding on candidate's achievements or second project)
        if len(highlights) > 0:
            exp_ref = highlights[0]
            text_q2 = (
                f"Looking closely at your background, you mentioned: '{exp_ref[:90]}...'. "
                f"What were the major technical bottlenecks or unexpected constraints you encountered, and how did you measure your success?"
            )
            grounding_q2 = f"Resume Experience Highlight: {exp_ref[:50]}"
        elif len(projects) > 1:
            p2 = projects[1]
            text_q2 = (
                f"You also featured '{p2}' in your resume. "
                f"Could you discuss how you structured the system to handle scaling, data consistency, or error recovery?"
            )
            grounding_q2 = f"Resume Project: {p2}"
        elif len(skills) > 1:
            s_top = skills[:3]
            text_q2 = (
                f"You highlighted hands-on expertise with {', '.join(s_top)}. "
                f"How have you leveraged these tools to build resilient, maintainable production workflows?"
            )
            grounding_q2 = f"Resume Skills: {', '.join(s_top)}"
        else:
            text_q2 = "Can you share an engineering challenge where you had to debug a difficult failure or performance bottleneck?"
            grounding_q2 = "Resume Technical Experience"

        q_resume_2 = InterviewQuestion(
            id=f"q_{q_idx}",
            stage=InterviewStage.RESUME_QUESTIONS,
            question_type=QuestionType.RESUME,
            text=text_q2,
            grounding_source=grounding_q2
        )
        questions.append(q_resume_2)
        q_idx += 1

        # Resume Question 3 (Grounding on another project, experience, or specialized skill)
        if len(projects) > 1:
            p_target = projects[1] if len(projects) == 2 else projects[2] if len(projects) > 2 else projects[0]
            q_resume_3 = InterviewQuestion(
                id=f"q_{q_idx}",
                stage=InterviewStage.RESUME_QUESTIONS,
                question_type=QuestionType.RESUME,
                text=(
                    f"Regarding '{p_target}', what was the most difficult edge case you resolved during development, "
                    f"and what testing or validation strategy did you employ to ensure production readiness?"
                ),
                grounding_source=f"Resume Project: {p_target}"
            )
        elif len(highlights) > 1:
            exp2 = highlights[1]
            q_resume_3 = InterviewQuestion(
                id=f"q_{q_idx}",
                stage=InterviewStage.RESUME_QUESTIONS,
                question_type=QuestionType.RESUME,
                text=(
                    f"In another section of your experience, you noted: '{exp2[:90]}...'. "
                    f"What architectural principles guided your implementation here?"
                ),
                grounding_source=f"Resume Experience: {exp2[:50]}"
            )
        else:
            focus_skill = skills[1] if len(skills) > 1 else (skills[0] if skills else "software architecture")
            q_resume_3 = InterviewQuestion(
                id=f"q_{q_idx}",
                stage=InterviewStage.RESUME_QUESTIONS,
                question_type=QuestionType.RESUME,
                text=f"Can you dive into a deep technical challenge you solved using {focus_skill} and how it influenced the overall architecture?",
                grounding_source=f"Resume Skill: {focus_skill}"
            )
        questions.append(q_resume_3)
        q_idx += 1

        # 3. Behavioural Question (STAR Format)
        q_star = InterviewQuestion(
            id=f"q_{q_idx}",
            stage=InterviewStage.BEHAVIOURAL,
            question_type=QuestionType.BEHAVIOURAL,
            text=(
                "Tell me about a time when an unexpected critical obstacle, production issue, or conflicting requirement arose on a project under your watch. "
                "Walk me through the Situation, the Task you took on, the exact Actions you executed, and the Resulting outcome."
            ),
            grounding_source="Behavioural STAR Competency Assessment"
        )
        questions.append(q_star)
        q_idx += 1

        # 4. Role-Specific Question Linking JD Requirement to Candidate Resume
        # Dynamically extract key requirement from ingested JD chunks
        jd_req_found = ""
        for chunk in self.ingestion.jd_chunks:
            if chunk.section in ["requirements", "responsibilities"] or any(k in chunk.content.lower() for k in ["responsible", "design", "build", "experience with", "architect", "develop", "lead"]):
                # Extract first meaningful requirement sentence
                sentences = [s.strip() for s in chunk.content.replace("\n", ". ").split(".") if len(s.strip()) > 20]
                if sentences:
                    jd_req_found = sentences[0].lstrip("-•* ")
                    break

        if not jd_req_found:
            jd_req_found = f"delivering scalable solutions for the {self.candidate.target_role} position"

        # Dynamically find the best matching candidate skill
        matched_skills = [s for s in skills if s.lower() in self.ingestion.raw_jd_text.lower()]
        matching_skill = ", ".join(matched_skills[:2]) if matched_skills else (skills[0] if skills else "your technical background")

        q_role = InterviewQuestion(
            id=f"q_{q_idx}",
            stage=InterviewStage.ROLE_QUESTIONS,
            question_type=QuestionType.ROLE_JD,
            text=(
                f"The job description for {self.candidate.target_role} emphasizes '{jd_req_found}'. "
                f"Given your background with {matching_skill}, how would you approach this responsibility, "
                f"and what architectural tradeoffs or best practices would you prioritize?"
            ),
            grounding_source=f"JD Requirement ({jd_req_found[:45]}...) linked to Resume Skill ({matching_skill})"
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
                f"We always want to ensure this role aligns with your career goals. "
                f"What questions do you have for our team regarding the {self.candidate.target_role} role or technical challenges?"
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
        """Dynamic, authentic response to candidate questions tailored to the target role."""
        lower = candidate_question.lower()
        role = self.candidate.target_role
        skills_str = ", ".join(self.candidate.skills[:3]) if self.candidate.skills else "modern technologies"

        if "culture" in lower or "team" in lower:
            return (
                f"Great question! For the {role} team, we prioritize autonomous ownership, "
                "continuous peer mentoring, and psychological safety. We value transparent technical reviews and blameless post-mortems."
            )
        elif "tech" in lower or "stack" in lower or "architecture" in lower:
            return (
                f"For the {role} position, our technical stack leverages {skills_str} and cloud-native services. "
                "We emphasize high reliability, observability, and clean architectural separation of concerns."
            )
        else:
            return (
                f"That's a thoughtful question regarding the {role} opportunity. In this position, you would collaborate closely "
                f"across engineering and product teams to solve high-impact challenges and drive technical excellence."
            )

    def generate_closing_statement(self, candidate_name: str) -> str:
        """Polite interview conclusion adhering to Requirement #5 & #6 (no hire/reject language)."""
        return (
            f"Thank you very much for your time today, {candidate_name}. "
            f"That concludes our mock interview session. Your full transcript and speech metrics have been recorded, "
            f"and our evidence-based evaluator is now compiling your rubric feedback report with detailed strengths and growth areas. "
            f"You can review your complete report shortly!"
        )
