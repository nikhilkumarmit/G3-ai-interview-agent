import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class RubricDimension(BaseModel):
    name: str
    description: str
    scale_meanings: Dict[int, str]

RUBRIC_DEFINITIONS: Dict[str, RubricDimension] = {
    "relevance": RubricDimension(
        name="Relevance & Directness",
        description="Assesses whether the response directly addresses the question asked without deflecting or drifting off-topic.",
        scale_meanings={
            1: "Completely off-topic or evasive; fails to answer the question.",
            2: "Partially answers the prompt, but rambles or focuses on unrelated trivia.",
            3: "Adequately addresses the main question, with minor irrelevant tangents.",
            4: "Directly and clearly answers all components of the prompt.",
            5: "Exceptionally targeted; directly addresses the core prompt and anticipates relevant context."
        }
    ),
    "structure_star": RubricDimension(
        name="Structure & STAR Method",
        description="Evaluates logical progression and adherence to the STAR framework (Situation, Task, Action, Result).",
        scale_meanings={
            1: "Unstructured stream of consciousness; no discernible situation or outcome.",
            2: "Missing critical elements (e.g. only describes the situation, no actions taken or results).",
            3: "Basic narrative arc present; mentions actions and outcome but transitions are abrupt.",
            4: "Well-structured STAR response; clearly delineates candidate actions from team actions and states the outcome.",
            5: "Masterful STAR delivery; seamless narrative flow with quantified results and reflection."
        }
    ),
    "specificity_depth": RubricDimension(
        name="Specificity & Technical Depth",
        description="Assesses concrete technical evidence, architecture details, tradeoffs, and metrics vs generic buzzwords.",
        scale_meanings={
            1: "Pure buzzword drop without comprehension; zero technical depth.",
            2: "High-level generic statements; mentions technologies without explaining how they were applied.",
            3: "Provides concrete examples and names specific tools, but lacks analysis of architectural tradeoffs.",
            4: "High technical fidelity; discusses implementation nuances, failure handling, and quantitative metrics.",
            5: "Deep engineering mastery; rigorously explains concurrency, tradeoffs, failure modes, and architectural rationale."
        }
    ),
    "communication": RubricDimension(
        name="Communication & Conciseness",
        description="Evaluates verbal conciseness, pacing, avoidance of filler words, and professional clarity.",
        scale_meanings={
            1: "Very difficult to follow; rampant fillers and disorganized phrasing.",
            2: "Frequently loses train of thought or repeats points excessively.",
            3: "Clear and understandable communication with acceptable pacing.",
            4: "Articulate, confident, and professional with good phrasing and minimal fillers.",
            5: "Compelling, crisp, and executive-level clarity; exceptional economy of words."
        }
    ),
    "technical_alignment": RubricDimension(
        name="Role & Technical Alignment",
        description="Evaluates alignment with the target job description requirements and technical competencies.",
        scale_meanings={
            1: "Lacks foundational prerequisites for the target role.",
            2: "Demonstrates only peripheral familiarity with target stack and principles.",
            3: "Meets baseline requirements for the position.",
            4: "Strong alignment; demonstrates hands-on mastery of required systems.",
            5: "Exemplary alignment; demonstrates senior-level architectural thought leadership."
        }
    )
}

COMMON_FILLER_WORDS = [
    "um", "uh", "like", "you know", "actually", "basically",
    "sort of", "kind of", "so yeah", "i mean", "right"
]

def analyze_speech_signals(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes speech metrics from STT transcript turns:
    - Words Per Minute (WPM)
    - Long pauses (> 3.0s)
    - Filler words count and breakdown
    """
    candidate_turns = [t for t in turns if t.get("speaker") == "candidate"]
    
    total_words = 0
    total_speaking_duration_secs = 0.0
    long_pauses_count = 0
    filler_counts: Dict[str, int] = {fw: 0 for fw in COMMON_FILLER_WORDS}
    total_fillers = 0

    prev_end_time: Optional[float] = None

    for turn in candidate_turns:
        text = turn.get("text", "")
        start_time = float(turn.get("start_time", 0.0))
        end_time = float(turn.get("end_time", 0.0))
        duration = max(1.0, end_time - start_time)

        words = text.split()
        total_words += len(words)
        total_speaking_duration_secs += duration

        # Check for pause before this turn
        if prev_end_time is not None:
            gap = start_time - prev_end_time
            if gap > 3.0:
                long_pauses_count += 1
        prev_end_time = end_time

        # Detect filler words using regex word boundaries
        lower_text = text.lower()
        for fw in COMMON_FILLER_WORDS:
            pattern = r'\b' + re.escape(fw) + r'\b'
            matches = len(re.findall(pattern, lower_text))
            if matches > 0:
                filler_counts[fw] += matches
                total_fillers += matches

    # Calculate WPM
    minutes = total_speaking_duration_secs / 60.0
    wpm = round(total_words / minutes, 1) if minutes > 0.1 else 135.0

    # Rate pacing
    pacing_verdict = "Optimal Pacing (120-160 WPM)"
    if wpm < 110:
        pacing_verdict = "Slow Pacing (< 110 WPM) - Consider speaking with more energy"
    elif wpm > 175:
        pacing_verdict = "Fast Pacing (> 175 WPM) - Consider breathing pauses to allow interviewer processing"

    return {
        "words_per_minute": wpm,
        "pacing_verdict": pacing_verdict,
        "total_words_spoken": total_words,
        "total_speaking_seconds": round(total_speaking_duration_secs, 1),
        "long_pauses_count": long_pauses_count,
        "total_fillers_detected": total_fillers,
        "filler_words_breakdown": {k: v for k, v in filler_counts.items() if v > 0}
    }
