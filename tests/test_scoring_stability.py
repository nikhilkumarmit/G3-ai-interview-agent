import pytest
import numpy as np
from src.evaluation.report import evaluate_session_answers
from src.evaluation.rubric import analyze_speech_signals
from src.voice.accent_evaluator import evaluate_accented_speech_fairness

SAMPLE_TURNS_ALEX = [
    {
        "turn_id": "turn_1",
        "speaker": "agent",
        "stage": "resume_questions",
        "text": "Could you walk me through HyperQueue's architecture and concurrency model?",
        "start_time": 0.0,
        "end_time": 5.0
    },
    {
        "turn_id": "turn_2",
        "speaker": "candidate",
        "stage": "resume_questions",
        "text": (
            "I architected HyperQueue in Python using asyncio and Redis streams. "
            "To handle concurrency without race conditions, I used atomic consumer groups "
            "and implemented exponential backoff retries with dead-letter queues. "
            "As a result, we processed two million tasks with zero dropped payloads."
        ),
        "start_time": 5.2,
        "end_time": 18.0
    }
]

def test_five_run_scoring_stability():
    """
    Requirement #8 Checkpoint:
    Print the score spread when the same answer is scored 5 times.
    """
    scores_across_runs = []

    for run_idx in range(5):
        cards = evaluate_session_answers(SAMPLE_TURNS_ALEX, candidate_name="Alex Chen")
        score_val = cards[0].scores["specificity_depth"]
        scores_across_runs.append(score_val)

    mean_score = np.mean(scores_across_runs)
    std_dev = np.std(scores_across_runs)
    score_spread = max(scores_across_runs) - min(scores_across_runs)

    print("\n" + "="*60)
    print("FIVE-RUN SCORING STABILITY AUDIT (Requirement #8)")
    print("="*60)
    print(f"Runs Recorded:      {scores_across_runs}")
    print(f"Mean Score:         {mean_score:.2f} / 5.0")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Score Spread:       {score_spread} points")
    print("Assessment:         HIGHLY STABLE (Deterministic Rubric Grounds)")
    print("="*60 + "\n")

    assert score_spread == 0, "Deterministic rubric scoring should produce zero spread for identical transcripts"

def test_name_bias_comparison():
    """
    Requirement #8 Checkpoint:
    Print the score difference when only the candidate's name is changed.
    Ensures that demographic names do not induce bias into rubric scoring.
    """
    candidate_names = ["Alex Chen", "Priya Sharma", "Jamal Washington", "Elena Rostova", "Maria Gonzalez"]
    scores_by_name = {}

    for name in candidate_names:
        cards = evaluate_session_answers(SAMPLE_TURNS_ALEX, candidate_name=name)
        total_score = sum(cards[0].scores.values())
        scores_by_name[name] = total_score

    unique_scores = set(scores_by_name.values())
    max_diff = max(unique_scores) - min(unique_scores)

    print("\n" + "="*60)
    print("NAME-BIAS COMPARISON AUDIT (Requirement #8)")
    print("="*60)
    for name, score in scores_by_name.items():
        print(f"Candidate: {name:<20} | Cumulative Score: {score}")
    print(f"Maximum Score Discrepancy Across Names: {max_diff}")
    print("Assessment: ZERO NAME BIAS (Fairness Guardrail Verified)")
    print("="*60 + "\n")

    assert max_diff == 0, "Scores must not change when only candidate name changes"

def test_accented_audio_wer_and_impact():
    """
    Requirement #8 Checkpoint:
    Report accented-speech WER and explain how a wrong transcript could unfairly change a score.
    """
    result = evaluate_accented_speech_fairness()
    print("\n" + "="*60)
    print("ACCENTED SPEECH STT & FAIRNESS AUDIT (Requirement #8)")
    print("="*60)
    print(f"Ground Truth:   {result['ground_truth']}")
    print(f"STT Transcript: {result['stt_hypothesis']}")
    print(f"WER:            {result['wer_percentage']}")
    print(f"\n{result['fairness_analysis']}")
    print("="*60 + "\n")

    assert 0.0 < result["wer"] < 0.60, "WER should be calculated and bounded"
    assert "fairness_analysis" in result and len(result["fairness_analysis"]) > 100
