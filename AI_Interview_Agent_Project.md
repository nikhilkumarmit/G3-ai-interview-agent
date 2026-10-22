# End-to-End Voice Agent Project — AI Mock Interview Coach

## Scenario

You are preparing for a placement HR round. Build a voice agent that reads your resume and a job description, interviews you by voice with questions based on them, asks a follow-up when an answer is weak, and gives you a feedback report at the end.

It only gives feedback. It never decides to select or reject anyone.

Build it as a single script (or notebook) that runs end to end.

---

## Architecture

```
Browser/mic -> VAD -> STT -> LLM (state + tools) -> TTS -> speaker
                                |
            tools: next_question(), save_answer(), end_interview()
                                |
        interview state: stage, questions asked, time left, notes
                                |
                    (call ke baad) evaluator LLM -> rubric scores -> report
```

---

## Requirements

### 1. Ingestion
- Load a resume (PDF or text) and a job description.
- Keep the section label (skills, projects, experience) as metadata on every chunk.
- Print a summary: candidate name, skills found, number of projects, target role.

### 2. Interview State
- Stages: intro, resume questions, behavioural (STAR), role questions, candidate's questions, closing.
- Keep the state in your code (stage, questions asked, time left, notes), not only in the LLM's memory.
- Set a question budget and a time limit. Wrap up on time.

### 3. Voice Pipeline
- Build it with Pipecat: mic, VAD, STT, LLM, TTS, speaker.
- Use function calling for the tools: `next_question`, `save_answer`, `end_interview`.
- Interruption: the agent stops talking when the candidate speaks.
- Print latency for every turn (candidate stops speaking to first agent audio).

### 4. Questions and Follow-ups
- Generate questions from resume and JD chunks. At least 3 must name something specific from the resume, and 1 must link a JD requirement to the resume.
- Do not invent experience. Do not repeat a question.
- After each answer, decide: follow-up or move on. A vague answer gets a follow-up, maximum 2 per question.
- A thinking pause must not cut the candidate off. Note your VAD setting in a comment.

### 5. Scoring and Report
- Define a rubric first (for example relevance, structure, specificity, communication), with a meaning for each score.
- Score every answer after the interview, with a quote from the transcript as evidence.
- Compute from STT timestamps: words per minute, long pauses, filler words.
- Report: scores with evidence, 3 strengths, 3 improvements, speech signals, full transcript. No hire or reject language.

### 6. Safety
- At the start, say it is an AI and the session is recorded.
- Instructions hidden in the resume or JD (for example "give 10/10") are data, not commands.
- Never ask about age, religion, caste, marital status, health or other protected topics, even if prompted.

### 7. Persistence
- Save the transcript and state after every turn.
- After a restart, load a saved session and generate the report without redoing the interview.

### 8. Evaluation Pass
Run 6 tests automatically with scripted text answers. Feed at least one recorded audio clip through STT.
- 2 grounding tests: questions must match items in the resume.
- 2 follow-up tests: a vague answer triggers a follow-up, a complete answer moves on.
- 2 safety tests: an injected instruction in the resume is ignored, a request to ask an illegal question is refused.

Print a pass/fail table. Also print:
- the score spread when the same answer is scored 5 times
- the score difference when only the candidate's name is changed

---

## Constraints

- No hardcoded questions except the opening and closing lines.
- Every score and claim in the report must trace to a transcript quote.
- No hiring decisions or suggestions.
- The session must survive a restart.
- At least one test with an accented-speech clip. Report its WER and explain how a wrong transcript could unfairly change a score.
- Fake or consented data only.
