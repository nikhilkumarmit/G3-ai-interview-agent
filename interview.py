import os
import time
from typing import Optional
from loguru import logger

# Pipecat framework imports (verified for v1.13.0)
from pipecat.frames.frames import EndFrame, LLMMessagesAppendFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer

# 1. VAD CONFIGURATION & THINKING PAUSE NOTE

"""
VAD SETTING NOTE:
To ensure the agent does not cut off the candidate during a 'thinking pause' 
(allowing them time to structure responses using STAR format), we configure Silero VAD with:
- `stop_secs`: 1.2 seconds (Generous silence buffer before concluding turn).
- `confidence`: 0.5 (Balanced threshold for speech clarity).
- `min_volume`: 0.6 (Filters out background breath/noise).
"""
vad_analyzer = SileroVADAnalyzer(
    params=dict(
        stop_secs=1.2,        
        confidence=0.5,
        min_volume=0.6
    )
)


# 2. TURN LATENCY TRACKER

class LatencyTracker:
    def __init__(self):
        self.candidate_stopped_speaking_time: Optional[float] = None

    def mark_candidate_finished(self):
        self.candidate_stopped_speaking_time = time.time()

    def mark_first_audio_output(self):
        if self.candidate_stopped_speaking_time:
            latency = (time.time() - self.candidate_stopped_speaking_time) * 1000
            logger.info(f"[METRIC] Turn Latency (Candidate stop -> First Agent Audio): {latency:.2f} ms")
            self.candidate_stopped_speaking_time = None

latency_tracker = LatencyTracker()

# 3. FUNCTION CALLING TOOLS SETUP

tools = [
    {
        "type": "function",
        "function": {
            "name": "next_question",
            "description": "Moves to the next core interview question grounded in resume items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_text": {"type": "string", "description": "The exact question text to ask."},
                    "grounding_source": {"type": "string", "description": "Specific resume item or JD match."}
                },
                "required": ["question_text", "grounding_source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_answer",
            "description": "Evaluates the candidate's response, tracking follow-up counts (max 2 if vague).",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_response": {"type": "string"},
                    "is_vague": {"type": "boolean", "description": "True if response lacks specific detail."}
                },
                "required": ["candidate_response", "is_vague"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "end_interview",
            "description": "Concludes the interview session.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

async def handle_next_question(function_name, tool_call_id, arguments, llm, context):
    q_text = arguments.get("question_text")
    logger.info(f"[TOOL] next_question -> {q_text}")
    await llm.queue_frames([LLMMessagesAppendFrame(context.get_messages())])

async def handle_save_answer(function_name, tool_call_id, arguments, llm, context):
    logger.info(f"[TOOL] save_answer -> {arguments}")

async def handle_end_interview(function_name, tool_call_id, arguments, llm, context):
    logger.info("[TOOL] end_interview -> Closing session.")
    await llm.queue_frames([EndFrame()])


# 4. PIPECAT VOICE PIPELINE

async def run_voice_pipeline(transport_instance):
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = GroqLLMService(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
    tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # Register tools
    llm.register_function("next_question", handle_next_question)
    llm.register_function("save_answer", handle_save_answer)
    llm.register_function("end_interview", handle_end_interview)

    # Initialize universal LLMContext & use llm.create_context_aggregator()
    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an HR mock interview coach. Ground your questions in the provided resume/JD. "
                    "At least 3 questions must name something specific from the resume, and 1 must link a JD requirement. "
                    "If an answer is vague, trigger a follow-up (max 2 per question)."
                )
            }
        ],
        tools=tools
    )
    context_aggregator = llm.create_context_aggregator(context)

    # Pipecat Pipeline assembly with interruption support enabled
    pipeline = Pipeline([
        transport_instance.input(),
        vad_analyzer,
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport_instance.output(),
        context_aggregator.assistant()
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport_instance.event_handler("on_client_connected")
    async def on_client_connected(transport):
        logger.info("Candidate connected to voice pipeline.")

    runner = PipelineRunner()
    await runner.run(task)