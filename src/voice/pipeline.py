import os
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transports.services.smallwebrtc import SmallWebRTCTransport

async def run_voice_pipeline():
    # Initialize Silero VAD with thinking-pause tuning
    vad = SileroVADAnalyzer(params={"stop_secs": 1.2})
    
    # Configure transport (using SmallWebRTC or local audio transport)
    transport = SmallWebRTCTransport()

    # Define pipeline flow (Microphone -> VAD -> STT -> LLM -> TTS -> Speaker)
    pipeline = Pipeline([
        transport.input(),
        vad,
        # Add STT, LLM, and TTS services here
        transport.output()
    ])

    task = PipelineTask(pipeline, PipelineParams(
        allow_interruptions=True,
        enable_metrics=True
    ))

    runner = PipelineRunner()
    await runner.run(task)