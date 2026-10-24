import sys
import uvicorn
from loguru import logger
import config

if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("Starting AI Mock Interview Coach Studio")
    logger.info(f"Server URL: http://localhost:{config.PORT}")
    logger.info(f"VAD Thinking Pause Buffer: {config.VAD_STOP_SECS} seconds")
    logger.info(f"Default Question Budget: {config.QUESTION_BUDGET} questions")
    logger.info("=" * 65)
    
    uvicorn.run(
        "src.api.app:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level="info"
    )
