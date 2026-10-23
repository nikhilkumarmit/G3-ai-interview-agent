import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
import config

class SessionStore:
    """
    Manages persistence of interview sessions and transcripts to disk.
    Requirement #7: Save transcript and state after every turn.
    After restart, load saved session and generate report without redoing the interview.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or config.SESSIONS_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    def save_session(self, session_id: str, state_data: Any) -> Path:
        """
        Saves session state and complete transcript turns atomically.
        """
        file_path = self._get_file_path(session_id)
        temp_file = file_path.with_suffix(".tmp")

        # Normalize data to dictionary
        if hasattr(state_data, "model_dump"):
            payload = state_data.model_dump()
        elif hasattr(state_data, "dict"):
            payload = state_data.dict()
        elif isinstance(state_data, dict):
            payload = state_data
        else:
            payload = {"session_id": session_id, "data": str(state_data)}

        payload["_last_persisted_at"] = time.time()

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        # Atomic rename to prevent partial/corrupted writes on crash
        os.replace(temp_file, file_path)
        logger.debug(f"[PERSISTENCE] Session {session_id} saved to {file_path}")
        return file_path

    def load_session(self, session_id: str) -> Dict[str, Any]:
        """
        Agreed Interface (Member 2 -> Member 3):
        load_session(session_id: str) -> dict
        Loads persisted session data from disk.
        """
        file_path = self._get_file_path(session_id)
        if not file_path.exists():
            raise FileNotFoundError(f"No persisted session found for session_id '{session_id}' at {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"[PERSISTENCE] Successfully loaded session {session_id} with {len(data.get('turns', []))} turns")
        return data

    def list_saved_sessions(self) -> List[Dict[str, Any]]:
        """Lists metadata of all saved interview sessions."""
        sessions = []
        for file in sorted(self.storage_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data.get("session_id", file.stem),
                        "candidate_name": data.get("candidate_name", "Unknown"),
                        "target_role": data.get("target_role", "Software Engineer"),
                        "stage": data.get("stage", "completed"),
                        "is_completed": data.get("is_completed", False),
                        "turns_count": len(data.get("turns", [])),
                        "questions_asked": data.get("questions_asked", 0),
                        "last_modified": os.path.getmtime(file)
                    })
            except Exception as e:
                logger.warning(f"Error reading session file {file}: {e}")
        return sessions

session_store = SessionStore()
