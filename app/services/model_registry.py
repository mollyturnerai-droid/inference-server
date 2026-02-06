import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from app.core.config import settings

class ModelRegistry:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.STORAGE_PATH + "/model_registry.db"
        self._init_db()

    def _init_db(self):
        """Initialize registry database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT UNIQUE NOT NULL,
                local_path TEXT NOT NULL,
                framework TEXT NOT NULL,
                size_gb REAL,
                download_method TEXT,
                downloaded_at TIMESTAMP,
                last_validated TIMESTAMP,
                validation_status TEXT,
                metadata_json TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS download_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT NOT NULL,
                method TEXT NOT NULL,
                success BOOLEAN,
                error_message TEXT,
                attempted_at TIMESTAMP,
                duration_seconds REAL
            )
        """)
        
        conn.commit()
        conn.close()

    def register_model(self, repo_id: str, local_path: Path, framework: str, size_gb: float, method: str, metadata: Dict[str, Any]):
        """Register successfully downloaded model."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO models 
            (repo_id, local_path, framework, size_gb, download_method, downloaded_at, validation_status, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            repo_id,
            str(local_path),
            framework,
            size_gb,
            method,
            datetime.now(),
            "validated",
            json.dumps(metadata)
        ))
        conn.commit()
        conn.close()

    def get_model_path(self, repo_id: str) -> Optional[Path]:
        """Retrieve local path for registered model."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT local_path FROM models WHERE repo_id = ? AND validation_status = 'validated'",
            (repo_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        return Path(result[0]) if result else None

    def log_download_attempt(self, repo_id: str, method: str, success: bool, 
                           error: Optional[str] = None, duration: float = 0.0):
        """Log download attempt for analytics."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO download_attempts 
            (repo_id, method, success, error_message, attempted_at, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (repo_id, method, success, error, datetime.now(), duration))
        conn.commit()
        conn.close()

    def get_best_method(self, repo_id: str) -> Optional[str]:
        """Determine best download method based on historical success."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT method, 
                   SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
                   COUNT(*) as total
            FROM download_attempts
            WHERE repo_id = ?
            GROUP BY method
            ORDER BY (SUM(CASE WHEN success THEN 1 ELSE 0 END) * 1.0 / COUNT(*)) DESC
            LIMIT 1
        """, (repo_id,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result and result[1] > 0 else None

model_registry = ModelRegistry()
