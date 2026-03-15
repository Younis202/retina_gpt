"""
db/cases_db.py — Cases Database
================================
SQLite database that saves every analysis automatically.
No external database needed — single file, zero config.

Stores:
    - Every analyzed image (results + metadata)
    - Patient ID, date, DR grade, confidence
    - Full JSON result for retrieval
    - Image path for re-analysis

Enables:
    GET /cases          → list all cases (dashboard)
    GET /cases/{id}     → single case detail
    Data Flywheel       → every scan saved for retraining
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_PATH = Path("database/retina_cases.db")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id          TEXT PRIMARY KEY,
                patient_id  TEXT DEFAULT 'Unknown',
                created_at  TEXT NOT NULL,
                image_name  TEXT,

                -- Quick-access fields (for list view)
                dr_grade        INTEGER,
                dr_label        TEXT,
                dr_confidence   REAL,
                dr_refer        INTEGER,
                quality_score   REAL,
                quality_adequate INTEGER,
                risk_level      TEXT,

                -- Full result JSON
                full_result     TEXT NOT NULL,

                -- Status
                status      TEXT DEFAULT 'completed'
            )
        """)
        conn.commit()


def save_case(
    result_dict: Dict[str, Any],
    patient_id: str = "Unknown",
    image_name: str = "",
) -> str:
    """
    Save an analysis result to the database.
    Called automatically after every /analyze.

    Returns the case ID.
    """
    init_db()

    case_id = result_dict.get("image_id") or str(uuid.uuid4())[:8]
    dr = result_dict.get("dr_grading", {})
    quality = result_dict.get("quality", {})

    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO cases
            (id, patient_id, created_at, image_name,
             dr_grade, dr_label, dr_confidence, dr_refer,
             quality_score, quality_adequate,
             full_result, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            case_id,
            patient_id,
            datetime.utcnow().isoformat(),
            image_name,
            dr.get("grade", -1),
            dr.get("label", ""),
            dr.get("confidence", 0.0),
            1 if dr.get("refer", False) else 0,
            quality.get("score", 0.0),
            1 if quality.get("adequate", True) else 0,
            json.dumps(result_dict),
            "completed",
        ))
        conn.commit()

    return case_id


def get_cases(
    limit: int = 50,
    offset: int = 0,
    patient_id: Optional[str] = None,
    dr_grade: Optional[int] = None,
    refer_only: bool = False,
) -> List[Dict]:
    """Get list of cases for dashboard."""
    init_db()

    query = "SELECT * FROM cases WHERE 1=1"
    params = []

    if patient_id:
        query += " AND patient_id LIKE ?"
        params.append(f"%{patient_id}%")
    if dr_grade is not None:
        query += " AND dr_grade = ?"
        params.append(dr_grade)
    if refer_only:
        query += " AND dr_refer = 1"

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def get_case(case_id: str) -> Optional[Dict]:
    """Get single case with full result."""
    init_db()

    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM cases WHERE id = ?", (case_id,)
        ).fetchone()

    if not row:
        return None

    case = dict(row)
    case["full_result"] = json.loads(case["full_result"])
    return case


def get_stats() -> Dict:
    """Get dashboard statistics."""
    init_db()

    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        today = conn.execute(
            "SELECT COUNT(*) FROM cases WHERE created_at >= date('now')"
        ).fetchone()[0]
        this_week = conn.execute(
            "SELECT COUNT(*) FROM cases WHERE created_at >= date('now', '-7 days')"
        ).fetchone()[0]
        referable = conn.execute(
            "SELECT COUNT(*) FROM cases WHERE dr_refer = 1"
        ).fetchone()[0]
        grade_dist = conn.execute(
            "SELECT dr_grade, COUNT(*) as count FROM cases GROUP BY dr_grade"
        ).fetchall()

    return {
        "total_cases": total,
        "today": today,
        "this_week": this_week,
        "referable_cases": referable,
        "dr_grade_distribution": {
            str(row[0]): row[1] for row in grade_dist if row[0] >= 0
        },
    }


def delete_case(case_id: str) -> bool:
    """Delete a case."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))
        conn.commit()
        return cursor.rowcount > 0
