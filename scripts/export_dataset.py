"""
Export anonymized dataset for publication.

Produces three CSV files in scripts/export/:
  sentences.csv      — all sentences (id, text, dataset_type, length)
  sessions.csv       — experiment_results with user_id replaced by anon_id
  guesses.csv        — character_guesses at positions >= 70, without timestamps

User anonymization: salted SHA-256, first 8 hex digits → 32-bit unsigned int.
The salt is read from EXPORT_SALT env var (set it once, keep it private).
"""

import asyncio
import csv
import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.init_db import async_session
from app.models.user import User
from app.models.experiment_results import ExperimentResult
from app.models.character_guess import CharacterGuess
from app.models.sentence import Sentence

OUTPUT_DIR = Path(__file__).parent.parent / "dataset_export"


def anon_id(user_id: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}{user_id}".encode()).hexdigest()
    return int(digest[:8], 16)


async def export():
    salt = os.environ.get("EXPORT_SALT")
    if not salt:
        print("ERROR: EXPORT_SALT env var not set. Generate one with:")
        print("  python3 -c \"import secrets; print(secrets.token_hex(32))\"")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    async with async_session() as session:
        users = list((await session.execute(select(User))).scalars().all())
        sentences = list((await session.execute(select(Sentence))).scalars().all())
        experiments = list((await session.execute(select(ExperimentResult))).scalars().all())
        guesses = list((await session.execute(
            select(CharacterGuess).where(CharacterGuess.position >= 70)
        )).scalars().all())

    user_map = {u.id: anon_id(u.id, salt) for u in users}

    if len(set(user_map.values())) != len(user_map):
        print("WARNING: anon_id collision detected — increase hash prefix length")

    sentences_path = OUTPUT_DIR / "sentences.csv"
    with open(sentences_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "length"])
        for s in sentences:
            w.writerow([s.id, s.text, s.length])
    print(f"Wrote {len(sentences):>5} rows → {sentences_path}")

    sessions_path = OUTPUT_DIR / "sessions.csv"
    with open(sessions_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "anon_user_id", "sentence_id", "finished"])
        for e in experiments:
            w.writerow([
                e.id,
                user_map[e.user_id],
                e.sentence_id,
                e.finished,
            ])
    print(f"Wrote {len(experiments):>5} rows → {sessions_path}")

    guesses_path = OUTPUT_DIR / "guesses.csv"
    with open(guesses_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "session_id", "position", "guess_number", "guessed_character", "is_correct"])
        for g in guesses:
            w.writerow([
                g.id,
                g.experiment_result_id,
                g.position,
                g.guess_number,
                g.guessed_character,
                g.is_correct,
            ])
    print(f"Wrote {len(guesses):>5} rows → {guesses_path}")



asyncio.run(export())