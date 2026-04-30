import datetime

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean, func
from app.db.init_db import Base


class CharacterGuess(Base):
    __tablename__ = "character_guesses"

    id = Column(Integer, primary_key=True, index=True)
    experiment_result_id = Column(Integer, ForeignKey("experiment_results.id"), nullable=False)
    position = Column(Integer, nullable=False)  # Character position in sentence
    guess_number = Column(Integer, nullable=False)  # Attempt number
    guessed_character = Column(String(1), nullable=False)
    is_correct = Column(Boolean, nullable=False, default=False)
    timestamp = Column(DateTime, default=func.now())