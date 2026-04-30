import enum
import datetime

from sqlalchemy import Column, Integer, ForeignKey, DateTime, Boolean, func, String
from app.db.init_db import Base


class DatasetType(str, enum.Enum):
    FORMAL = "formal"
    INFORMAL = "informal"


class ExperimentResult(Base):
    __tablename__ = "experiment_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), ForeignKey("users.id"), nullable=False)
    sentence_id = Column(Integer, ForeignKey("sentences.id"), nullable=False)
    finished = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    ended_at = Column(DateTime, nullable=True)
