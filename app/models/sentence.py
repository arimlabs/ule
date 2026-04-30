from sqlalchemy import Column, Integer, String, Enum, DateTime, func
from app.db.init_db import Base
from app.models.experiment_results import DatasetType


class Sentence(Base):
    __tablename__ = "sentences"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    dataset_type = Column(Enum(DatasetType, native_enum=False, length=50), nullable=False)
    source = Column(String(255), nullable=True)
    length = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now())