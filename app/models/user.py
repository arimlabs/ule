from sqlalchemy import Column, Integer, String, DateTime, func
from app.db.init_db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String(255), primary_key=True, index=True)
    name = Column(String(100), nullable=True)  # Optional name for acknowledgments
    created_at = Column(DateTime, default=func.now())