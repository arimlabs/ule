import os

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase


class Base(DeclarativeBase): # Base class for db models
    pass


POSTGRESQL_URL = os.getenv("POSTGRESQL_URL")

engine = create_async_engine(POSTGRESQL_URL)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    async with async_session() as session:
        yield session

async def init_db_on_fastapi_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)