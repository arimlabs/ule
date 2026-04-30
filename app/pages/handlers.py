from fastapi import Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import FileResponse

from app.db.init_db import get_db
from app.db.init_redis import validate_user_id
from app.models.experiment_results import ExperimentResult
from conf import (
    GIVEAWAY_THRESHOLD,
    TOTAL_TARGET,
    COOLDOWN_MS_CORRECT,
    COOLDOWN_MS_INCORRECT,
    KEYBOARD_REVEAL_THRESHOLD
)

templates = Jinja2Templates(directory="app/templates")


async def main_page(request: Request, db: AsyncSession = Depends(get_db)):
    # Count total completed experiments across all users
    result = await db.execute(
        select(func.count(ExperimentResult.id))
        .where(ExperimentResult.finished == True)
    )
    total_completed = result.scalar() or 0
    progress_percentage = (total_completed / TOTAL_TARGET * 100) if TOTAL_TARGET > 0 else 0

    return templates.TemplateResponse("landing.html", {
        "request": request,
        "total_completed": total_completed,
        "total_target": TOTAL_TARGET,
        "progress_percentage": progress_percentage
    })


async def introduction_page(request: Request, db: AsyncSession = Depends(get_db)):
    user_id = request.query_params.get("user_id")

    # Validate user_id exists in Redis
    is_valid = await validate_user_id(user_id)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid user ID. Please use the link provided to you.")

    # Count user's completed experiments
    user_result = await db.execute(
        select(func.count(ExperimentResult.id))
        .where(ExperimentResult.user_id == user_id)
        .where(ExperimentResult.finished == True)
    )
    user_completed = user_result.scalar() or 0

    # Count total completed experiments across all users
    total_result = await db.execute(
        select(func.count(ExperimentResult.id))
        .where(ExperimentResult.finished == True)
    )
    total_completed = total_result.scalar() or 0
    progress_percentage = (total_completed / TOTAL_TARGET * 100) if TOTAL_TARGET > 0 else 0

    return templates.TemplateResponse("introduction.html", {
        "request": request,
        "user_id": user_id,
        "giveaway_threshold": GIVEAWAY_THRESHOLD,
        "user_completed": user_completed,
        "total_completed": total_completed,
        "total_target": TOTAL_TARGET,
        "progress_percentage": progress_percentage
    })


async def experiment_page(request: Request, db: AsyncSession = Depends(get_db)):
    user_id = request.query_params.get("user_id")

    # Validate user_id exists in Redis
    is_valid = await validate_user_id(user_id)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid user ID. Please use the link provided to you.")

    # Count total completed experiments across all users
    total_result = await db.execute(
        select(func.count(ExperimentResult.id))
        .where(ExperimentResult.finished == True)
    )
    total_completed = total_result.scalar() or 0
    progress_percentage = (total_completed / TOTAL_TARGET * 100) if TOTAL_TARGET > 0 else 0

    return templates.TemplateResponse("experiment.html", {
        "request": request,
        "user_id": user_id,
        "cooldown_ms_correct": COOLDOWN_MS_CORRECT,
        "cooldown_ms_incorrect": COOLDOWN_MS_INCORRECT,
        "keyboard_reveal_threshold": KEYBOARD_REVEAL_THRESHOLD,
        "total_completed": total_completed,
        "total_target": TOTAL_TARGET,
        "progress_percentage": progress_percentage
    })


async def thank_you_page(request: Request, db: AsyncSession = Depends(get_db)):
    user_id = request.query_params.get("user_id")

    # Validate user_id exists in Redis
    is_valid = await validate_user_id(user_id)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid user ID. Please use the link provided to you.")

    # Count completed experiments for this user
    user_result = await db.execute(
        select(func.count(ExperimentResult.id))
        .where(ExperimentResult.user_id == user_id)
        .where(ExperimentResult.finished == True)
    )
    completed_count = user_result.scalar() or 0

    # Count total completed experiments across all users
    total_result = await db.execute(
        select(func.count(ExperimentResult.id))
        .where(ExperimentResult.finished == True)
    )
    total_completed = total_result.scalar() or 0
    progress_percentage = (total_completed / TOTAL_TARGET * 100) if TOTAL_TARGET > 0 else 0

    eligible_for_giveaway = completed_count >= GIVEAWAY_THRESHOLD

    return templates.TemplateResponse("thank_you.html", {
        "request": request,
        "user_id": user_id,
        "completed_count": completed_count,
        "giveaway_threshold": GIVEAWAY_THRESHOLD,
        "eligible_for_giveaway": eligible_for_giveaway,
        "total_completed": total_completed,
        "total_target": TOTAL_TARGET,
        "progress_percentage": progress_percentage
    })