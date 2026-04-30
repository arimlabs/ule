from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel
import random

from app.db.init_db import get_db
from app.db.init_redis import validate_user_id
from app.models.user import User
from app.models.sentence import Sentence
from app.models.experiment_results import ExperimentResult
from app.models.character_guess import CharacterGuess
from conf import VALID_CHARACTERS, INITIAL_REVEAL_COUNT


class CreateUserRequest(BaseModel):
    user_id: str
    name: str | None = None


async def create_user(request: CreateUserRequest, db: AsyncSession = Depends(get_db)):
    """Create or update user with optional name for acknowledgments"""

    # Validate user_id exists in Redis
    is_valid = await validate_user_id(request.user_id)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid user ID. Please use the link provided to you.")

    # Check if user already exists
    existing_user = await db.get(User, request.user_id)

    if existing_user:
        # User already exists, optionally update name if provided
        if request.name:
            existing_user.name = request.name
            await db.commit()
        return {"message": "User already exists", "user_id": request.user_id}

    # Create new user
    new_user = User(
        id=request.user_id,
        name=request.name
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return {"message": "User created successfully", "user_id": new_user.id}


class RequestSentenceRequest(BaseModel):
    user_id: str


async def request_sentence(request: RequestSentenceRequest, db: AsyncSession = Depends(get_db)):
    """Request a new sentence for the user to guess"""

    # Validate user_id exists in Redis
    is_valid = await validate_user_id(request.user_id)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid user ID. Please use the link provided to you.")

    # Check if user exists in database
    user = await db.get(User, request.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found. Please complete the introduction first.")

    # Get sentences the user has already attempted
    result = await db.execute(
        select(ExperimentResult.sentence_id).where(ExperimentResult.user_id == request.user_id)
    )
    attempted_sentence_ids = [row[0] for row in result.fetchall()]

    # Get a random sentence that the user hasn't attempted yet
    query = select(Sentence)
    if attempted_sentence_ids:
        query = query.where(Sentence.id.not_in(attempted_sentence_ids))

    result = await db.execute(query)
    available_sentences = result.scalars().all()

    if not available_sentences:
        raise HTTPException(status_code=404, detail="No more sentences available. You've completed all available sentences!")

    # Select a random sentence
    sentence = random.choice(available_sentences)

    # Create a new experiment result record
    experiment = ExperimentResult(
        user_id=request.user_id,
        sentence_id=sentence.id,
        finished=False
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)

    # Return sentence metadata with initial revealed characters
    initial_text = sentence.text[:INITIAL_REVEAL_COUNT]
    # Send the correct letter for the first position they'll guess
    next_correct_letter = sentence.text[INITIAL_REVEAL_COUNT] if INITIAL_REVEAL_COUNT < sentence.length else None
    return {
        "experiment_id": experiment.id,
        "sentence_id": sentence.id,
        "sentence_length": sentence.length,
        "max_guesses": sentence.length - INITIAL_REVEAL_COUNT,  # Shannon's rule: max guesses = sentence length
        "initial_text": initial_text,
        "start_position": INITIAL_REVEAL_COUNT,
        "next_correct_letter": next_correct_letter 
    }


class SubmitGuessRequest(BaseModel):
    user_id: str
    experiment_id: int
    position: int
    guessed_character: str


async def submit_guess(request: SubmitGuessRequest, db: AsyncSession = Depends(get_db)):
    is_valid = await validate_user_id(request.user_id)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid user ID. Please use the link provided to you.")

    if len(request.guessed_character) != 1:
        raise HTTPException(status_code=400, detail="Guessed character must be exactly one character")

    if request.guessed_character.upper() not in VALID_CHARACTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid character. Must be a Ukrainian letter or space. Valid characters: {VALID_CHARACTERS}"
        )

    experiment = await db.get(ExperimentResult, request.experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if experiment.user_id != request.user_id:
        raise HTTPException(status_code=403, detail="This experiment does not belong to you")

    if experiment.finished:
        raise HTTPException(status_code=400, detail="This experiment is already finished")

    sentence = await db.get(Sentence, experiment.sentence_id)
    if not sentence:
        raise HTTPException(status_code=404, detail="Sentence not found")

    if request.position < INITIAL_REVEAL_COUNT or request.position >= sentence.length:
        raise HTTPException(status_code=400, detail=f"Invalid position. Must be between {INITIAL_REVEAL_COUNT} and {sentence.length - 1}")

    result = await db.execute(
        select(CharacterGuess.guessed_character).where(
            CharacterGuess.experiment_result_id == request.experiment_id,
            CharacterGuess.position == request.position
        )
    )
    previous_guesses = [row[0] for row in result.fetchall()]

    if request.guessed_character.upper() in previous_guesses:
        raise HTTPException(status_code=400, detail="Ви вже вгадували цю літеру на цій позиції")

    actual_character = sentence.text[request.position]
    is_correct = request.guessed_character.upper() == actual_character.upper()

    result = await db.execute(
        select(func.count(CharacterGuess.id)).where(
            CharacterGuess.experiment_result_id == request.experiment_id
        )
    )
    total_guesses = result.scalar() or 0

    result = await db.execute(
        select(func.count(CharacterGuess.id)).where(
            CharacterGuess.experiment_result_id == request.experiment_id,
            CharacterGuess.position == request.position
        )
    )
    position_guesses = result.scalar() or 0

    guess = CharacterGuess(
        experiment_result_id=request.experiment_id,
        position=request.position,
        guess_number=position_guesses + 1,
        guessed_character=request.guessed_character.upper(),
        is_correct=is_correct
    )
    db.add(guess)

    max_guesses = sentence.length - INITIAL_REVEAL_COUNT
    guesses_remaining = max_guesses - (total_guesses + 1)
    experiment_complete = is_correct and request.position == sentence.length - 1
    guesses_exhausted = guesses_remaining <= 0 and not experiment_complete

    # CRITICAL FIX: Add database validation before marking as finished
    if experiment_complete or guesses_exhausted:
        # Verify that we actually have guesses in the database
        # This prevents marking experiments as finished due to race conditions
        actual_guesses_count = await db.execute(
            select(func.count(CharacterGuess.id)).where(
                CharacterGuess.experiment_result_id == request.experiment_id
            )
        )
        total_actual_guesses = actual_guesses_count.scalar() or 0

        # Only mark as finished if we have at least some guesses recorded
        # The +1 accounts for the current guess that was just added above
        if total_actual_guesses > 0:
            # Double-check the experiment isn't already finished (race condition protection)
            await db.refresh(experiment)
            if not experiment.finished:
                experiment.finished = True
                experiment.ended_at = func.now()
        else:
            # This should never happen, but if it does, log it and don't mark as finished
            print(f"WARNING: Attempted to finish experiment {request.experiment_id} with 0 guesses in database")

    await db.commit()

    # Send the next correct letter if they guessed correctly and there's more to guess
    next_correct_letter = None
    if is_correct and request.position + 1 < sentence.length:
        next_correct_letter = sentence.text[request.position + 1]

    return {
        "is_correct": is_correct,
        "actual_character": actual_character if is_correct else None,
        "position": request.position,
        "total_guesses_used": total_guesses + 1,
        "guesses_remaining": guesses_remaining,
        "experiment_complete": experiment_complete,
        "guesses_exhausted": guesses_exhausted,
        "full_sentence": sentence.text if (experiment_complete or guesses_exhausted) else None,
        "next_correct_letter": next_correct_letter
    }