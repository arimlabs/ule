from fastapi import APIRouter

from app.api.v1 import handlers

router = APIRouter(prefix="/api/v1")


router.add_api_route("/create_user", handlers.create_user, methods=["POST"], name="create_user")
router.add_api_route("/request_sentence", handlers.request_sentence, methods=["POST"], name="request_sentence")
router.add_api_route("/submit_guess", handlers.submit_guess, methods=["POST"], name="submit_guess")