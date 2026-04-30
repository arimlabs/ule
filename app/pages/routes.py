from fastapi import APIRouter

from app.pages import handlers

router = APIRouter()

# Pages
router.add_api_route("/", handlers.main_page, methods=["GET"], name="main_page") # Homepage
router.add_api_route("/introduction", handlers.introduction_page, methods=["GET"], name="introduction")
router.add_api_route("/experiment", handlers.experiment_page, methods=["GET"], name="experiment")
router.add_api_route("/thank-you", handlers.thank_you_page, methods=["GET"], name="thank_you")