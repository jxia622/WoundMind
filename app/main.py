from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    ALLOWED_IMAGE_CONTENT_TYPES,
    API_TITLE,
    API_VERSION,
    DISCLAIMER,
    MODEL_VERSION,
)
from app.pipeline import WoundAnalysisPipeline
from app.preprocessing.image_io import load_upload_as_rgb
from app.preprocessing.validation import validate_image_basic
from app.routers.agent_router import router as agent_router
from app.schemas import FeedbackPayload
from app.utils.feedback_logger import append_feedback


app = FastAPI(
    title=API_TITLE,
    description="Human-in-the-loop wound diagnostic assistant research prototype",
    version=API_VERSION,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(agent_router)

pipeline: WoundAnalysisPipeline | None = None
startup_error: str | None = None


def _require_pipeline() -> WoundAnalysisPipeline:
    if pipeline is None:
        detail = "Pipeline is not loaded."
        if startup_error:
            detail = f"{detail} Startup error: {startup_error}"
        raise HTTPException(status_code=503, detail=detail)
    return pipeline


def _validate_upload_content_type(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WEBP.",
        )


async def _read_rgb_upload(file: UploadFile):
    _validate_upload_content_type(file)
    try:
        return await load_upload_as_rgb(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.on_event("startup")
def startup_event():
    """
    Load model wrappers once when the API server starts.
    """
    global pipeline, startup_error

    try:
        pipeline = WoundAnalysisPipeline()
        startup_error = None
        app.state.pipeline = pipeline
        app.state.startup_error = None
        print("Wound analysis pipeline loaded.")
    except Exception as e:
        pipeline = None
        startup_error = str(e)
        app.state.pipeline = None
        app.state.startup_error = startup_error
        print(f"Pipeline startup failed: {startup_error}")


@app.get("/")
def root():
    return {
        "message": "WoundMind diagnostic assistant API is running.",
        "model_version": MODEL_VERSION,
        "disclaimer": DISCLAIMER,
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/validate-image")
async def validate_image(file: UploadFile = File(...)):
    image = await _read_rgb_upload(file)
    return validate_image_basic(image).model_dump()


@app.post("/predict-condition")
async def predict_condition(file: UploadFile = File(...)):
    image = await _read_rgb_upload(file)
    service = _require_pipeline()
    return service.predict_condition(image)


@app.post("/confirm-condition")
def confirm_condition(
    case_id: str = Form(...),
    model_condition: str = Form(...),
    user_selected_condition: str | None = Form(None),
    abort: bool = Form(False),
):
    service = _require_pipeline()
    return service.confirm_condition(
        case_id=case_id,
        model_condition=model_condition,
        user_selected_condition=user_selected_condition,
        abort=abort,
    )


@app.post("/generate-mask")
async def generate_mask(file: UploadFile = File(...), case_id: str | None = Form(None)):
    image = await _read_rgb_upload(file)
    service = _require_pipeline()
    return service.generate_mask(image, case_id=case_id)


@app.post("/generate-depth")
async def generate_depth(file: UploadFile = File(...), case_id: str | None = Form(None)):
    image = await _read_rgb_upload(file)
    service = _require_pipeline()
    return service.generate_depth(image, case_id=case_id)


@app.post("/predict-severity")
async def predict_severity(
    file: UploadFile = File(...),
    selected_condition: str = Form(...),
    condition_source: str = Form(...),
    mask_status: str = Form(...),
    depth_status: str = Form(...),
    mask_file: UploadFile | None = File(None),
    depth_file: UploadFile | None = File(None),
):
    image = await _read_rgb_upload(file)
    service = _require_pipeline()

    mask = None
    if mask_file is not None and mask_status == "custom":
        mask_image = await _read_rgb_upload(mask_file)
        mask = mask_image.convert("L")
    elif mask_status == "custom":
        raise HTTPException(status_code=400, detail="mask_file is required when mask_status is 'custom'.")

    depth = None
    if depth_file is not None and depth_status == "accepted":
        depth_image = await _read_rgb_upload(depth_file)
        depth = depth_image.convert("L")

    try:
        return service.predict_severity(
            image=image,
            selected_condition=selected_condition,
            condition_source=condition_source,
            mask_status=mask_status,
            depth_status=depth_status,
            mask=mask,
            depth=depth,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), case_id: str | None = Form(None)):
    image = await _read_rgb_upload(file)
    service = _require_pipeline()
    return service.analyze_default(image, case_id=case_id)


@app.post("/feedback")
def feedback(payload: FeedbackPayload = Body(...)):
    result = append_feedback(payload.model_dump())
    return {
        **result,
        "case_id": payload.case_id,
    }
