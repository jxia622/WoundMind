from __future__ import annotations

import asyncio
import json
from json import JSONDecodeError

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.agent import AgentConfig, WoundMindAgent
from app.config import ALLOWED_IMAGE_CONTENT_TYPES
from app.preprocessing.image_io import load_upload_as_rgb

router = APIRouter(prefix="/agent", tags=["agent"])


def _pipeline_from_request(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    startup_error = getattr(request.app.state, "startup_error", None)
    if pipeline is None:
        detail = "Pipeline is not loaded."
        if startup_error:
            detail = f"{detail} Startup error: {startup_error}"
        raise HTTPException(status_code=503, detail=detail)
    return pipeline


def _validate_upload_content_type(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WEBP.",
        )


def _parse_patient_context(raw_context: str | None) -> dict:
    if not raw_context:
        return {}
    try:
        parsed = json.loads(raw_context)
    except JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="patient_context must be valid JSON.") from e
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="patient_context must be a JSON object.")
    return parsed


@router.post("/analyze")
async def analyze_agent(
    request: Request,
    image: UploadFile = File(...),
    patient_context: str | None = Form(None),
    case_id: str | None = Form(None),
):
    _validate_upload_content_type(image)
    try:
        image_rgb = await load_upload_as_rgb(image)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    config = AgentConfig.from_env()
    context = _parse_patient_context(patient_context)
    pipeline = _pipeline_from_request(request)
    agent = WoundMindAgent(pipeline=pipeline, config=config)

    try:
        result = await asyncio.wait_for(
            agent.analyze(
                image=image_rgb,
                patient_context=context,
                case_id=case_id,
            ),
            timeout=config.timeout_seconds,
        )
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail="Agent analysis timed out.") from e

    return result.model_dump()
