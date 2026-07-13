from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

MASK_CANVAS = components.declare_component(
    "mask_canvas",
    path=str(Path(__file__).resolve().parent / "app" / "frontend" / "mask_canvas"),
)


DEFAULT_API_URL = "http://localhost:8000"
SUPPORTED_CONDITIONS = [
    "Healthy_Normal_Skin",
    "Pressure_Injury_(PI)",
    "Diabetic_Foot_Ulcers_(DFU)",
    "Venous_Leg_Ulcers_(VLU)",
    "Atopic_Dermatitis_(Eczema)",
    "Contact_Dermatitis",
    "Seborrheic_Dermatitis",
    "Acne",
    "Hidradenitis_Suppurativa_(HS)",
    "Sunburn",
    "Psoriasis",
    "Bruising-Contusions",
    "Melasma",
    "Cold_Injury",
    "Rosacea",
    "Surgical_Wounds",
    "Radiation_Injury",
    "Burn_Wounds",
]


st.set_page_config(
    page_title="Wound Analysis Pipeline",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    defaults = {
        "case_id": None,
        "validation": None,
        "condition": None,
        "mask_result": None,
        "depth_result": None,
        "severity": None,
        "feedback": None,
        "selected_condition": None,
        "condition_source": None,
        "mask_decision": "accepted",
        "depth_decision": "accepted",
        "custom_mask": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.15rem;
        }
        .status-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
            color: #111827;
        }
        .subtle {
            color: #6b7280;
            font-size: 0.92rem;
        }
        .result-box {
            border-left: 4px solid #2563eb;
            background: #f8fafc;
            color: #111827;
            padding: 12px 14px;
            border-radius: 6px;
        }
        .danger-box {
            border-left: 4px solid #dc2626;
            background: #fef2f2;
            color: #7f1d1d;
            padding: 12px 14px;
            border-radius: 6px;
        }
        .ok-box {
            border-left: 4px solid #16a34a;
            background: #f0fdf4;
            color: #14532d;
            padding: 12px 14px;
            border-radius: 6px;
        }
        .route-box {
            border-left: 4px solid #0f766e;
            background: #f0fdfa;
            color: #134e4a;
            padding: 12px 14px;
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API_URL).rstrip("/")


def request_files(uploaded_file, field_name: str = "file") -> dict[str, tuple[str, bytes, str]]:
    uploaded_file.seek(0)
    return {
        field_name: (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "image/jpeg",
        )
    }


def post_file(endpoint: str, uploaded_file, data: dict[str, Any] | None = None) -> dict:
    response = requests.post(
        f"{api_url()}{endpoint}",
        files=request_files(uploaded_file),
        data=data or {},
        timeout=180,
    )
    return parse_response(response)


def parse_response(response: requests.Response) -> dict:
    try:
        payload = response.json()
    except Exception:
        payload = {"detail": response.text}

    if response.status_code >= 400:
        detail = payload.get("detail", payload)
        raise RuntimeError(str(detail))

    return payload


def preview_from_payload(preview: dict | None) -> Image.Image | None:
    if not preview:
        return None
    raw = base64.b64decode(preview["data"])
    return Image.open(BytesIO(raw)).convert("RGB")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_data_url(image: Image.Image) -> str:
    encoded = base64.b64encode(image_to_png_bytes(image)).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def mask_from_data_url(data_url: str | None, size: tuple[int, int]) -> Image.Image | None:
    if not data_url:
        return None

    try:
        _, encoded = data_url.split(",", 1)
        image = Image.open(BytesIO(base64.b64decode(encoded))).convert("L")
    except Exception:
        return None

    if image.size != size:
        image = image.resize(size, resample=Image.NEAREST)

    return image


def uploaded_image(uploaded_file) -> Image.Image | None:
    if uploaded_file is None:
        return None
    uploaded_file.seek(0)
    return Image.open(uploaded_file).convert("RGB")


def scaled_image(image: Image.Image, max_width: int = 720) -> tuple[Image.Image, float]:
    if image.width <= max_width:
        return image, 1.0
    scale = max_width / image.width
    resized = image.resize((max_width, int(image.height * scale)))
    return resized, scale


def show_json(label: str, payload: dict | None) -> None:
    with st.expander(label, expanded=False):
        st.json(payload or {})


def render_header() -> None:
    left, right = st.columns([0.72, 0.28], vertical_alignment="center")
    with left:
        st.title("Wound Analysis Pipeline")
        st.caption("Human-in-the-loop condition classification, mask review, depth review, and severity routing.")
    with right:
        st.session_state["api_url"] = st.text_input(
            "FastAPI server",
            value=st.session_state.get("api_url", DEFAULT_API_URL),
            help="Run the backend with: uvicorn app.main:app --host 0.0.0.0 --port 8000",
        )


def render_upload_and_validation(uploaded_file) -> None:
    st.subheader("1. Image")
    if uploaded_file is None:
        st.info("Upload a wound image to begin.")
        return

    image = uploaded_image(uploaded_file)
    col_a, col_b = st.columns([0.42, 0.58])
    with col_a:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col_b:
        if st.button("Validate image", use_container_width=True):
            with st.spinner("Validating image..."):
                st.session_state.validation = post_file("/validate-image", uploaded_file)

        if st.session_state.validation:
            validation = st.session_state.validation
            box_class = "ok-box" if validation.get("passed") else "danger-box"
            st.markdown(
                f"""
                <div class="{box_class}">
                    <strong>Validation:</strong> {"Passed" if validation.get("passed") else "Needs review"}<br>
                    <span class="subtle">Blur score: {validation.get("blur_score", "n/a")}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for warning in validation.get("warnings", []):
                st.warning(warning)
            show_json("Validation JSON", validation)


def render_condition(uploaded_file) -> None:
    st.subheader("2. Condition")
    if uploaded_file is None:
        return

    if st.button("Predict condition", type="primary", use_container_width=True):
        with st.spinner("Running condition classifier..."):
            st.session_state.condition = post_file("/predict-condition", uploaded_file)
            st.session_state.case_id = st.session_state.condition.get("case_id")
            st.session_state.selected_condition = st.session_state.condition.get("top1_label")
            st.session_state.condition_source = "model_accepted"

    condition = st.session_state.condition
    if not condition:
        st.caption("Run prediction to enable condition confirmation or override.")
        return

    st.markdown(
        f"""
        <div class="result-box">
            <strong>Top condition:</strong> {condition.get("top1_label")}<br>
            <span class="subtle">Confidence: {condition.get("confidence", 0):.4f} · Level: {condition.get("confidence_level")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for item in condition.get("top3", []):
        st.progress(float(item["probability"]), text=f"{item['rank']}. {item['class_name']} ({item['probability']:.3f})")

    st.divider()
    accept_model = st.toggle("Accept model top condition", value=True)
    if accept_model:
        st.session_state.selected_condition = condition.get("top1_label")
        st.session_state.condition_source = "model_accepted"
        st.success(f"Using model condition: {st.session_state.selected_condition}")
    else:
        selected = st.selectbox(
            "Override condition",
            SUPPORTED_CONDITIONS,
            index=SUPPORTED_CONDITIONS.index(st.session_state.selected_condition)
            if st.session_state.selected_condition in SUPPORTED_CONDITIONS
            else 0,
        )
        st.session_state.selected_condition = selected
        st.session_state.condition_source = "user_override"

    case_id = st.session_state.case_id or condition.get("case_id")
    st.caption(f"Case ID: {case_id}")
    show_json("Condition JSON", condition)


def render_mask(uploaded_file) -> Image.Image | None:
    st.subheader("3. Mask Review")
    if uploaded_file is None or not st.session_state.condition:
        st.caption("Predict and confirm a condition before mask review.")
        return None

    col_a, col_b = st.columns([0.42, 0.58])
    with col_a:
        if st.button("Generate mask", use_container_width=True):
            with st.spinner("Generating mask preview..."):
                st.session_state.mask_result = post_file(
                    "/generate-mask",
                    uploaded_file,
                    data={"case_id": st.session_state.case_id or ""},
                )

        mask_preview = preview_from_payload((st.session_state.mask_result or {}).get("mask_preview"))
        mask_overlay = preview_from_payload((st.session_state.mask_result or {}).get("mask_overlay_preview"))
        if mask_overlay:
            st.image(mask_overlay, caption="Blue mask overlay", use_container_width=True)
        elif mask_preview:
            st.warning("Overlay preview unavailable. Restart the backend, reset UI state, and generate the mask again.")
        elif st.session_state.mask_result:
            st.warning("No mask preview returned.")

    with col_b:
        st.session_state.mask_decision = st.radio(
            "Mask decision",
            ["accepted", "custom", "none", "aborted"],
            horizontal=True,
            index=["accepted", "custom", "none", "aborted"].index(st.session_state.mask_decision),
        )

        custom_mask = None
        if st.session_state.mask_decision == "custom":
            custom_mask = render_mask_canvas(uploaded_file)
        elif st.session_state.mask_decision == "none":
            st.info("Severity routing will ignore mask and use an RGB or RGB+depth pathway.")
        elif st.session_state.mask_decision == "aborted":
            st.error("Analysis will stop at mask review.")

        show_json("Mask JSON", st.session_state.mask_result)
        return custom_mask

    return None


def render_mask_canvas(uploaded_file) -> Image.Image | None:
    image = uploaded_image(uploaded_file)
    if image is None:
        return None

    display_image, _ = scaled_image(image)

    st.caption("Brush paints the blue mask layer. Eraser removes the mask layer live without changing the image.")

    data_url = MASK_CANVAS(
        image_data_url=image_to_data_url(display_image),
        width=display_image.width,
        height=display_image.height,
        key="mask_canvas",
    )

    mask = mask_from_data_url(data_url, image.size)
    if mask is not None and np.asarray(mask).max() > 0:
        st.session_state.custom_mask = mask
        with st.expander("Custom mask preview", expanded=False):
            st.image(mask, caption="Exported custom mask", use_container_width=True)
        return mask

    if st.session_state.custom_mask is not None:
        return st.session_state.custom_mask

    st.caption("Draw directly on the image to create a custom mask.")
    return None


def render_depth(uploaded_file) -> None:
    st.subheader("4. Depth Review")
    if uploaded_file is None or not st.session_state.condition:
        st.caption("Predict condition first.")
        return

    col_a, col_b = st.columns([0.42, 0.58])
    with col_a:
        if st.button("Generate depth map", use_container_width=True):
            with st.spinner("Generating depth preview..."):
                st.session_state.depth_result = post_file(
                    "/generate-depth",
                    uploaded_file,
                    data={"case_id": st.session_state.case_id or ""},
                )
        depth_preview = preview_from_payload((st.session_state.depth_result or {}).get("depth_preview"))
        if depth_preview:
            st.image(depth_preview, caption="Colorized full-image depth canvas", use_container_width=True)

    with col_b:
        st.session_state.depth_decision = st.radio(
            "Depth decision",
            ["accepted", "rejected", "aborted"],
            horizontal=True,
            index=["accepted", "rejected", "aborted"].index(st.session_state.depth_decision),
        )
        depth_result = st.session_state.depth_result or {}
        roi = depth_result.get("roi")
        st.caption("Depth is an experimental auxiliary input, not clinically validated.")
        if depth_result.get("preview_note"):
            st.info(depth_result["preview_note"])
        if roi:
            st.markdown(
                f"""
                <div class="status-card">
                    <strong>Depth crop ROI:</strong> {roi.get("box")}<br>
                    <span class="subtle">Raw wound box: {roi.get("raw_box")} · Mask fraction: {roi.get("mask_fraction")} · Used full image: {roi.get("used_full_image")}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        show_json("Depth JSON", st.session_state.depth_result)


def render_severity(uploaded_file, custom_mask: Image.Image | None) -> None:
    st.subheader("5. Severity Routing")
    if uploaded_file is None or not st.session_state.condition:
        st.caption("Complete the condition step first.")
        return

    selected_condition = st.session_state.selected_condition
    condition_source = st.session_state.condition_source
    mask_status = st.session_state.mask_decision
    depth_status = st.session_state.depth_decision

    st.markdown(
        f"""
        <div class="route-box">
            <strong>Selected condition:</strong> {selected_condition}<br>
            <strong>Condition source:</strong> {condition_source}<br>
            <strong>Mask:</strong> {mask_status} · <strong>Depth:</strong> {depth_status}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Predict severity", type="primary", use_container_width=True):
        if mask_status == "custom" and custom_mask is None:
            st.error("Draw a custom mask before predicting severity, or choose accepted/none.")
            return

        files = request_files(uploaded_file)
        if mask_status == "custom" and custom_mask is not None:
            files["mask_file"] = ("custom_mask.png", image_to_png_bytes(custom_mask), "image/png")

        data = {
            "selected_condition": selected_condition,
            "condition_source": condition_source,
            "mask_status": mask_status,
            "depth_status": depth_status,
        }

        with st.spinner("Routing and running severity model..."):
            response = requests.post(
                f"{api_url()}/predict-severity",
                files=files,
                data=data,
                timeout=180,
            )
            st.session_state.severity = parse_response(response)

    severity = st.session_state.severity
    if severity:
        if severity.get("severity_available"):
            st.markdown(
                f"""
                <div class="ok-box">
                    <strong>Severity:</strong> {severity.get("severity_prediction")}<br>
                    <span class="subtle">Confidence: {severity.get("severity_confidence", 0):.4f} · Model: {severity.get("model_used")}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(severity.get("routing_explanation", ""))
            probs = severity.get("class_probabilities") or {}
            for label, probability in probs.items():
                st.progress(float(probability), text=f"{label}: {probability:.3f}")
            if severity.get("warning"):
                st.warning(severity["warning"])
            if severity.get("mock_output"):
                st.info("This severity response is a mock fallback until real severity checkpoints are copied into the deployment package.")
        else:
            st.warning(severity.get("message", "Severity is unavailable for this condition."))
        show_json("Severity JSON", severity)


def render_feedback() -> None:
    st.subheader("6. Feedback")
    if not st.session_state.condition:
        st.caption("Feedback becomes available after prediction.")
        return

    corrected = st.text_input("Corrected severity, if any")
    comment = st.text_area("Comment", height=90)

    if st.button("Submit feedback", use_container_width=True):
        payload = {
            "case_id": st.session_state.case_id or "unknown",
            "model_condition": (st.session_state.condition or {}).get("top1_label"),
            "user_condition": st.session_state.selected_condition,
            "mask_choice": st.session_state.mask_decision,
            "depth_choice": st.session_state.depth_decision,
            "severity_prediction": (st.session_state.severity or {}).get("severity_prediction"),
            "user_corrected_severity": corrected or None,
            "comment": comment or None,
        }
        response = requests.post(f"{api_url()}/feedback", json=payload, timeout=30)
        st.session_state.feedback = parse_response(response)

    if st.session_state.feedback:
        st.success("Feedback saved.")
        show_json("Feedback JSON", st.session_state.feedback)


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Run locally")
        st.code(
            'uvicorn app.main:app --host 0.0.0.0 --port 8000\n'
            'streamlit run streamlit_app.py',
            language="bash",
        )
        st.divider()
        if st.button("Reset UI state", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != "api_url":
                    del st.session_state[key]
            st.rerun()
        st.caption("The UI keeps workflow state in Streamlit session state. The backend remains mostly stateless.")


def main() -> None:
    init_state()
    css()
    render_sidebar()
    render_header()

    uploaded_file = st.file_uploader(
        "Upload wound image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    top_left, top_right = st.columns([0.52, 0.48], gap="large")
    with top_left:
        render_upload_and_validation(uploaded_file)
        render_condition(uploaded_file)
    with top_right:
        custom_mask = render_mask(uploaded_file)
        render_depth(uploaded_file)

    st.divider()
    bottom_left, bottom_right = st.columns([0.58, 0.42], gap="large")
    with bottom_left:
        render_severity(uploaded_file, custom_mask)
    with bottom_right:
        render_feedback()


if __name__ == "__main__":
    main()
