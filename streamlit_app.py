"""Run age and gender estimation through a Streamlit web interface."""

from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from age_and_gender import AgeAndGender, FacePrediction


@st.cache_resource
def load_predictor() -> AgeAndGender:
    """Create and cache the predictor for this Streamlit server process."""
    return AgeAndGender()


def annotate_image(
    image: Image.Image,
    predictions: list[FacePrediction],
) -> Image.Image:
    """Draw face boxes and prediction labels onto a copy of an image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for prediction in predictions:
        left, top, right, bottom = prediction["face"]
        age = prediction["age"]
        gender = prediction["gender"]
        label = f"Age: {age['value']} | Gender: {gender['value']}"

        draw.rectangle((left, top, right, bottom), outline="#36c98f", width=3)
        text_bounds = draw.textbbox((0, 0), label, font=font)
        label_width = text_bounds[2] - text_bounds[0]
        label_height = text_bounds[3] - text_bounds[1]
        label_top = max(0, top - label_height - 8)
        draw.rounded_rectangle(
            (left, label_top, left + label_width + 12, label_top + label_height + 8),
            radius=4,
            fill="#12352b",
        )
        draw.text((left + 6, label_top + 4), label, font=font, fill="white")

    return annotated


def read_uploaded_image(data: bytes) -> Image.Image:
    """Decode image bytes and normalize their orientation and color mode."""
    with Image.open(BytesIO(data)) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def main() -> None:
    """Render the Streamlit app and process the selected image."""
    st.set_page_config(
        page_title="Age & Gender Detector",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Age & Gender Detector")
    st.write("Estimate age and gender from a photo using the bundled models.")
    st.caption(
        "Predictions are estimates. Images are processed locally by this app and are not saved."
    )

    source = st.radio("Image source", ["Take a photo", "Upload an image"], horizontal=True)
    if source == "Take a photo":
        captured_image = st.camera_input("Capture a photo")
        selected_image = captured_image
    else:
        uploaded_image = st.file_uploader(
            "Choose an image",
            type=("jpg", "jpeg", "png", "webp", "bmp"),
        )
        selected_image = uploaded_image
    if selected_image is None:
        st.info("Take a photo or upload an image to get started.")
        return

    try:
        image = read_uploaded_image(selected_image.getvalue())
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"Could not read that image: {error}")
        return

    with st.spinner("Finding faces and estimating age and gender..."):
        try:
            predictions = load_predictor().predict(image)
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            return

    result_column, details_column = st.columns((3, 2), gap="large")
    with result_column:
        st.image(
            annotate_image(image, predictions),
            caption="Detected faces and predictions",
            use_container_width=True,
        )

    with details_column:
        st.subheader("Results")
        if not predictions:
            st.info("No faces were detected. Try a clearer, front-facing photo.")
            return

        st.write(f"Detected **{len(predictions)} face(s)**.")
        for index, prediction in enumerate(predictions, start=1):
            age = prediction["age"]
            gender = prediction["gender"]
            with st.container(border=True):
                st.markdown(f"**Face {index}**")
                age_column, gender_column = st.columns(2)
                age_column.metric(
                    "Estimated age",
                    f"{age['value']} years",
                    f"{age['confidence']}% confidence",
                )
                gender_column.metric(
                    "Estimated gender",
                    str(gender["value"]).title(),
                    f"{gender['confidence']}% confidence",
                )


if __name__ == "__main__":
    main()
