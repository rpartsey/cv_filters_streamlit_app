import os
from io import BytesIO
import cv2
from PIL import Image
import numpy as np
import streamlit as st
from rembg import remove


def read_image(upload_file_or_path):
    """Read image from file path or file upload."""
    image = Image.open(upload_file_or_path)
    return image


def convert_image(img):
    """Convert image to bytes for download."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def remove_background(image):
    """Remove background from image using rembg library."""
    processed_image = remove(image)
    return processed_image


def to_grayscale(image):
    """Convert image to grayscale."""
    image_np = np.array(image)
    processed_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    image_pil = Image.fromarray(processed_image)

    return image_pil


def to_grayscale(image):
    """Convert image to grayscale."""
    image_np = np.array(image)
    processed_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    image_pil = Image.fromarray(processed_image)

    return image_pil


def laplacian(image):
    """Apply laplacian filter to image."""
    image_np = np.array(image)
    image_np_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    processed_image = np.absolute(cv2.Laplacian(image_np_gray, cv2.CV_64F)).astype("uint8")
    image_pil = Image.fromarray(processed_image)

    return image_pil


def canny_edge_detection(image):
    """Apply canny edge detection filter to image."""
    image_np = np.array(image)
    image_np_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    processed_image = cv2.Canny(image_np_gray, 100, 200)
    image_pil = Image.fromarray(processed_image)

    return image_pil


FILTERS = {
    "Remove Background": remove_background,
    "Grayscale": to_grayscale,
    "Laplacian Derivatives": laplacian,
    "Canny Edge Detection": canny_edge_detection,
}


def app():
    # Web app title
    st.set_page_config(layout="wide", page_title="Image Background Remover")
    
    # Web app description
    st.write("## Demo image filters App :camera:")
    st.write(
        "Upload an image and watch different filters applyed to it."
        "This code is open source and available [here](https://github.com/rpartsey/cv_filters_streamlit_app) on GitHub. "
        "Special thanks to the [rembg library](https://github.com/danielgatis/rembg) :grin: "
        "and [BackgroundRemoval exapmle](https://github.com/tyler-simons/BackgroundRemoval)"
    )

    # Web app sidebar widgets
    st.sidebar.write("## Upload :arrow_up:")
    st.sidebar.write("### Upload custom image from file path")

    # Web app text input
    text_input_file_path = st.sidebar.text_input("Image file path", "")  
    text_input_file_path_status = st.sidebar.empty()

    # Web app file upload
    st.sidebar.write("### Upload custom image from file upload")
    upload_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    # Web app selectbox
    st.sidebar.write("### Choose pre-uploaded image")
    selectbox_file_name = st.sidebar.selectbox("Pre-uploaded image", ["zebra.jpg", "wallaby.png"])
    selectbox_file_path = f"./data/images/{selectbox_file_name}"
    
    # Web app button
    st.sidebar.write("## Apply Filter :wrench:")
    selectbox_filter_name = st.sidebar.selectbox("Choose Image Filter", list(FILTERS.keys()))
    button_apply_filter = st.sidebar.button("Apply")

    # Read image
    if text_input_file_path:
        if not os.path.exists(text_input_file_path) or not os.path.isfile(text_input_file_path):
            text_input_file_path_status.error("File does not exist.")
        else:
            image = read_image(text_input_file_path)
            file_path = text_input_file_path
    elif upload_file is not None:
        image = read_image(upload_file)
        file_path = upload_file.name
    else:
        image = read_image(selectbox_file_path)
        file_path = selectbox_file_path

    # Create two-column layout
    col1, col2 = st.columns(2)

    # Visualize original image
    col1.write("Original Image :camera:")
    col1.image(image)

    # Create a placeholder for processed image
    col2.write("Processed Image :wrench:")
    processed_image_ocation = col2.empty()

    # If button_apply_filter is clicked, apply filter to image
    if button_apply_filter:
        # Choose filter to apply
        image_filter = FILTERS[selectbox_filter_name]
        # Apply filter to image
        processed_image = image_filter(image)
        # Visualize processed image
        processed_image_ocation.image(processed_image)

        # Download processed image
        st.sidebar.markdown("\n")
        st.sidebar.write("## Download :arrow_down:")
        st.sidebar.download_button(
            "Download processed image", 
            convert_image(processed_image), 
            f"{os.path.basename(file_path).split('.')[0]}_processed.png", 
            "image/png"
        )


if __name__ == "__main__":
    app()
