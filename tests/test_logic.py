"""
Unit Testing of the application's logic (Image Processor)
"""
import io
import os
import pytest
import numpy as np
from PIL import Image
from mylib.inference_image_processor import (
    predict_image, 
    resize_image, 
    convert_to_grayscale, 
    rotate_image,
    apply_blur, 
    normalize_image, 
    get_image_info
)

# --- Fixtures ---

@pytest.fixture
def dummy_image():
    """Creates a simple 224x224 RGB red image for testing."""
    img = Image.new("RGB", (224, 224), color="red")
    return img

@pytest.fixture
def dummy_image_file(dummy_image):
    """Creates a file-like object (buffer) containing the dummy image."""
    buffer = io.BytesIO()
    dummy_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer

# --- Tests ---

def test_resize_image(dummy_image_file):
    """Test that the image is resized to the correct dimensions."""
    width, height = 50, 50
    resized_img = resize_image(dummy_image_file, width, height)
    assert resized_img.size == (width, height)

def test_convert_to_grayscale(dummy_image):
    """Test that the image is converted to grayscale (Mode 'L')."""
    gray_img = convert_to_grayscale(dummy_image)
    assert gray_img.mode == "L"

def test_rotate_image(dummy_image):
    """Test that the image rotates without errors."""
    angle = 90
    rotated_img = rotate_image(dummy_image, angle)
    assert isinstance(rotated_img, Image.Image)

def test_apply_blur(dummy_image):
    """Test that the blur filter runs and returns an image."""
    blurred_img = apply_blur(dummy_image, radius=2)
    assert isinstance(blurred_img, Image.Image)

def test_normalize_image(dummy_image):
    """Test that normalization returns a numpy array with values between 0 and 1."""
    norm_array = normalize_image(dummy_image)
    assert isinstance(norm_array, np.ndarray)
    assert norm_array.min() >= 0.0
    assert norm_array.max() <= 1.0

def test_get_image_info(dummy_image):
    """Test that metadata extraction returns correct information."""
    info = get_image_info(dummy_image)
    assert isinstance(info, dict)
    assert info["width"] == 224
    assert info["height"] == 224
    assert info["is_color"] is True

# --- Prediction Test (Conditional) ---

def test_predict_image_behavior(dummy_image):
    """
    Test prediction logic.
    - If model exists: Checks if it returns a valid string (prediction).
    - If model missing: Checks if it returns the expected error message.
    """
    prediction = predict_image(dummy_image)
    
    # Check if artifacts exist locally to determine expected behavior
    artifacts_exist = os.path.exists("model.onnx") and os.path.exists("class_labels.json")
    
    if artifacts_exist:
        # If model is present, we expect a real class name (str)
        assert isinstance(prediction, str)
        assert not prediction.startswith("Error") 
    else:
        # If model is missing, we expect our handled error message
        assert isinstance(prediction, str)
        assert prediction.startswith("Error") or "not loaded" in prediction
