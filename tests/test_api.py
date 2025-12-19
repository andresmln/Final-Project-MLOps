"""
Integration testing with the API
"""
import io
import os
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from api.api import app

# --- Fixtures ---

@pytest.fixture
def client():
    """Testing client from FastAPI."""
    return TestClient(app)

@pytest.fixture
def dummy_image_bytes():
    """
    Creates a simple 100x100 RGB red image in memory.
    Returns the bytes, ready to be sent as a file.
    """
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)  # Reset pointer to the beginning
    return img_byte_arr

# --- Tests ---

def test_home_endpoint(client):
    """Verify that the home endpoint returns 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_predict_endpoint(client, dummy_image_bytes):
    """
    Verify that /predict returns a prediction.
    Handles logic for both local (model exists) and CI (model might be missing).
    """
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    
    response = client.post("/predict", files=files)
    
    # Check if artifacts exist locally to determine expected behavior
    artifacts_exist = os.path.exists("model.onnx") and os.path.exists("class_labels.json")
    
    if artifacts_exist:
        # If model exists, we expect a successful prediction
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], str)
    else:
        # If model is missing, API should return 500 (as coded in api.py)
        # This prevents the test from failing in clean environments
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

def test_resize_endpoint(client, dummy_image_bytes):
    """Verify that /resize returns an image with the correct dimensions."""
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    data = {"width": "50", "height": "50"}
    
    response = client.post("/resize", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    # Verify the output image size
    resized_image = Image.open(io.BytesIO(response.content))
    assert resized_image.size == (50, 50)

def test_info_endpoint(client, dummy_image_bytes):
    """Verify that /info returns correct metadata."""
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    
    response = client.post("/info", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["width"] == 100
    assert data["height"] == 100
    assert data["format"] == "JPEG"

def test_grayscale_endpoint(client, dummy_image_bytes):
    """Verify that /grayscale returns a PNG image."""
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    
    response = client.post("/grayscale", files=files)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_rotate_endpoint(client, dummy_image_bytes):
    """Verify that /rotate works with an angle."""
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    data = {"angle": "90"}
    
    response = client.post("/rotate", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_blur_endpoint(client, dummy_image_bytes):
    """Verify that /blur works with a radius."""
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    data = {"radius": "5"}
    
    response = client.post("/blur", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

def test_normalize_endpoint(client, dummy_image_bytes):
    """Verify that /normalize returns a downloadable PNG image."""
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    
    response = client.post("/normalize", files=files)
    
    assert response.status_code == 200
    # UPDATED: We now expect an image, not JSON
    assert response.headers["content-type"] == "image/png"
    assert "attachment; filename=normalized_visualization.png" in response.headers["content-disposition"]

# --- Negative / Error Tests ---

def test_resize_missing_parameters(client, dummy_image_bytes):
    """Verify validation error when form data is missing."""
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    # Missing 'width' and 'height'
    
    response = client.post("/resize", files=files)
    
    # FastAPI returns 422 Unprocessable Entity for missing required form fields
    assert response.status_code == 422

def test_invalid_image_file(client):
    """Verify error handling for invalid image files."""
    files = {"file": ("test.txt", b"this is not an image", "text/plain")}
    
    response = client.post("/predict", files=files)
    
    # API catches exception and returns 400 or 500 depending on where it failed
    assert response.status_code in [400, 500]
