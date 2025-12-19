# Import the libraries, classes and functions
import uvicorn
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, StreamingResponse

# Import your image processing functions
from mylib.inference_image_processor import (
    predict_image,
    resize_image,
    convert_to_grayscale,
    rotate_image,
    apply_blur,
    normalize_image,
    get_image_info
)

# Create an instance of FastAPI
app = FastAPI(
    title="API for Image Processing",
    description="API to perform image processing operations using mylib.inference_image_processor",
    version="1.0.0",
)

# We use the templates folder to obtain HTML files
templates = Jinja2Templates(directory="templates")


# --- Utility function to load image ---
# This is a helper to avoid repeating code
async def load_image_from_uploadfile(file: UploadFile) -> Image.Image:
    """Reads an UploadFile and converts it to a PIL Image."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        return image
    except Exception as e:
        # Pylint Fix (W0707): Explicit exception chaining
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e


# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serves the home.html page."""
    return templates.TemplateResponse(request, "home.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predicts the class of an image using the trained MobileNetV2 model.
    Expects a 'multipart/form-data' with a 'file' field.
    """
    try:
        image = await load_image_from_uploadfile(file)
        prediction = predict_image(image)
        
        # Check if the prediction string indicates an internal error
        if prediction.startswith("Error") or prediction.startswith("Prediction Error"):
            raise HTTPException(status_code=500, detail=prediction)
            
        return {"prediction": prediction}
    except HTTPException as he:
        raise he # Re-raise HTTP exceptions directly
    except Exception as e:
        # Pylint Fix (W0707): Explicit exception chaining
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/resize", response_class=StreamingResponse)
async def resize(
    file: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...)
):
    """
    Resizes an image.
    Expects 'multipart/form-data' with 'file', 'width', and 'height' fields.
    Returns the resized image as a downloadable JPEG file.
    """
    try:
        # Our resize_image function takes a file-like object directly
        resized_img = resize_image(file.file, width, height)
        
        # Save resized image to a memory buffer
        buffer = io.BytesIO()
        resized_img.save(buffer, format="JPEG")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer, 
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=resized_image.jpg"}
        )
    except Exception as e:
        # Pylint Fix (W0707): Explicit exception chaining
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/info")
async def info_endpoint(file: UploadFile = File(...)): # Pylint Fix (W0621): Renamed function
    """
    Gets metadata from an image (size, mode, format, etc.).
    Expects 'multipart/form-data' with a 'file' field.
    """
    try:
        image = await load_image_from_uploadfile(file)
        image_info = get_image_info(image) # Pylint Fix (W0621): Renamed variable
        return image_info
    except Exception as e:
        # Pylint Fix (W0707): Explicit exception chaining
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/grayscale", response_class=StreamingResponse)
async def grayscale(file: UploadFile = File(...)):
    """
    Converts an image to grayscale.
    Expects 'multipart/form-data' with a 'file' field.
    Returns the grayscale image as a downloadable PNG file.
    """
    try:
        image = await load_image_from_uploadfile(file)
        gray_img = convert_to_grayscale(image)
        
        buffer = io.BytesIO()
        gray_img.save(buffer, format="PNG")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer, 
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=grayscale_image.png"}
        )
    except Exception as e:
        # Pylint Fix (W0707): Explicit exception chaining
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/rotate", response_class=StreamingResponse)
async def rotate(
    file: UploadFile = File(...),
    angle: float = Form(...)
):
    """
    Rotates an image by a given angle.
    Expects 'multipart/form-data' with 'file' and 'angle' fields.
    Returns the rotated image as a downloadable PNG file.
    """
    try:
        image = await load_image_from_uploadfile(file)
        rotated_img = rotate_image(image, angle)
        
        buffer = io.BytesIO()
        # Use PNG for rotate as it might create transparent areas
        rotated_img.save(buffer, format="PNG")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer, 
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=rotated_image.png"}
        )
    except Exception as e:
        # Pylint Fix (W0707): Explicit exception chaining
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/blur", response_class=StreamingResponse)
async def blur(
    file: UploadFile = File(...),
    radius: int = Form(default=2)
):
    """
    Applies a Gaussian blur to an image.
    Expects 'multipart/form-data' with 'file' and optional 'radius'.
    Returns the blurred image as a downloadable JPEG file.
    """
    try:
        image = await load_image_from_uploadfile(file)
        blurred_img = apply_blur(image, radius)
        
        buffer = io.BytesIO()
        blurred_img.save(buffer, format="JPEG")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer, 
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=blurred_image.jpg"}
        )
    except Exception as e:
        # Pylint Fix (W0707): Explicit exception chaining
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/normalize", response_class=StreamingResponse)
async def normalize(file: UploadFile = File(...)):
    """
    Normalizes an image and returns a visualization of the normalized array.
    Expects 'multipart/form-data' with a 'file' field.
    Returns: A downloadable PNG image representing the normalized data.
    """
    try:
        image = await load_image_from_uploadfile(file)
        
        # Get the normalized numpy array (values roughly 0-1 or standardized)
        norm_array = normalize_image(image)
        
        # To make it downloadable/viewable, we must convert the float array back to uint8 [0-255]
        # We perform a simple Min-Max scaling to map the values to 0-255 for visualization
        norm_min, norm_max = norm_array.min(), norm_array.max()
        norm_visual = (norm_array - norm_min) / (norm_max - norm_min) * 255.0
        norm_visual = norm_visual.astype(np.uint8)
        
        # Create a PIL image from the array
        visual_image = Image.fromarray(norm_visual)
        
        buffer = io.BytesIO()
        visual_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=normalized_visualization.png"}
        )
    except Exception as e:
        # Pylint Fix (W0707): Explicit exception chaining
        raise HTTPException(status_code=500, detail=str(e)) from e


# Entry point (for direct execution only)
if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
