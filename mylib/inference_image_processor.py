"""
Image Processing Library with ONNX Inference
"""
import os
import json
import numpy as np
from PIL import Image, ImageFilter
import onnxruntime as ort

# Global variables to cache the model and labels (Singleton pattern)
_ONNX_SESSION = None
_CLASS_LABELS = None

# Constants for preprocessing (Must match training!)
IMAGE_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def _load_model_artifacts():
    """
    Internal function to load the ONNX model and class labels.
    """
    # pylint: disable=global-statement
    global _ONNX_SESSION, _CLASS_LABELS

    if _ONNX_SESSION is None:
        model_path = "model.onnx"
        labels_path = "class_labels.json"

        if not os.path.exists(model_path) or not os.path.exists(labels_path):
            print(f"Warning: Artifacts not found at {model_path} or {labels_path}.")
            return

        # Requirement: Get the class labels (read the JSON file)
        # Fixed W1514: Added encoding="utf-8"
        with open(labels_path, "r", encoding="utf-8") as f:
            _CLASS_LABELS = json.load(f)

        # Requirement: Assign the sess_options to an instance of SessionOptions()
        sess_options = ort.SessionOptions()
        # Requirement: Setting the intra_op_num_threads to 4
        sess_options.intra_op_num_threads = 4

        # Requirement: Instantiate the InferenceSession class
        # Requirement: providers=["CPUExecutionProvider"]
        _ONNX_SESSION = ort.InferenceSession(
            model_path, sess_options, providers=["CPUExecutionProvider"]
        )
        print("ONNX Model and Labels loaded successfully.")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Requirement: Define a function/method to preprocess the data...
    accommodate it to the format used to train (RGB, size, norm, batch dim)
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize and Crop (Matching training transforms)
    img = image.resize((256, 256))
    left = (256 - IMAGE_SIZE) / 2
    top = (256 - IMAGE_SIZE) / 2
    right = (256 + IMAGE_SIZE) / 2
    bottom = (256 + IMAGE_SIZE) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to Numpy and Scale to [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0

    # Normalize
    mean = np.array(NORM_MEAN, dtype=np.float32)
    std = np.array(NORM_STD, dtype=np.float32)
    img_np = (img_np - mean) / std

    # Transpose to (Channels, Height, Width)
    img_np = img_np.transpose((2, 0, 1))

    # Requirement: expansion to include the batch dimension
    img_np = np.expand_dims(img_np, axis=0)

    return img_np


def predict_image(image: Image.Image) -> str:
    """
    Requirement: Define a function/method to predict the class label
    """
    # Load model if not loaded
    if _ONNX_SESSION is None:
        _load_model_artifacts()

    if _ONNX_SESSION is None:
        return "Error: Model not loaded."

    try:
        # Preprocess
        input_tensor = preprocess_image(image)

        # Requirement: Obtain the session name
        input_name = _ONNX_SESSION.get_inputs()[0].name

        # Requirement: Create the inputs dictionary
        inputs = {input_name: input_tensor}

        # Requirement: Obtain the output of the model
        outputs = _ONNX_SESSION.run(None, inputs)

        # Requirement: Obtain the logits (first dimension of outputs)
        logits = outputs[0][0]  # [0] is the batch, [0] inside that is the logits vector

        # Requirement: Obtain the class label based on logits and JSON labels
        predicted_idx = np.argmax(logits)

        return _CLASS_LABELS[predicted_idx]

    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Prediction Error: {str(e)}"


# --- Existing Utility Functions ---


def resize_image(image_file, width: int, height: int) -> Image.Image:
    """Resizes an image to a specific width and height."""
    with Image.open(image_file) as img:
        img = img.convert("RGB")
        resized_img = img.resize((width, height))
    return resized_img


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """Optional: Converts a PIL Image to grayscale."""
    return image.convert("L")


def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """Optional: Rotates a PIL Image by a given angle."""
    return image.rotate(angle, expand=True)


def apply_blur(image: Image.Image, radius: int = 2) -> Image.Image:
    """Optional: Applies a Gaussian blur filter to the image."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def normalize_image(image: Image.Image) -> np.ndarray:
    """Optional: Normalizes a PIL Image (CLI utility version)."""
    img_rgb = image.convert("RGB")
    img_array = np.asarray(img_rgb)
    normalized_array = img_array / 255.0
    return normalized_array


def get_image_info(image: Image.Image) -> dict:
    """Optional: Extracts metadata from a PIL Image."""
    return {
        "size": image.size,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
        "channels": len(image.getbands()),
        "is_color": image.mode not in ("L", "1"),
    }
