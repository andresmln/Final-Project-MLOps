import os
import pytest

def test_artifacts_exist():
    """
    Verifies that the necessary model artifacts for production exist.
    """
    assert os.path.exists("model.onnx"), "model.onnx not found. Run 'serialize.py' first."
    assert os.path.exists("class_labels.json"), "class_labels.json not found."
    
    # Check for .data file if your model is large
    assert os.path.exists("model.onnx.data"), "model.onnx.data not found."
    if os.path.exists("model.onnx.data"):
        assert os.path.getsize("model.onnx.data") > 0, "model.onnx.data is empty."
