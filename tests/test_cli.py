"""
Integration testing with the CLI
"""
import os
import pytest
from click.testing import CliRunner
from PIL import Image
from cli.cli import cli

# --- Fixtures ---

@pytest.fixture
def runner():
    """Creates a Click runner instance."""
    return CliRunner()

@pytest.fixture
def sample_image(tmp_path):
    """
    Creates a temporary 100x100 RGB image for testing.
    Returns the file path as a string.
    """
    # Create a red image
    img = Image.new("RGB", (100, 100), color="red")
    
    # Save it to the temporary directory provided by pytest
    file_path = tmp_path / "test_image.jpg"
    img.save(file_path)
    
    return str(file_path)

# --- Tests ---

def test_help(runner):
    """Tests that the help command works."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Show this message and exit." in result.output


def test_predict_cli(runner, sample_image):
    """Tests the predict command."""
    # Command: predict <filepath>
    result = runner.invoke(cli, ["predict", sample_image])
    
    assert result.exit_code == 0
    assert "Prediction:" in result.output


def test_resize_cli(runner, sample_image, tmp_path):
    """Tests the resize command."""
    output_path = str(tmp_path / "resized.jpg")
    
    # Command: resize <filepath> -w 50 -h 50 -o <output>
    result = runner.invoke(cli, [
        "resize", sample_image, 
        "-w", "50", 
        "-h", "50", 
        "-o", output_path
    ])
    
    assert result.exit_code == 0
    assert "Image resized and saved" in result.output
    assert os.path.exists(output_path)


def test_info_cli(runner, sample_image):
    """Tests the info command."""
    # Command: info <filepath>
    result = runner.invoke(cli, ["info", sample_image])
    
    assert result.exit_code == 0
    assert "width" in result.output
    assert "100" in result.output


def test_grayscale_cli(runner, sample_image, tmp_path):
    """Tests the grayscale command."""
    output_path = str(tmp_path / "gray.jpg")
    
    # Command: grayscale <filepath> -o <output>
    result = runner.invoke(cli, [
        "grayscale", sample_image, 
        "-o", output_path
    ])
    
    assert result.exit_code == 0
    assert "Grayscale image saved" in result.output
    assert os.path.exists(output_path)


def test_rotate_cli(runner, sample_image, tmp_path):
    """Tests the rotate command."""
    output_path = str(tmp_path / "rotated.jpg")
    
    # Command: rotate <filepath> 90 -o <output>
    result = runner.invoke(cli, [
        "rotate", sample_image, 
        "90", 
        "-o", output_path
    ])
    
    assert result.exit_code == 0
    assert "Rotated image saved" in result.output
    assert os.path.exists(output_path)


def test_normalize_cli(runner, sample_image):
    """Tests the normalize command."""
    # Command: normalize <filepath>
    result = runner.invoke(cli, ["normalize", sample_image])
    
    assert result.exit_code == 0
    assert "Image normalized successfully" in result.output
    assert "Mean value" in result.output


def test_missing_file(runner):
    """Tests that the CLI handles missing files gracefully."""
    result = runner.invoke(cli, ["predict", "non_existent_file.jpg"])
    
    # Click usually returns exit code 2 for invalid file arguments (path does not exist)
    assert result.exit_code != 0
    assert "does not exist" in result.output
