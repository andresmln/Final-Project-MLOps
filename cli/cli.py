"""
Main CLI or app entry point
"""

import click
import json
from PIL import Image
import numpy as np

# Import your new image processing functions
from mylib.inference_image_processor import (
    predict_image,
    resize_image,
    convert_to_grayscale,
    rotate_image,
    apply_blur,
    normalize_image,
    get_image_info
)

# We create a group of commands
@click.group()
def cli():
    """A CLI for image processing."""


# --- Command for Prediction ---
@cli.command("predict")
@click.argument("filepath", type=click.Path(exists=True))
def predict_cli(filepath):
    """
    Predicts the class of an image using the ONNX model.

    Example:
        uv run python -m cli.cli predict path/to/your/image.jpg
    """
    try:
        with Image.open(filepath) as img:
            prediction = predict_image(img)
        
        # Check if the logic returned an error string
        if prediction.startswith("Error") or prediction.startswith("Prediction Error"):
            click.echo(click.style(prediction, fg="red"))
        else:
            click.echo(click.style(f"Prediction: {prediction}", fg="green"))
            
    except (IOError, ValueError) as e:
        # Catch specific image loading or data conversion errors
        click.echo(click.style(f"Error processing image: {e}", fg="red"))


# --- Command for Resizing ---
@cli.command("resize")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("-w", "--width", type=int, required=True, help="New width for the image.")
@click.option("-h", "--height", type=int, required=True, help="New height for the image.")
@click.option("-o", "--output", type=click.Path(), required=True, help="Path to save the resized image.")
def resize_cli(filepath, width, height, output):
    """
    Resizes an image to a new width and height.

    Example:
        uv run python -m cli.cli resize "input.jpg" -w 100 -h 100 -o "output.jpg"
    """
    try:
        # This function handles its own file opening
        resized_img = resize_image(filepath, width, height)
        resized_img.save(output)
        click.echo(click.style(f"Image resized and saved to {output}", fg="green"))
    except (IOError, ValueError) as e:
        # Catch errors related to file saving or dimension conversion
        click.echo(click.style(f"Error resizing image: {e}", fg="red"))


# --- Command for Image Info ---
@cli.command("info")
@click.argument("filepath", type=click.Path(exists=True))
def info_cli(filepath):
    """
    Gets metadata from an image (size, mode, format, etc.).

    Example:
        uv run python -m cli.cli info "path/to/image.jpg"
    """
    try:
        with Image.open(filepath) as img:
            info = get_image_info(img)
        # Use json.dumps for a nicely formatted dictionary output
        click.echo(json.dumps(info, indent=4))
    except (IOError, ValueError) as e:
        click.echo(click.style(f"Error reading image info: {e}", fg="red"))


# --- Command for Grayscale ---
@cli.command("grayscale")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), required=True, help="Path to save the grayscale image.")
def grayscale_cli(filepath, output):
    """
    Converts an image to grayscale.

    Example:
        uv run python -m cli.cli grayscale "input.jpg" -o "output_gray.jpg"
    """
    try:
        with Image.open(filepath) as img:
            gray_img = convert_to_grayscale(img)
        gray_img.save(output)
        click.echo(click.style(f"Grayscale image saved to {output}", fg="green"))
    except (IOError, ValueError) as e:
        click.echo(click.style(f"Error converting image: {e}", fg="red"))


# --- Command for Rotate ---
@cli.command("rotate")
@click.argument("filepath", type=click.Path(exists=True))
@click.argument("angle", type=float)
@click.option("-o", "--output", type=click.Path(), required=True, help="Path to save the rotated image.")
def rotate_cli(filepath, angle, output):
    """
    Rotates an image by a given angle (in degrees).

    Example:
        uv run python -m cli.cli rotate "input.jpg" 90 -o "output_rotated.jpg"
    """
    try:
        with Image.open(filepath) as img:
            rotated_img = rotate_image(img, angle)
        rotated_img.save(output)
        click.echo(click.style(f"Rotated image saved to {output}", fg="green"))
    except (IOError, ValueError) as e:
        click.echo(click.style(f"Error rotating image: {e}", fg="red"))


# --- Command for Blur ---
@cli.command("blur")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("-r", "--radius", type=int, default=2, help="Blur radius (default: 2).")
@click.option("-o", "--output", type=click.Path(), required=True, help="Path to save the blurred image.")
def blur_cli(filepath, radius, output):
    """
    Applies a Gaussian blur to an image.

    Example:
        uv run python -m cli.cli blur "input.jpg" -r 5 -o "output_blurred.jpg"
    """
    try:
        with Image.open(filepath) as img:
            blurred_img = apply_blur(img, radius)
        blurred_img.save(output)
        click.echo(click.style(f"Blurred image saved to {output}", fg="green"))
    except (IOError, ValueError) as e:
        click.echo(click.style(f"Error blurring image: {e}", fg="red"))


# --- Command for Normalize ---
@cli.command("normalize")
@click.argument("filepath", type=click.Path(exists=True))
def normalize_cli(filepath):
    """
    Normalizes image pixels to [0, 1] and prints array info.

    Example:
        uv run python -m cli.cli normalize "input.jpg"
    """
    try:
        with Image.open(filepath) as img:
            norm_array = normalize_image(img)
        click.echo(click.style("Image normalized successfully.", fg="green"))
        click.echo(f"  Shape: {norm_array.shape}")
        click.echo(f"  Mean value: {np.mean(norm_array):.4f}")
        click.echo(f"  Min value: {np.min(norm_array):.4f}")
        click.echo(f"  Max value: {np.max(norm_array):.4f}")
    except (IOError, ValueError) as e:
        click.echo(click.style(f"Error normalizing image: {e}", fg="red"))

# Main entry point
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
