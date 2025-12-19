import mlflow
from mlflow.tracking import MlflowClient
import torch
import os
import shutil

# --- Configuration ---
MLFLOW_TRACKING_URI = "mlruns"
MODEL_NAME = "PetImageClassifier"
# Output names for production artifacts
ONNX_MODEL_PATH = "model.onnx"
LABELS_PATH = "class_labels.json"


def main():
    # 1. Setup MLFlow Client
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print(f"Searching for registered versions of model: '{MODEL_NAME}'...")

    # 2. Query Registered Models
    # Filter by the specific model name we used in train.py
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error querying MLFlow: {e}")
        print("Did you run the training script yet?")
        return

    if not versions:
        print(f"No registered versions found for '{MODEL_NAME}'. Run training first.")
        return

    print(f"Found {len(versions)} versions. Comparing metrics...")

    # 3. Compare Models to Select the Best
    best_run_id = None
    best_version = None
    best_accuracy = -1.0

    for version in versions:
        run_id = version.run_id
        try:
            # Get run details to access metrics
            run = client.get_run(run_id)
            metrics = run.data.metrics

            # We logged 'final_val_accuracy' in train.py
            accuracy = metrics.get("final_val_accuracy", 0.0)

            print(
                f" - Version {version.version} (Run {run_id[:8]}...): Val Acc = {accuracy:.4f}"
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_run_id = run_id
                best_version = version

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"   Could not process version {version.version}: {e}")

    if not best_run_id:
        print("Could not identify a best model.")
        return

    print(
        f"\nBest Model: Version {best_version.version} (Run {best_run_id}) with Accuracy {best_accuracy:.4f}"
    )

    # 4. Load the Best Model
    print("Loading model for export...")
    model_uri = f"runs:/{best_run_id}/model"

    # Load model using MLFlow PyTorch flavor
    model = mlflow.pytorch.load_model(model_uri)

    # Move to CPU (Render doesn't support GPU) and set to Eval mode
    model.to("cpu")
    model.eval()

    # 5. Serialize to ONNX
    # Fixed W1309: Removed 'f' prefix as there are no variables
    print("Exporting model to ONNX format (opset 18)...")

    # Create dummy input matching the input size (Batch=1, Channels=3, H=224, W=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        opset_version=18,  # As requested in assignment
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model saved to {ONNX_MODEL_PATH}")

    # 6. Retrieve Class Labels Artifact
    print("Downloading class labels artifact...")

    # download_artifacts returns the local path where it downloaded the file
    local_path = client.download_artifacts(best_run_id, "class_labels.json", ".")

    # If the file wasn't downloaded directly to LABELS_PATH, move/rename it
    # (MLFlow might download it to a temp folder or keep original name)
    if local_path != LABELS_PATH and os.path.exists(local_path):
        shutil.move(local_path, LABELS_PATH)

    print(f"Class labels saved to {LABELS_PATH}")
    print("\nSerialization complete. Artifacts ready for production.")


if __name__ == "__main__":
    main()
