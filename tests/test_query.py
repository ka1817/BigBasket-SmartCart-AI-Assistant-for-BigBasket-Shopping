import os
import pytest

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

@pytest.fixture(scope="module")
def models_dir_exists():
    """Check if the models folder exists."""
    if not os.path.isdir(MODELS_DIR):
        pytest.fail(f"Models folder not found at {MODELS_DIR}. Please train and save the model first.")
    return MODELS_DIR


def test_models_folder_exists(models_dir_exists):
    """Ensure that the models directory exists."""
    assert os.path.isdir(models_dir_exists), f"Expected models directory at {models_dir_exists}"


def test_pipeline_file_exists(models_dir_exists):
    """Ensure that the classifier_pipeline.pkl file exists inside models/."""
    pipeline_path = os.path.join(models_dir_exists, "RandomForest_pipeline.joblib")
    assert os.path.isfile(pipeline_path), f"Expected pipeline file not found at {pipeline_path}"