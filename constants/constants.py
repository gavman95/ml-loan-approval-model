"""Constants for the project"""

import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
FILE_NAME = DATA_DIR / "loan_approval_dataset.csv"
TEST_FILE = DATA_DIR / "test_data.csv"
MODEL_PATH = ROOT_DIR / "models" / "loan_approval_model.pkl"
PREDICTIONS_PATH = DATA_DIR / "loan_approval_predictions.csv"
# AWS S3 Cloud Storage Configurations
S3_BUCKET_NAME = "loan-models"
S3_MODEL_PREFIX = ""
S3_MODEL_FILENAME = "model.pkl"
