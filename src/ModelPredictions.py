import pandas as pd
import numpy as np
import io
import os
import boto3
import joblib
import re
from src.DataPreprocessing import DataLoader
from constants.constants import MODEL_PATH
from constants.constants import S3_MODEL_PREFIX, S3_MODEL_FILENAME
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class ModelPredictions(DataLoader):
    def __init__(self, file_path=None, input_df=None, model_path=MODEL_PATH):

        if input_df is not None:
            # API mode ✅
            self.data = input_df
        else:
            # File mode ✅
            super().__init__(file_path)

        # preprocess AFTER data is set
        self.preprocess_data()

        # S3 Constant configuration binding
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.prefix = S3_MODEL_PREFIX
        self.filename = S3_MODEL_FILENAME

        # 1. Evaluate your running environment variable mode
        self.env_mode = os.getenv("ENV_MODE", "local").lower()

        # 2. Automatically load the model from the correct regional storage target
        self.model = self._load_model()

    def _load_model(self):
        """Load model depending on environment (MinIO for local, S3 for prod)."""

        if self.env_mode == "production":
            print("🚀 [PRODUCTION MODE] Loading model from AWS S3...")
            return self._load_model_from_s3()

        else:
            print("💻 [LOCAL MODE] Loading model from MinIO (S3-compatible)...")
            return self._load_model_from_s3()


    def _get_latest_date_key(self):
        """Scans S3 date-style prefixes and returns the latest model path."""

        s3_client = self._get_s3_client()
        paginator = s3_client.get_paginator("list_objects_v2")

        pages = paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=self.prefix,
            Delimiter="/",
        )

        date_folders = []

        # strict YYYY-MM-DD folder filter
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}/$")

        for page in pages:
            for folder in page.get("CommonPrefixes", []):
                prefix = folder.get("Prefix")

                if not prefix:
                    continue

                # ignore noisy / invalid prefixes
                if prefix == self.prefix:
                    continue

                # only keep real date folders
                if not date_pattern.search(prefix):
                    continue

                date_folders.append(prefix)

        if not date_folders:
            raise FileNotFoundError(
                f"No valid date subfolders found under prefix '{self.prefix}' "
                f"in bucket '{self.bucket_name}'"
            )

        # sort chronologically (works because YYYY-MM-DD format is lexicographically sortable)
        latest_folder = sorted(date_folders)[-1]

        return f"{latest_folder}{self.filename}"

    def _load_model_from_s3(self):
        """Loads latest model from S3 or MinIO (same S3 API)."""
        try:
            s3_key = self._get_latest_date_key()
            print(f"Loading model from: s3://{self.bucket_name}/{s3_key}")

            s3_client = self._get_s3_client()

            response = s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)

            model_buffer = io.BytesIO(response["Body"].read())
            loaded_model = joblib.load(model_buffer)

            print("Model loaded successfully into memory.")
            return loaded_model

        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e

    def _get_s3_client(self):

        # LOCAL → MinIO
        if self.env_mode == "local":
            return boto3.client(
                "s3",
                endpoint_url=os.getenv("S3_ENDPOINT_URL"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name="us-east-1",
            )

        # PRODUCTION → AWS S3
        return boto3.client("s3")

    def generate_predictions(self):
        # Use self.X (the processed unseen data)
        self.predictions = self.model.predict(self.X)
        self.predictions_labels = np.where(
            self.predictions == 1, "Approved", "Not Approved"
        )

    def evaluate_model(self):
        metrics = {
            "accuracy": accuracy_score(self.y, self.predictions),
            "confusion_matrix": confusion_matrix(self.y, self.predictions),
            "classification_report": classification_report(self.y, self.predictions),
        }
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Confusion Matrix: {metrics['confusion_matrix']}")
        print(f"Classification Report: {metrics['classification_report']}")

    # def save_predictions(self):
    #     self.predictions_df = pd.DataFrame({'loan_id': self.data['loan_id'], 'predictions': self.predictions_labels})
    #     self.predictions_df.to_csv(self.predictions_path, index=False)
    #     print(f"Predictions saved to {self.predictions_path}")


def model_predictions_pipeline(input_df: pd.DataFrame):
    model_predictions = ModelPredictions(input_df=input_df)
    model_predictions.generate_predictions()
    return model_predictions


if __name__ == "__main__":
    model_predictions_pipeline()
