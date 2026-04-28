import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import sys
from src.DataPreprocessing import DataLoader
from constants.constants import TEST_FILE, MODEL_PATH, PREDICTIONS_PATH
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
from typing import Optional
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

        self.model = mlflow.sklearn.load_model(
            "models:/Loan_Approval_Prod_Model/latest"
        )


    def generate_predictions(self):
        # Use self.X (the processed unseen data)
        self.predictions = self.model.predict(self.X)
        self.predictions_labels = np.where(self.predictions==1,'Approved','Not Approved')

    
    def evaluate_model(self):
        metrics = {
            'accuracy': accuracy_score(self.y, self.predictions),
            'confusion_matrix': confusion_matrix(self.y, self.predictions),
            'classification_report': classification_report(self.y, self.predictions)
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