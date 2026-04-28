from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from src.DataPreprocessing import DataLoader
import joblib
from constants.constants import FILE_NAME, MODEL_PATH
import mlflow
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTraining(DataLoader):
    def __init__(self, file_path: str = FILE_NAME, model_path: str = MODEL_PATH):
        super().__init__(file_path)
        self.X, self.y = self.preprocess_data()
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None 
        self.model_path = model_path

    def train_model(self, model_obj=None):
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        
        # 1. Use the passed model or default to Logistic
        if model_obj is not None:
            self.model = model_obj
        else:
            self.model = LogisticRegression()
        
        # 2. Fit the model
        self.model.fit(self.X_train, self.y_train)
        
        # 3. Log Parameters (This automatically logs to the pipeline's active run)
        mlflow.log_param("model_type", type(self.model).__name__)
        
        # Optional: Log all hyperparameters of the model automatically
        if hasattr(self.model, "get_params"):
            mlflow.log_params(self.model.get_params())
            
        print(f"Finished training: {type(self.model).__name__}")
        return self.model

    def evaluate_model(self):
        # Use self.model (renamed from self.lr to be generic)
        y_pred = self.model.predict(self.X_test)
        
        # Calculate numerical metrics for MLflow logging
        # Zero_division=0 prevents crashes if a model predicts only one class
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, zero_division=0)
        }

        # 1. Log metrics to MLflow so you can compare them in the UI
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # 2. Log the Confusion Matrix as an 'Artifact' (Image)
        # This is better than printing a text matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix: {type(self.model).__name__}')
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        # 3. Print for immediate feedback
        print(f"--- {type(self.model).__name__} Results ---")
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")
        
        return metrics

    def save_model(self):
        # Get the name of the algorithm (e.g., 'LogisticRegression')
        algo_name = type(self.model).__name__.lower()
        
        # Construct a dynamic path: e.g., .../models/logisticregression_loan_approval_model.pkl
        # .parent gets the directory, .name gets the filename
        dynamic_path = self.model_path.parent / f"{algo_name}_{self.model_path.name}"

        # 1. Local Save
        joblib.dump(self.model, dynamic_path)
        print(f"Model saved locally to {dynamic_path}")

        # 2. MLflow Save (Crucial for POCs)
        # This stores the model inside the MLflow experiment, linked to its metrics
        mlflow.sklearn.log_model(self.model, "model")
        print("Model logged to MLflow as an artifact")

def model_training_pipeline():
    trainer = ModelTraining()
    
    # List of models to try
    models = [
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(n_estimators=100)
            ]

    mlflow.set_experiment("Loan_Approval_Experiment")

    for m in models:
        # Wrap the entire process in ONE run
        with mlflow.start_run(run_name=type(m).__name__):
            trainer.train_model(m)
            trainer.evaluate_model()
            trainer.save_model()


    # --- THE WINNER SELECTION ---
    # We search the experiment we just ran and sort by F1 Score
    print("\n--- Selecting Best Model ---")
    runs_df = mlflow.search_runs(experiment_names=["Loan_Approval_Experiment"], order_by=["metrics.f1_score DESC"])
    
    if not runs_df.empty:
        best_run = runs_df.iloc[0]
        best_run_id = best_run.run_id
        best_model_type = best_run["params.model_type"]
        
        # Register this specific model version to the Registry
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, "Loan_Approval_Prod_Model")
        
        print(f"Winner: {best_model_type} with F1: {best_run['metrics.f1_score']:.4f}")
        print(f"Registered as 'Loan_Approval_Prod_Model' version {best_run_id}")
    
    return trainer

    

if __name__ == "__main__":
    model_training_pipeline()