"""Class to load the data and preprocess it"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, file_path: str):
        self.data_path = file_path
        self.data = self._load_data()
        self.X = None
        self.y = None

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df.columns = [c.strip() for c in df.columns]
        return df

    def preprocess_data(self):
        # Create a deep copy to avoid warnings and keep original data intact
        df = self.data.copy()

        # 1. Target and Feature split (Added quotes)
        if "loan_status" in df.columns:
            if isinstance(df["loan_status"].iloc[0], str):
                self.y = np.where(df["loan_status"].str.strip() == "Approved", 1, 0)
            else:
                self.y = df["loan_status"]

            X = df.drop(columns=["loan_status", "loan_id"], errors="ignore")
        else:
            # 🔥 API / inference mode
            self.y = None
            X = df.drop(columns=["loan_id"], errors="ignore")

        # 2. Asset Aggregation
        X["total_assets_value"] = (
            X["residential_assets_value"]
            + X["commercial_assets_value"]
            + X["luxury_assets_value"]
            + X["bank_asset_value"]
        )

        X.drop(
            columns=[
                "residential_assets_value",
                "commercial_assets_value",
                "luxury_assets_value",
                "bank_asset_value",
            ],
            inplace=True,
        )

        # 3. Categorical Encoding (Added quotes and stripping)
        X["education"] = np.where(X["education"].str.strip() == "Graduate", 1, 0)
        X["self_employed"] = np.where(X["self_employed"].str.strip() == "Yes", 1, 0)

        # 4. Log Transformation (Using log1p to handle zeros safely)
        log_cols = ["income_annum", "loan_amount", "total_assets_value"]
        X[log_cols] = np.log1p(X[log_cols])

        self.X = X
        return self.X, self.y
