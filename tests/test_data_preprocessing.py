import numpy as np
import pandas as pd

from src.DataPreprocessing import DataLoader


def test_preprocess_data_creates_expected_features(tmp_path):
    df = pd.DataFrame(
        [
            {
                "loan_id": 1,
                "no_of_dependents": 2,
                "education": " Graduate ",
                "self_employed": " Yes ",
                "income_annum": 100000,
                "loan_amount": 20000,
                "loan_term": 12,
                "cibil_score": 700,
                "residential_assets_value": 1000,
                "commercial_assets_value": 2000,
                "luxury_assets_value": 3000,
                "bank_asset_value": 4000,
                "loan_status": "Approved",
            }
        ]
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    loader = DataLoader(str(csv_path))
    X, y = loader.preprocess_data()

    assert y.tolist() == [1]
    assert "loan_id" not in X.columns
    assert "total_assets_value" in X.columns
    assert "residential_assets_value" not in X.columns
    assert X.loc[0, "education"] == 1
    assert X.loc[0, "self_employed"] == 1
    assert np.isclose(X.loc[0, "total_assets_value"], np.log1p(10000))
