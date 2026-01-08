import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def loadCSV():
    data_path = Path("/app/data")
    ROOT = Path(__file__).resolve().parents[2]
    data_path = ROOT / "data" / "dataset.csv"
    df = pd.read_csv(data_path)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
