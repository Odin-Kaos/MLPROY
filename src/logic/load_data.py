"""
Load dataset from CSV and split into train, validation, and test sets.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def load_csv():
    """
    Load dataset and return splits.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    root = Path(__file__).resolve().parents[2]
    data_path = root / "data" / "dataset.csv"
    df = pd.read_csv(data_path)

    # Convert categorical 'Yes/No' columns to 0/1
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train/test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.3, random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

