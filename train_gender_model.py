from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_FEATURES = [
    "first_name",
    "first_name_length",
    "last_letter",
    "vowel_count",
    "vowel_ratio",
    "age",
    "company"
]

CATEGORICAL_FEATURES = [
    "last_letter",
    "company"
]

NUMERICAL_FEATURES = [
    "first_name_length",
    "vowel_count",
    "vowel_ratio",
    "age"
]


def load_data(data_path):
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path.resolve()}"
        )

    df = pd.read_csv(data_path)

    required_columns = {
        "code",
        "company",
        "name",
        "gender",
        "age"
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df


def clean_data(df):
    clean_df = df.copy()

    clean_df = clean_df.drop_duplicates().copy()

    clean_df["company"] = (
        clean_df["company"]
        .astype(str)
        .str.strip()
    )

    clean_df["name"] = (
        clean_df["name"]
        .astype(str)
        .str.strip()
    )

    clean_df["gender"] = (
        clean_df["gender"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    clean_df["age"] = pd.to_numeric(
        clean_df["age"],
        errors="coerce"
    )

    clean_df = clean_df[
        clean_df["gender"].isin(["male", "female"])
    ].copy()

    clean_df = clean_df.dropna(
        subset=[
            "company",
            "name",
            "gender",
            "age"
        ]
    )

    clean_df = clean_df[
        clean_df["name"].str.len() > 0
    ].copy()

    clean_df = clean_df[
        clean_df["company"].str.len() > 0
    ].copy()

    return clean_df.reset_index(drop=True)


def engineer_features(df):
    feature_df = df.copy()

    feature_df["name"] = (
        feature_df["name"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    feature_df["first_name"] = (
        feature_df["name"]
        .str.split()
        .str[0]
    )

    feature_df["first_name_length"] = (
        feature_df["first_name"]
        .str.len()
    )

    feature_df["last_letter"] = (
        feature_df["first_name"]
        .str[-1]
    )

    feature_df["vowel_count"] = (
        feature_df["first_name"]
        .str.count(r"[aeiou]")
    )

    feature_df["vowel_ratio"] = (
        feature_df["vowel_count"]
        / feature_df["first_name_length"].replace(0, np.nan)
    ).fillna(0)

    return feature_df


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "name_tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    lowercase=True,
                    min_df=2,
                    sublinear_tf=True
                ),
                "first_name"
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES
            )
        ],
        remainder="drop"
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LinearSVC(
                random_state=RANDOM_STATE,
                max_iter=5000
            )
        )
    ])

    return pipeline


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    }

    return metrics, predictions


def save_artifacts(model, model_dir):
    model_dir = Path(model_dir)

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        model_dir
        / "gender_classification_model.joblib"
    )

    preprocessor_path = (
        model_dir
        / "gender_preprocessor.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        model.named_steps["preprocessor"],
        preprocessor_path
    )

    return model_path, preprocessor_path


def verify_saved_model(
    model_path,
    X_test,
    original_predictions
):
    loaded_model = joblib.load(
        model_path
    )

    reloaded_predictions = loaded_model.predict(
        X_test
    )

    return np.array_equal(
        original_predictions,
        reloaded_predictions
    )


def train_gender_model(
    data_path,
    model_dir
):
    df = load_data(
        data_path
    )

    original_records = len(df)

    clean_df = clean_data(
        df
    )

    feature_df = engineer_features(
        clean_df
    )

    X = feature_df[
        MODEL_FEATURES
    ].copy()

    y = feature_df[
        "gender"
    ].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = build_pipeline()

    model.fit(
        X_train,
        y_train
    )

    metrics, predictions = evaluate_model(
        model,
        X_test,
        y_test
    )

    model_path, preprocessor_path = save_artifacts(
        model,
        model_dir
    )

    model_verified = verify_saved_model(
        model_path,
        X_test,
        predictions
    )

    if not model_verified:
        raise RuntimeError(
            "Saved model verification failed."
        )

    return {
        "model": model,
        "metrics": metrics,
        "original_records": original_records,
        "clean_records": len(clean_df),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "correct_predictions": int(
            (predictions == y_test).sum()
        ),
        "incorrect_predictions": int(
            (predictions != y_test).sum()
        ),
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "model_verified": model_verified
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent

    data_path = (
        project_root
        / "data"
        / "users.csv"
    )

    model_dir = (
        project_root
        / "models"
    )

    result = train_gender_model(
        data_path=data_path,
        model_dir=model_dir
    )

    print("\nGENDER MODEL TRAINING COMPLETE")
    print("-" * 60)

    print(
        f"Original records: "
        f"{result['original_records']}"
    )

    print(
        f"Training-ready records: "
        f"{result['clean_records']}"
    )

    print(
        f"Training samples: "
        f"{result['training_samples']}"
    )

    print(
        f"Testing samples: "
        f"{result['testing_samples']}"
    )

    print("\nMODEL PERFORMANCE")
    print("-" * 60)

    print(
        f"Accuracy: "
        f"{result['metrics']['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{result['metrics']['precision'] * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{result['metrics']['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score: "
        f"{result['metrics']['f1_score'] * 100:.2f}%"
    )

    print(
        f"Correct predictions: "
        f"{result['correct_predictions']}"
    )

    print(
        f"Incorrect predictions: "
        f"{result['incorrect_predictions']}"
    )

    print("\nMODEL ARTIFACTS")
    print("-" * 60)

    print(
        f"Model saved to: "
        f"{result['model_path']}"
    )

    print(
        f"Preprocessor saved to: "
        f"{result['preprocessor_path']}"
    )

    print(
        "Reload verification: "
        f"{'PASSED' if result['model_verified'] else 'FAILED'}"
    )

    