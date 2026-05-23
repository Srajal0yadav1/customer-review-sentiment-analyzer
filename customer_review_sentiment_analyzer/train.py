import os
import pickle
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_reviews.csv")
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")

def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Data loaded. Shape: {df.shape}")
    logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    return df

def build_pipeline(algorithm: str = "logistic_regression") -> Pipeline:
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    if algorithm == "logistic_regression":
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="auto",
            random_state=42,
        )
    elif algorithm == "naive_bayes":
        clf = MultinomialNB(alpha=0.1)
    elif algorithm == "svm":
        from sklearn.calibration import CalibratedClassifierCV
        base_clf = LinearSVC(
            C=1.0,
            max_iter=2000,
            random_state=42,
        )
        clf = CalibratedClassifierCV(estimator=base_clf, cv=5)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf", clf),
    ])

    logger.info(f"Pipeline built with {algorithm}")
    return pipeline

def tune_hyperparameters(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    param_grid = {
        "tfidf__max_features": [30000, 50000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=2,
    )

    logger.info("Starting hyperparameter grid search...")
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def evaluate_model(pipeline, X_test, y_test) -> None:
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    logger.info(f"\nTest Accuracy: {acc:.4f}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Negative", "Neutral", "Positive"],
        )
    )

    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Neutral", "Positive"])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Neutral", "Actual Positive"],
        columns=["Pred Negative", "Pred Neutral", "Pred Positive"],
    )
    print(cm_df)
    print("=" * 60)

def save_model(pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Model saved to {path}")


def load_model(path: str):
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    logger.info(f"Model loaded from {path}")
    return pipeline

def train(tune: bool = False, algorithm: str = "svm") -> None:
    df = load_data(PROCESSED_DATA_PATH)
    X = df["clean_text"].values
    y = df["sentiment"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    pipeline = build_pipeline(algorithm)

    if tune:
        pipeline = tune_hyperparameters(pipeline, X_train, y_train)
    else:
        logger.info("Training model...")
        pipeline.fit(X_train, y_train)
        logger.info("Training complete.")

    evaluate_model(pipeline, X_test, y_test)

    save_model(pipeline, MODEL_OUTPUT_PATH)

if __name__ == "__main__":
    train(tune=False, algorithm="svm")
