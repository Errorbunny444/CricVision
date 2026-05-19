# src/model_training.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# src/model_training.py
"""
Enhanced model_training.py for CRICVISION

- Loads match_summary from MySQL
- Builds features and target
- Trains a RandomForest pipeline
- Evaluates and saves model to models/match_predictor.pkl
- Saves confusion matrix image to models/confusion_matrix.png
"""

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import inspect, text
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from database.db_connections import get_engine

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]  # project root
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "match_predictor.pkl"
CM_PATH = MODEL_DIR / "confusion_matrix.png"


def load_match_summary(engine):
    """Load match_summary table into a DataFrame, with safety checks."""
    inspector = inspect(engine)
    if "match_summary" not in inspector.get_table_names():
        raise RuntimeError("Table 'match_summary' not found in database. Run src/prepare_match_summary.py first.")

    df = pd.read_sql(text("SELECT * FROM match_summary;"), engine)
    if df.shape[0] == 0:
        raise RuntimeError("Table 'match_summary' is empty. Rebuild/populate it before training.")
    return df


def prepare_xy(df):
    """
    Prepare X (features) and y (target).
    Expected feature columns (best-effort): team1, team2, venue, toss_winner, toss_decision, season
    Target: winner
    """
    # Detect column names flexibly
    cols = set(df.columns.str.lower())
    # mapping keys to prefered names
    col_map = {
        "team1": next((c for c in df.columns if c.lower() == "team1"), None),
        "team2": next((c for c in df.columns if c.lower() == "team2"), None),
        "venue": next((c for c in df.columns if c.lower() == "venue"), None),
        "toss_winner": next((c for c in df.columns if c.lower() == "toss_winner"), None),
        "toss_decision": next((c for c in df.columns if c.lower() == "toss_decision"), None),
        "season": next((c for c in df.columns if c.lower() == "season"), None),
        "winner": next((c for c in df.columns if c.lower() == "winner"), None),
    }

    missing = [k for k, v in col_map.items() if v is None and k != "winner"]
    if col_map["winner"] is None:
        raise RuntimeError("Target column 'winner' not found in match_summary.")
    # Not all feature columns are strictly required; we'll keep those that exist
    features = [col_map[k] for k in ("team1", "team2", "venue", "toss_winner", "toss_decision", "season") if col_map[k] is not None]

    if len(features) < 2:
        raise RuntimeError(f"Not enough feature columns found in match_summary. Found: {features}")

    X = df[features].copy()
    y = df[col_map["winner"]].copy()

    # Basic cleaning: fillna and canonicalize toss_decision
    X = X.fillna("Unknown")
    y = y.fillna("Unknown")

    # Normalize toss_decision to short forms if present
    toss_col = next((c for c in X.columns if c.lower() == "toss_decision"), None)
    if toss_col:
        X[toss_col] = X[toss_col].astype(str).str.lower().map({"bat": "bat", "field": "field", "bowl": "field"}).fillna("unknown")

    # Ensure strings
    for c in X.columns:
        X[c] = X[c].astype(str)

    y = y.astype(str)

    return X, y


def build_pipeline(categorical_cols):
    """Build sklearn pipeline with OneHotEncoder + RandomForest."""
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    transformer = ColumnTransformer([("ohe", ohe, categorical_cols)], remainder="drop")
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_depth=20,
        min_samples_leaf=2,
    )
    pipe = Pipeline([("pre", transformer), ("clf", clf)])
    return pipe


def plot_and_save_confusion(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel="True label", xlabel="Predicted label", title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    os.makedirs(out_path.parent, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    engine = get_engine()
    print("✅ MySQL Connection Successful!")

    df = load_match_summary(engine)
    print(f"📥 Loaded {len(df):,} rows from match_summary")

    X, y = prepare_xy(df)
    print("🔎 Feature columns:", X.columns.tolist())
    print("🎯 Target distribution:")
    print(y.value_counts().head(20))

    # Check classes
    classes = sorted(y.unique().tolist())
    if len(classes) < 2:
        raise RuntimeError("Not enough distinct target classes to train a classifier (need >=2).")

    # Train/test split. If classes are few per-class, stratify may fail — catch and fallback
    stratify = y if y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    except Exception:
        # fallback: no stratify
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

    # Build pipeline
    categorical_cols = X_train.columns.tolist()
    pipeline = build_pipeline(categorical_cols)

    # Fit
    print("🚀 Training model (this may take a minute)...")
    pipeline.fit(X_train, y_train)

    # Eval
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Done training. Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"💾 Saved pipeline to: {MODEL_PATH}")

    # Save confusion matrix (classes order as pipeline.classes_ may differ; use sorted classes)
    ordered_classes = sorted(list(set(y_test) | set(y_pred)))
    plot_and_save_confusion(y_test, y_pred, ordered_classes, CM_PATH)
    print(f"🖼 Saved confusion matrix image to: {CM_PATH}")

    print("🎯 Training finished. You can now use models/match_predictor.pkl in the Streamlit app.")

if __name__ == "__main__":
    main()

