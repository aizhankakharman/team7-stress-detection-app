from pathlib import Path
import os
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

DATA_FILE = "df_windows.csv"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

TARGET = "label"
DROP_COLS = ["subject"]
LOW_VARIANCE_DROP = ["HRV_mean", "HRV_std", "HRV_min", "HRV_max", "HRV_range", "HRV_median"]
LOG_FEATURES = [
    "EDA_mean", "EDA_std", "EDA_min", "EDA_max", "EDA_range", "EDA_median",
    "HR_std", "HR_range",
    "ACC_X_std", "ACC_X_range", "ACC_Y_std", "ACC_Y_range", "ACC_Z_std", "ACC_Z_range"
]
WRIST_AXES = ["ACC_X_mean", "ACC_Y_mean", "ACC_Z_mean"]


def prepare_features(df_windows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None, LabelEncoder | None]:
    df = df_windows.copy()
    y = None
    label_encoder = None

    if TARGET in df.columns:
        y_raw = df[TARGET].copy()
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        X_raw = df.drop(columns=[TARGET] + [c for c in DROP_COLS if c in df.columns])
    else:
        X_raw = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    X_base = X_raw.drop(columns=[c for c in LOW_VARIANCE_DROP if c in X_raw.columns])

    for feature in LOG_FEATURES:
        if feature in X_base.columns:
            min_value = X_base[feature].min()
            shift = abs(min_value) + 1e-6 if min_value < 0 else 0
            X_base[f"{feature}_log"] = np.log1p(X_base[feature] + shift)

    if all(col in X_base.columns for col in WRIST_AXES):
        X_base["acc_wrist_mag"] = np.sqrt((X_base[WRIST_AXES] ** 2).sum(axis=1))

    return X_base, y, label_encoder


if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("df_windows.csv was not found. Put df_windows.csv in this folder first.")

print("Loading dataset...")
df_windows = pd.read_csv(DATA_FILE)
X_base, y, label_encoder = prepare_features(df_windows)

if y is None or label_encoder is None:
    raise ValueError("The dataset must contain a label column for training.")

X_train, X_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.20, random_state=SEED, stratify=y
)

NUM_FEATURES = X_train.columns.tolist()


def build_pipeline(model):
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numerical_transformer, NUM_FEATURES)],
        remainder="drop"
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selector", SelectKBest(score_func=f_classif, k=20)),
            ("classifier", model)
        ]
    )

print("Training Random Forest model...")
pipeline = build_pipeline(RandomForestClassifier(random_state=SEED, n_jobs=-1))
param_grid = {
    "feature_selector__k": [20],
    "classifier__n_estimators": [150],
    "classifier__max_depth": [None],
    "classifier__min_samples_split": [2],
    "classifier__min_samples_leaf": [1]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    refit=True,
    cv=cv,
    n_jobs=-1,
    verbose=1
)
search.fit(X_train, y_train)
best_model = search.best_estimator_

y_pred = best_model.predict(X_test)
y_score = best_model.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred).tolist()
metrics = {
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    "test_roc_auc": float(roc_auc_score(y_test, y_score)),
    "test_pr_auc": float(average_precision_score(y_test, y_score)),
    "best_params": search.best_params_,
    "confusion_matrix": cm,
    "classes": [str(x) for x in label_encoder.classes_]
}

selector = best_model.named_steps["feature_selector"]
selected_features = np.array(NUM_FEATURES)[selector.get_support()].tolist()

# Save artifacts
joblib.dump(best_model, ARTIFACT_DIR / "stress_model.pkl")
joblib.dump(NUM_FEATURES, ARTIFACT_DIR / "feature_columns.pkl")
joblib.dump(selected_features, ARTIFACT_DIR / "selected_features.pkl")
joblib.dump(label_encoder, ARTIFACT_DIR / "label_encoder.pkl")
joblib.dump(X_train.sample(min(150, len(X_train)), random_state=SEED), ARTIFACT_DIR / "shap_background.pkl")
joblib.dump(metrics, ARTIFACT_DIR / "metrics.pkl")

# Save prepared data for sample demo inside the app
sample_df = X_base.copy()
if TARGET in df_windows.columns:
    sample_df[TARGET] = df_windows[TARGET].values
sample_df.to_csv(ARTIFACT_DIR / "prepared_samples.csv", index=False)

print("\nModel exported successfully.")
print("Saved files into artifacts/ folder.")
print(metrics)
