# task5.py (imbalanced class handling version with integrated unit tests)
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
import sys

# ------------------------------
# Unit tests (quick checks before full training)
# ------------------------------
def run_unit_tests(df):
    print("Running quick unit tests...")
    # 1. Train-test split shapes
    X = df.drop(columns=['CustomerId', 'is_high_risk'])
    y = df['is_high_risk']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0], "Train/Test split row mismatch"
    assert X_train.shape[1] == X.shape[1], "Train/Test split column mismatch"
    # 2. Logistic Regression quick training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    assert 0.0 <= roc_auc <= 1.0, "ROC-AUC out of bounds"
    # 3. MLflow run check
    mlflow.set_experiment("Credit_Risk_Testing")
    with mlflow.start_run(run_name="test_run") as run:
        mlflow.log_metric("test_metric", 0.5)
        assert run.info.run_id is not None, "MLflow run ID not created"
    print("All unit tests passed!\n")

# ------------------------------
# Step 1: Load processed dataset
# ------------------------------
data_path = r"processed_customer_with_target.csv"
df = pd.read_csv(data_path)

# Run unit tests
run_unit_tests(df)

# ------------------------------
# Step 2: Features and target
# ------------------------------
X = df.drop(columns=['CustomerId', 'is_high_risk'])
y = df['is_high_risk']

# ------------------------------
# Step 3: Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Step 4: Scale numeric features
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# Step 5: Setup MLflow
# ------------------------------
mlflow.set_experiment("Credit_Risk_Modeling")

# ------------------------------
# Step 6: Define models and hyperparameters
# ------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced')
}

param_grids = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs"]
    },
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    }
}

# ------------------------------
# Step 7: Train, tune, and log experiments
# ------------------------------
best_model = None
best_roc_auc = 0

for model_name in models:
    model = models[model_name]
    param_grid = param_grids[model_name]

    grid = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
    grid.fit(X_train_scaled, y_train)

    best_estimator = grid.best_estimator_
    y_pred = best_estimator.predict(X_test_scaled)
    y_prob = best_estimator.predict_proba(X_test_scaled)[:, 1] if hasattr(best_estimator, "predict_proba") else y_pred

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    # MLflow logging
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })
        mlflow.sklearn.log_model(best_estimator, artifact_path="model")

        print(f"Model: {model_name}")
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        print("-"*50)

    # Track best model
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_model = best_estimator

# ------------------------------
# Step 8: Save best model locally
# ------------------------------
joblib.dump(best_model, "best_credit_risk_model.pkl")
print("Best model saved as 'best_credit_risk_model.pkl'.")
