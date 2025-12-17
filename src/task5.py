# task5.py (imbalanced class handling version)
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib

# ------------------------------
# Step 1: Load processed dataset
# ------------------------------
data_path = r"processed_customer_with_target.csv"
df = pd.read_csv(data_path)

# Features and target
X = df.drop(columns=['CustomerId', 'is_high_risk'])
y = df['is_high_risk']

# ------------------------------
# Step 2: Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Step 3: Scale numeric features
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# Step 4: Setup MLflow
# ------------------------------
mlflow.set_experiment("Credit_Risk_Modeling")

# ------------------------------
# Step 5: Define models and hyperparameters
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
# Step 6: Train, tune, and log experiments
# ------------------------------
best_model = None
best_roc_auc = 0

for model_name in models:
    model = models[model_name]
    param_grid = param_grids[model_name]
    
    # Grid search
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
# Step 7: Save best model locally
# ------------------------------
joblib.dump(best_model, "best_credit_risk_model.pkl")
print("Best model saved as 'best_credit_risk_model.pkl'.")
