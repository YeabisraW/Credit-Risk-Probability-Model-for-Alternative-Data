import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def create_aggregate_features(df):
    if 'CustomerId' not in df.columns:
        raise ValueError(f"'CustomerId' not found in dataframe. Columns: {df.columns}")

    # Aggregate numeric features by CustomerId
    agg_df = df.groupby("CustomerId").agg({
        "Amount": ["sum", "mean", "max"],
        "Value": ["sum", "mean", "max"]
    })

    # Flatten MultiIndex columns
    agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)
    return agg_df

def prepare_model_data(df):
    # Add aggregated features
    agg_df = create_aggregate_features(df)
    df = df.merge(agg_df, on="CustomerId", how="left")

    # Features and target
    target_col = "FraudResult"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Remove CustomerId and other identifiers from features
    for col in ["CustomerId", "TransactionId", "AccountId", "SubscriptionId", "BatchId"]:
        if col in categorical_cols:
            categorical_cols.remove(col)
        if col in numerical_cols:
            numerical_cols.remove(col)

    # Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor
