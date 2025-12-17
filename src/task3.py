# task3.py - fully fixed

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------------
# Custom Transformer for aggregation
# ------------------------------

class TransactionAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount', datetime_col='TransactionStartTime'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()

        # Parse datetime
        X_[self.datetime_col] = pd.to_datetime(X_[self.datetime_col], errors='coerce')

        # Numeric aggregation
        agg_numeric = X_.groupby(self.customer_id_col)[self.amount_col].agg(
            total_amount='sum',
            avg_amount='mean',
            transaction_count='count',
            std_amount='std'
        ).reset_index()
        agg_numeric['std_amount'] = agg_numeric['std_amount'].fillna(0)

        # Datetime aggregation
        X_['hour'] = X_[self.datetime_col].dt.hour
        X_['day'] = X_[self.datetime_col].dt.day
        X_['month'] = X_[self.datetime_col].dt.month
        X_['year'] = X_[self.datetime_col].dt.year

        agg_datetime = X_.groupby(self.customer_id_col).agg(
            hour=('hour', 'mean'),
            day=('day', 'mean'),
            month=('month', 'mean'),
            year=('year', 'mean')
        ).reset_index()

        # Categorical aggregation (mode)
        cat_cols = X_.select_dtypes(include='object').columns.tolist()
        exclude_cols = [self.customer_id_col, self.datetime_col, 'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId']
        cat_cols = [c for c in cat_cols if c not in exclude_cols]

        agg_categorical = pd.DataFrame()
        if len(cat_cols) > 0:
            agg_categorical = X_.groupby(self.customer_id_col)[cat_cols].agg(lambda x: x.mode()[0]).reset_index()

        # Merge all
        df_agg = agg_numeric.merge(agg_datetime, on=self.customer_id_col, how='left')
        if not agg_categorical.empty:
            df_agg = df_agg.merge(agg_categorical, on=self.customer_id_col, how='left')

        return df_agg

# ------------------------------
# Load dataset
# ------------------------------

file_path = r"C:\Users\sciec\Credit-Risk-Probability-Model-for-Alternative-Data\Data\raw\data.csv"
df = pd.read_csv(file_path)

# ------------------------------
# Define features
# ------------------------------

numeric_features = ['total_amount', 'avg_amount', 'transaction_count', 'std_amount', 'hour', 'day', 'month', 'year']

# After aggregation, categorical columns
categorical_features = df.select_dtypes(include='object').columns.tolist()
exclude_cols = ['CustomerId', 'TransactionStartTime', 'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId']
categorical_features = [c for c in categorical_features if c not in exclude_cols]

# ------------------------------
# Preprocessing pipelines
# ------------------------------

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ------------------------------
# Full pipeline
# ------------------------------

pipeline = Pipeline(steps=[
    ('aggregation', TransactionAggregator(
        customer_id_col='CustomerId',
        amount_col='Amount',
        datetime_col='TransactionStartTime'
    )),
    ('preprocessor', preprocessor)
])

# ------------------------------
# Transform dataset
# ------------------------------

X_transformed = pipeline.fit_transform(df)

# Convert to dense if sparse
X_transformed_dense = X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed

# Get column names
cat_cols_encoded = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
columns = numeric_features + list(cat_cols_encoded)

# Convert to DataFrame
X_transformed_df = pd.DataFrame(X_transformed_dense, columns=columns)

# Add CustomerId
agg_ids = df['CustomerId'].unique()
X_transformed_df.insert(0, 'CustomerId', agg_ids)

# Save CSV
X_transformed_df.to_csv('processed_customer_level.csv', index=False)
print("Customer-level feature engineering complete. Output saved to 'processed_customer_level.csv'.")
