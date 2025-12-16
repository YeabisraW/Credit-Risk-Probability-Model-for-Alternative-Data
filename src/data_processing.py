# src/data_processing.py

import pandas as pd
import numpy as np

# -----------------------------
# 1. Load Data
# -----------------------------
def load_data(filepath, nrows=None):
    """
    Load CSV data from the specified path.
    
    Parameters:
        filepath (str): Path to CSV file
        nrows (int): Number of rows to read (optional)
    
    Returns:
        pd.DataFrame
    """
    df = pd.read_csv(filepath, nrows=nrows)
    return df

# -----------------------------
# 2. Basic Cleaning
# -----------------------------
def clean_data(df):
    """
    Perform basic cleaning of the dataset:
    - Converts TransactionStartTime to datetime
    - Ensures Amount and Value are numeric
    - Drops duplicate TransactionId
    
    Parameters:
        df (pd.DataFrame)
    
    Returns:
        pd.DataFrame
    """
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.drop_duplicates(subset=['TransactionId'])
    return df

# -----------------------------
# 3. Handle Missing Values
# -----------------------------
def handle_missing(df, categorical_strategy='mode', numerical_strategy='median'):
    """
    Fill missing values in categorical and numerical columns.
    
    Parameters:
        df (pd.DataFrame)
        categorical_strategy (str): 'mode' or 'constant'
        numerical_strategy (str): 'median' or 'mean'
    
    Returns:
        pd.DataFrame
    """
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Fill categorical
    for col in cat_cols:
        if categorical_strategy == 'mode':
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif categorical_strategy == 'constant':
            df[col].fillna('Missing', inplace=True)
    
    # Fill numerical
    for col in num_cols:
        if numerical_strategy == 'median':
            df[col].fillna(df[col].median(), inplace=True)
        elif numerical_strategy == 'mean':
            df[col].fillna(df[col].mean(), inplace=True)
    
    return df

# -----------------------------
# 4. RFM Calculation
# -----------------------------
def calculate_rfm(df):
    """
    Calculate Recency, Frequency, Monetary features for credit risk proxy.
    
    Parameters:
        df (pd.DataFrame): Must include 'CustomerId', 'TransactionId', 'TransactionStartTime', 'Amount'
    
    Returns:
        pd.DataFrame with columns: CustomerId, Recency, Frequency, Monetary
    """
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Recency
    last_transaction = df.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
    last_transaction['Recency'] = (df['TransactionStartTime'].max() - last_transaction['TransactionStartTime']).dt.days
    
    # Frequency
    frequency = df.groupby('CustomerId')['TransactionId'].count().reset_index().rename(columns={'TransactionId': 'Frequency'})
    
    # Monetary
    monetary = df.groupby('CustomerId')['Amount'].sum().reset_index().rename(columns={'Amount': 'Monetary'})
    
    # Merge RFM
    rfm = last_transaction.merge(frequency, on='CustomerId').merge(monetary, on='CustomerId')
    
    return rfm

# -----------------------------
# 5. Feature Selection
# -----------------------------
def select_features(df, threshold=0.9):
    """
    Drop features that are highly correlated above a threshold.
    
    Parameters:
        df (pd.DataFrame)
        threshold (float): Correlation threshold to drop features
    
    Returns:
        pd.DataFrame with reduced features
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > threshold)]
    df_reduced = df.drop(columns=drop_cols)
    return df_reduced

# -----------------------------
# 6. Create Default Proxy (Optional)
# -----------------------------
def create_default_proxy(rfm_df, recency_threshold=30, frequency_threshold=5, monetary_threshold=500):
    """
    Assign a simple Good/Bad label based on RFM thresholds.
    
    Parameters:
        rfm_df (pd.DataFrame): Must include Recency, Frequency, Monetary
        recency_threshold (int)
        frequency_threshold (int)
        monetary_threshold (float)
    
    Returns:
        pd.DataFrame with 'DefaultProxy' column (1=High risk, 0=Low risk)
    """
    df = rfm_df.copy()
    df['DefaultProxy'] = np.where(
        (df['Recency'] > recency_threshold) | 
        (df['Frequency'] < frequency_threshold) | 
        (df['Monetary'] < monetary_threshold),
        1, 0
    )
    return df
