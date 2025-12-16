# tests/test_data_processing.py

import sys
import os
import pandas as pd
import pytest

# -----------------------------
# Fix for Windows + Python 3.13: Add src/ to path
# -----------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_processing import clean_data, calculate_rfm, handle_missing, create_default_proxy

# -----------------------------
# Sample Data for Testing
# -----------------------------
sample_data = pd.DataFrame({
    'CustomerId': [1, 1, 2, 2],
    'TransactionId': [101, 102, 201, 202],
    'TransactionStartTime': ['2025-12-01', '2025-12-05', '2025-12-02', '2025-12-06'],
    'Amount': [100, 200, 50, 75],
    'Value': [100, 200, 50, 75],
    'ChannelId': ['Web', None, 'Android', 'Web'],
})

# -----------------------------
# Test clean_data
# -----------------------------
def test_clean_data():
    df = clean_data(sample_data.copy())
    # Duplicates removed
    assert df.duplicated(subset=['TransactionId']).sum() == 0
    # Amount numeric
    assert pd.api.types.is_numeric_dtype(df['Amount'])

# -----------------------------
# Test handle_missing
# -----------------------------
def test_handle_missing():
    df = handle_missing(sample_data.copy(), categorical_strategy='mode', numerical_strategy='median')
    # No missing values
    assert df.isnull().sum().sum() == 0

# -----------------------------
# Test calculate_rfm
# -----------------------------
def test_calculate_rfm():
    df_clean = clean_data(sample_data.copy())
    rfm = calculate_rfm(df_clean)
    # Columns exist
    assert all(col in rfm.columns for col in ['CustomerId', 'Recency', 'Frequency', 'Monetary'])
    # Two unique customers
    assert rfm.shape[0] == 2

# -----------------------------
# Test create_default_proxy
# -----------------------------
def test_create_default_proxy():
    df_clean = clean_data(sample_data.copy())
    rfm = calculate_rfm(df_clean)
    proxy = create_default_proxy(rfm, recency_threshold=1, frequency_threshold=1, monetary_threshold=60)
    # Column exists
    assert 'DefaultProxy' in proxy.columns
    # Values are 0 or 1
    assert proxy['DefaultProxy'].isin([0, 1]).all()
