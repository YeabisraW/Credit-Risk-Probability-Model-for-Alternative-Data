# src/data_processing.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def load_data(filepath, nrows=None):
    """Load CSV data with basic error handling."""
    try:
        df = pd.read_csv(filepath, nrows=nrows)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def plot_numerical_distribution(df, numerical_cols):
    """Plot numerical columns with histogram and KDE."""
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=50)
        plt.title(f'Distribution of {col}')
        plt.show()


def plot_categorical_distribution(df, categorical_cols):
    """Plot categorical columns with count plots."""
    for col in categorical_cols:
        plt.figure(figsize=(7, 4))
        sns.countplot(y=df[col], data=df, order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.show()


def correlation_matrix(df, numerical_cols):
    """Plot correlation heatmap of numerical columns."""
    plt.figure(figsize=(5, 4))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()


def missing_values(df):
    """Print missing values per column."""
    missing = df.isnull().sum()
    print("\nMissing values per column:\n", missing)


def outlier_boxplot(df, numerical_cols):
    """Plot boxplots for numerical columns to detect outliers."""
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()


def calculate_rfm(df):
    """Compute RFM features for customers."""
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

