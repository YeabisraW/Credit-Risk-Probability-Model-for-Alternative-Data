import pandas as pd

# Load the processed CSV
df_processed = pd.read_csv('processed_customer_level.csv')

# 1. Number of rows and columns
print("Shape of processed dataset:", df_processed.shape)

# 2. First few rows
print(df_processed.head())

# 3. Check numeric columns
numeric_cols = ['total_amount', 'avg_amount', 'transaction_count', 'std_amount', 'hour', 'day', 'month', 'year']
print("Numeric columns present:", all(col in df_processed.columns for col in numeric_cols))

# 4. Check for missing values
print("Missing values per column:\n", df_processed.isna().sum())

# 5. Unique customers
print("Number of unique customers:", df_processed['CustomerId'].nunique())
