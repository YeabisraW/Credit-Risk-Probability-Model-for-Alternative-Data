# task4.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime

# ------------------------------
# Step 0: Load raw transaction data
# ------------------------------
file_path = r"C:\Users\sciec\Credit-Risk-Probability-Model-for-Alternative-Data\Data\raw\data.csv"
df = pd.read_csv(file_path)

# Parse datetime column
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')

# ------------------------------
# Step 1: Calculate RFM metrics per customer
# ------------------------------
# Define snapshot date as one day after last transaction
snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerId').agg(
    recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
    frequency=('TransactionId', 'count'),
    monetary=('Amount', 'sum')
).reset_index()

# ------------------------------
# Step 2: Scale RFM features for clustering
# ------------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

# ------------------------------
# Step 3: K-Means clustering into 3 groups
# ------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

# ------------------------------
# Step 4: Identify high-risk cluster
# ------------------------------
# Analyze cluster RFM means
cluster_summary = rfm.groupby('cluster').agg(
    recency_mean=('recency', 'mean'),
    frequency_mean=('frequency', 'mean'),
    monetary_mean=('monetary', 'mean')
).sort_values('recency_mean', ascending=False)

print("Cluster summary (for identifying high-risk cluster):")
print(cluster_summary)

# High-risk cluster: highest recency, lowest frequency & monetary
high_risk_cluster = cluster_summary.index[0]  # first row after sorting by recency

# Assign binary target
rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

# ------------------------------
# Step 5: Merge target into processed customer-level dataset
# ------------------------------
processed_df = pd.read_csv('processed_customer_level.csv')

# Merge is_high_risk
processed_df = processed_df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

# Save updated dataset
processed_df.to_csv('processed_customer_with_target.csv', index=False)
print("Target variable 'is_high_risk' added. Saved to 'processed_customer_with_target.csv'.")
