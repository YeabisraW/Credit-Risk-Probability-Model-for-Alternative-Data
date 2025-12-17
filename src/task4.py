# task4.py
# RFM-based target creation (parameterized and robust)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import argparse
import os

# ------------------------------
# Step 0: Parse arguments / parameters
# ------------------------------
parser = argparse.ArgumentParser(description="RFM Target Creation")
parser.add_argument(
    "--raw_data", default="Data/raw/data.csv",
    help="Path to raw transaction CSV"
)
parser.add_argument(
    "--processed_data", default="processed_customer_level_woe.csv",
    help="Path to processed customer-level CSV from Task 3"
)
parser.add_argument(
    "--output", default="processed_customer_with_target.csv",
    help="Output CSV with target"
)
parser.add_argument(
    "--snapshot_date", default=None,
    help="Reference date for RFM (YYYY-MM-DD). Defaults to last transaction + 1 day"
)
parser.add_argument(
    "--n_clusters", type=int, default=3,
    help="Number of KMeans clusters"
)
args = parser.parse_args()

# ------------------------------
# Step 1: Ensure output directory exists (fix for Windows)
# ------------------------------
output_dir = os.path.dirname(args.output)
if output_dir != "":
    os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Step 2: Load raw transaction data
# ------------------------------
df = pd.read_csv(args.raw_data)
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')

# ------------------------------
# Step 3: Define snapshot date
# ------------------------------
if args.snapshot_date:
    snapshot_date = pd.to_datetime(args.snapshot_date)
else:
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

# ------------------------------
# Step 4: Calculate RFM metrics per customer
# ------------------------------
rfm = df.groupby('CustomerId').agg(
    recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
    frequency=('TransactionId', 'count'),
    monetary=('Amount', 'sum')
).reset_index()

# ------------------------------
# Step 5: Scale RFM features for clustering
# ------------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

# ------------------------------
# Step 6: K-Means clustering
# ------------------------------
kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

# ------------------------------
# Step 7: Identify high-risk cluster
# ------------------------------
cluster_summary = rfm.groupby('cluster').agg(
    recency_mean=('recency', 'mean'),
    frequency_mean=('frequency', 'mean'),
    monetary_mean=('monetary', 'mean')
).sort_values('recency_mean', ascending=False)

high_risk_cluster = cluster_summary.index[0]
rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

# ------------------------------
# Step 8: Merge target into processed customer-level dataset
# ------------------------------
processed_df = pd.read_csv(args.processed_data)
processed_df = processed_df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

# ------------------------------
# Step 9: Save updated dataset
# ------------------------------
processed_df.to_csv(args.output, index=False)
print(f"Target variable 'is_high_risk' added. Saved to '{args.output}'.")
