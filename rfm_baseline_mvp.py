"""
MVP: RFM baseline on Olist + heuristic validators for k
=========================================================
Goal: one figure showing that standard heuristics (elbow, silhouette,
Davies-Bouldin) disagree on the "right" number of customer segments.
This motivates the SpecialK / concentration-inequality validation step.

Setup
-----
1. Download the Olist dataset from Kaggle:
   https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Unzip into a folder called `data/` next to this script.
   Files needed:
     - olist_orders_dataset.csv
     - olist_customers_dataset.csv
     - olist_order_payments_dataset.csv
3. pip install pandas numpy scikit-learn matplotlib
4. python rfm_baseline_mvp.py

Output: rfm_heuristics.png + a printed summary table.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

DATA_DIR = "data"
K_RANGE = range(2, 11)
RANDOM_STATE = 42
SILHOUETTE_SAMPLE = 10_000  # silhouette is O(n^2); subsample for speed

# ---------------------------------------------------------------
# 1. Load and join
# ---------------------------------------------------------------
orders = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv",
                     parse_dates=["order_purchase_timestamp"])
customers = pd.read_csv(f"{DATA_DIR}/olist_customers_dataset.csv")
payments = pd.read_csv(f"{DATA_DIR}/olist_order_payments_dataset.csv")

# Keep only delivered orders (completed purchase behaviour)
orders = orders[orders["order_status"] == "delivered"]

# IMPORTANT: customer_id is unique per ORDER in Olist.
# customer_unique_id is the actual person -> needed to see repeat purchases.
orders = orders.merge(customers[["customer_id", "customer_unique_id"]],
                      on="customer_id", how="left")

# Total payment per order (payments can have multiple rows per order)
order_value = payments.groupby("order_id")["payment_value"].sum().rename("order_value")
orders = orders.merge(order_value, on="order_id", how="left")
orders = orders.dropna(subset=["order_value"])

# ---------------------------------------------------------------
# 2. RFM features per customer
# ---------------------------------------------------------------
snapshot = orders["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

rfm = orders.groupby("customer_unique_id").agg(
    recency=("order_purchase_timestamp", lambda s: (snapshot - s.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("order_value", "sum"),
).reset_index()

print(f"Customers: {len(rfm):,}")
print(f"Share with a single purchase: {(rfm['frequency'] == 1).mean():.1%}")
print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

# Log-transform skewed features, then standardize.
X = rfm[["recency", "frequency", "monetary"]].copy()
X["frequency"] = np.log1p(X["frequency"])
X["monetary"] = np.log1p(X["monetary"])
X = StandardScaler().fit_transform(X)

# ---------------------------------------------------------------
# 3. k-means across k, three heuristics
# ---------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)
sil_idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)

inertias, silhouettes, dbs = [], [], []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X[sil_idx], labels[sil_idx]))
    dbs.append(davies_bouldin_score(X, labels))
    print(f"k={k}: inertia={km.inertia_:,.0f}  "
          f"silhouette={silhouettes[-1]:.3f}  DB={dbs[-1]:.3f}")

best_sil = list(K_RANGE)[int(np.argmax(silhouettes))]
best_db = list(K_RANGE)[int(np.argmin(dbs))]
print(f"\nSilhouette prefers k={best_sil}, Davies-Bouldin prefers k={best_db}, "
      f"and the elbow is wherever you squint hard enough.")

# ---------------------------------------------------------------
# 4. The figure
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(K_RANGE, inertias, "o-")
axes[0].set_title("Elbow (inertia)")
axes[0].set_xlabel("k")

axes[1].plot(K_RANGE, silhouettes, "o-", color="tab:orange")
axes[1].axvline(best_sil, ls="--", color="gray", alpha=0.6)
axes[1].set_title(f"Silhouette (max at k={best_sil})")
axes[1].set_xlabel("k")

axes[2].plot(K_RANGE, dbs, "o-", color="tab:green")
axes[2].axvline(best_db, ls="--", color="gray", alpha=0.6)
axes[2].set_title(f"Davies-Bouldin (min at k={best_db})")
axes[2].set_xlabel("k")

fig.suptitle("RFM segmentation on Olist: standard heuristics for choosing k",
             fontsize=13)
fig.tight_layout()
fig.savefig("rfm_heuristics.png", dpi=150, bbox_inches="tight")
print("\nSaved rfm_heuristics.png")

km2 = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
rfm["c"] = km2.labels_
print(pd.crosstab(rfm["c"], rfm["frequency"] == 1))

# Recency + Monetary only, same preprocessing
X_rm = rfm[["recency", "monetary"]].copy()
X_rm["monetary"] = np.log1p(X_rm["monetary"])
X_rm = StandardScaler().fit_transform(X_rm)

for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_rm)
    sil = silhouette_score(X_rm[sil_idx], km.labels_[sil_idx])
    print(f"k={k}: silhouette={sil:.3f}")

