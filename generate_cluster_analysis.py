import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading Data...")
df = pd.read_csv("nyc_housing_base.csv")

# Column Mapping
column_mapping = {
    'sale_price': 'SALE PRICE',
    'bldgarea': 'GROSS SQUARE FEET',
    'yearbuilt': 'YEAR BUILT',
    'latitude': 'LATITUDE',
    'longitude': 'LONGITUDE',
    'borough_y': 'BOROUGH'
}
df = df.rename(columns=column_mapping)

# ==========================================
# 2. ETL & FEATURE ENGINEERING
# ==========================================
print("Cleaning Data...")
# Clean rows
df = df[df['SALE PRICE'] > 100_000] 
df = df[df['GROSS SQUARE FEET'] > 0]
df = df[df['YEAR BUILT'] > 1600] # Basic sanity check

# Feature Engineering
df['PRICE_PER_SQFT'] = df['SALE PRICE'] / df['GROSS SQUARE FEET']
df['BUILDING_AGE'] = 2026 - df['YEAR BUILT']

# Remove Outliers (99th percentile) to keep clusters stable
p99 = df['PRICE_PER_SQFT'].quantile(0.99)
df = df[df['PRICE_PER_SQFT'] <= p99]
df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'PRICE_PER_SQFT', 'BUILDING_AGE'])

print(f"Data Ready: {len(df)} records.")

# ==========================================
# 3. MACHINE LEARNING (K-MEANS)
# ==========================================
print("Running K-Means Clustering...")

# Features to cluster on
features = ['PRICE_PER_SQFT', 'BUILDING_AGE', 'LATITUDE', 'LONGITUDE']

# Normalize
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# K-Means (k=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster_id'] = kmeans.fit_predict(df_scaled)

# Create readable labels
df['Cluster Label'] = "Cluster " + df['cluster_id'].astype(str)

# Sort by cluster for consistent coloring
df = df.sort_values('cluster_id')

# ==========================================
# 4. OUTPUT 1: THE HIDDEN MAP
# ==========================================
print("Generating Output 1: nyc_cluster_map.html...")

fig_map = px.scatter_mapbox(
    df,
    lat="LATITUDE",
    lon="LONGITUDE",
    color="Cluster Label",
    hover_name="BOROUGH",
    hover_data={
        "LATITUDE": False, 
        "LONGITUDE": False,
        "Cluster Label": True,
        "PRICE_PER_SQFT": ":$.0f",
        "BUILDING_AGE": "0 years",
        "SALE PRICE": ":$,.0f"
    },
    color_discrete_sequence=px.colors.qualitative.Bold, # High contrast
    zoom=10,
    height=800,
    opacity=0.6,
    title="NYC Economic Zones: Detected via Machine Learning (K-Means)"
)

fig_map.update_layout(
    mapbox_style="carto-darkmatter",
    margin={"r":0,"t":40,"l":0,"b":0},
    paper_bgcolor="black",
    font_color="#00ffcc"
)

fig_map.write_html("nyc_cluster_map.html")
print("Saved nyc_cluster_map.html")

# ==========================================
# 5. OUTPUT 2: CLUSTER DNA ANALYSIS
# ==========================================
print("Generating Output 2: nyc_cluster_analysis.html...")

# Aggregate data for Parcoords (Mean profile of each cluster)
cluster_profile = df.groupby("cluster_id")[features].mean().reset_index()
cluster_profile['Cluster Label'] = "Cluster " + cluster_profile['cluster_id'].astype(str)

# We need to map colors to expected range. 
# Parcoords is tricky with categorical colors. 
# Easier to plot ALL points (sampled) to show density, 
# or plot the centroids. 
# Let's plot a Sample of 2000 points to show the distribution structure.
sample_df = df.sample(min(2000, len(df)), random_state=42).sort_values('cluster_id')

fig_par = px.parallel_coordinates(
    sample_df,
    color="cluster_id",
    dimensions=['PRICE_PER_SQFT', 'BUILDING_AGE', 'LATITUDE', 'LONGITUDE'],
    color_continuous_scale=px.colors.diverging.Tealrose, # Cyberpunk-ish
    title="Cluster DNA: What defines an Economic Zone?"
)

fig_par.update_layout(
    template="plotly_dark",
    paper_bgcolor="black",
    font_color="#00ffcc",
    margin={"r":50,"t":80,"l":50,"b":50}
)

# Fix Colorbar to show Cluster IDs integer
fig_par.update_coloraxes(colorbar=dict(title="Cluster ID", tickvals=[0,1,2,3,4]))

fig_par.write_html("nyc_cluster_analysis.html")
print("Saved nyc_cluster_analysis.html")

print("Analysis Complete.")
