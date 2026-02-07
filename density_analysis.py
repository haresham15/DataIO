import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Settings
# -----------------------------
INPUT_PATH = "nyc_housing_cleaned.csv"
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv(INPUT_PATH)

# -----------------------------
# 2. Basic filtering
# -----------------------------
required_cols = ["sale_price", "bldgarea", "resarea", "unitsres", "borough", "zip_code"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[
    (df["sale_price"] > 0) &
    (df["bldgarea"] > 0) &
    (df["resarea"] > 0) &
    (df["unitsres"] > 0)
].copy()

# -----------------------------
# 3. Derived metrics
# -----------------------------
df["resident_density"] = df["unitsres"] / df["resarea"]
df["price_per_sqft"] = df["sale_price"] / df["bldgarea"]

# -----------------------------
# 4. Clamp outliers (95th percentile)
# -----------------------------
caps = {
    "resident_density": df["resident_density"].quantile(0.95),
    "price_per_sqft": df["price_per_sqft"].quantile(0.95),
    "sale_price": df["sale_price"].quantile(0.95),
    "bldgarea": df["bldgarea"].quantile(0.95),
}

df = df[
    (df["resident_density"] <= caps["resident_density"]) &
    (df["price_per_sqft"] <= caps["price_per_sqft"]) &
    (df["sale_price"] <= caps["sale_price"]) &
    (df["bldgarea"] <= caps["bldgarea"])
].copy()

# -----------------------------
# 5. Correlations (overall)
# -----------------------------
pearson = df["resident_density"].corr(df["price_per_sqft"], method="pearson")
spearman = df["resident_density"].corr(df["price_per_sqft"], method="spearman")

print("Overall correlations (density vs price_per_sqft)")
print("Pearson:", pearson)
print("Spearman:", spearman)

# -----------------------------
# 6. Correlations by borough
# -----------------------------
borough_corr = (
    df.groupby("borough")
      .apply(lambda x: pd.Series({
          "n": len(x),
          "pearson": x["resident_density"].corr(x["price_per_sqft"], method="pearson"),
          "spearman": x["resident_density"].corr(x["price_per_sqft"], method="spearman"),
      }))
      .reset_index()
      .sort_values("n", ascending=False)
)

print("\nBy-borough correlations (density vs price_per_sqft)")
print(borough_corr.to_string(index=False))

borough_corr.to_csv("borough_correlations.csv", index=False)

# -----------------------------
# 7. Visualizations (save + close; no blocking windows)
# -----------------------------
boroughs = sorted(df["borough"].dropna().unique())

# 7A) Total price vs building area (log-log), per borough
for borough in boroughs:
    sub = df[df["borough"] == borough]
    if len(sub) < 30:
        continue

    plt.figure(figsize=(6, 5))
    plt.scatter(sub["bldgarea"], sub["sale_price"], s=8, alpha=0.4)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Building Area (sq ft, log scale)")
    plt.ylabel("Total Sale Price ($, log scale)")
    plt.title(f"Total Sale Price vs Building Area ({borough})")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"{borough}_price_vs_area.png", dpi=200)
    plt.close()

# 7B) Density vs price per sqft, per borough
for borough in boroughs:
    sub = df[df["borough"] == borough]
    if len(sub) < 30:
        continue

    plt.figure(figsize=(6, 5))
    plt.scatter(sub["resident_density"], sub["price_per_sqft"], s=8, alpha=0.4)

    plt.xlabel("Residential Density (units / residential area)")
    plt.ylabel("Price per Square Foot ($)")
    plt.title(f"Residential Density vs Price per Sq Ft ({borough})")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"{borough}_density_vs_price.png", dpi=200)
    plt.close()

# -----------------------------
# 8. Export visualization-ready CSV
# -----------------------------
df[[
    "zip_code", "borough",
    "resident_density", "price_per_sqft",
    "bldgarea", "sale_price",
    "resarea", "unitsres"
]].to_csv("density_vs_price_viz_ready.csv", index=False)

print("\nSaved:")
print("- borough_correlations.csv")
print("- density_vs_price_viz_ready.csv")
print(f"- plots/ (PNG files per borough) -> {OUTPUT_DIR.resolve()}")
