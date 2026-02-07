import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
OUTPUT_DIR = "force_field_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aesthetic Settings
plt.style.use('dark_background')
CMAP = 'plasma' # Neon Orange/Purple

# ==========================================
# 2. DATA LOADING & CLEANING
# ==========================================
def load_data():
    print("Loading Data...")
    df = pd.read_csv("nyc_housing_base.csv")
    
    # Column Mapping (Raw CSV has lowercase, mapped to standard upper)
    column_mapping = {
        'sale_price': 'SALE PRICE',
        'bldgarea': 'GROSS SQUARE FEET',
        'yearbuilt': 'YEAR BUILT',
        'latitude': 'LATITUDE',
        'longitude': 'LONGITUDE',
        'borough_y': 'BOROUGH'
    }
    df = df.rename(columns=column_mapping)
    
    # Cleaning Logic
    df = df[df['SALE PRICE'] > 100_000] 
    df = df[df['GROSS SQUARE FEET'] > 0]
    
    # Feature Engineering
    df['PRICE_PER_SQFT'] = df['SALE PRICE'] / df['GROSS SQUARE FEET']
    
    # Remove outliers for grid stability (99th percentile)
    p99 = df['PRICE_PER_SQFT'].quantile(0.99)
    df = df[df['PRICE_PER_SQFT'] <= p99]
    
    # Cleaning coordinates
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'PRICE_PER_SQFT'])
    print(f"Data Loaded: {len(df)} records.")
    return df

# ==========================================
# 3. VISUALIZATION ENGINE
# ==========================================
def generate_force_field(df, area_name, grid_res=200):
    print(f"Generating Force Field for: {area_name}...")
    
    # 1. Prepare Data
    x = df['LONGITUDE'].values
    y = df['LATITUDE'].values
    z = df['PRICE_PER_SQFT'].values
    
    # Log-scale Z for better gradient visualization
    z_log = np.log1p(z)
    
    # 2. Create Grid
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Add buffer to grid margins
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    
    grid_x, grid_y = np.mgrid[
        x_min-x_margin:x_max+x_margin:complex(0, grid_res), 
        y_min-y_margin:y_max+y_margin:complex(0, grid_res)
    ]
    
    # 3. Interpolate Surface
    grid_z = griddata((x, y), z_log, (grid_x, grid_y), method='linear')
    
    # Fill NaNs with local min (voids)
    grid_z = np.nan_to_num(grid_z, nan=np.nanmin(z_log))
    
    # 4. Calculate Gradient (Vectors)
    dy, dx = np.gradient(grid_z)
    U, V = dx, dy
    speed = np.sqrt(U**2 + V**2)
    
    # 5. Plot
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='black')
    ax.set_facecolor('black')
    
    # Streamplot
    # Vary linewidth by speed
    lw = 1.0 + (3 * speed / speed.max())
    
    strm = ax.streamplot(
        grid_x[:, 0], grid_y[0, :], U, V,
        color=speed,
        linewidth=lw,
        cmap=CMAP,
        density=2.5, # High density for detail
        arrowstyle='->',
        arrowsize=2.5
    )
    
    # Overlay "Gravity Stars" (Top 2% in this area)
    threshold = df['PRICE_PER_SQFT'].quantile(0.98)
    high_val = df[df['PRICE_PER_SQFT'] > threshold]
    
    ax.scatter(high_val['LONGITUDE'], high_val['LATITUDE'], 
               color='white', alpha=0.6, s=15, marker='*', 
               label='Gravity Centers (Top 2%)')
    
    # Annotations
    ax.set_title(f"MARKET FORCE FIELD: {area_name.upper()}", 
                 color='#00ffcc', fontsize=24, fontname='Courier New', pad=20, weight='bold')
    
    # Caption
    caption = "Streamlines show the 'uphill' flow of wealth concentration."
    fig.text(0.5, 0.05, caption, ha='center', color='#ff00ff', fontsize=12, style='italic', fontname='Courier New')
    
    ax.axis('off')
    
    # Save
    filename = f"{OUTPUT_DIR}/force_field_{area_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved: {filename}")

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    df = load_data()
    
    # 1. Generate FULL NYC Map
    generate_force_field(df, "New York City", grid_res=300)
    
    # 2. Generate Map for EACH Borough
    boroughs = df['BOROUGH'].unique()
    for borough in boroughs:
        b_df = df[df['BOROUGH'] == borough]
        if len(b_df) > 50: # Only generate if enough data
            generate_force_field(b_df, borough)
            
    print("All visualizations generated.")
