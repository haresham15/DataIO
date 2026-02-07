import pandas as pd
import pydeck as pdk
import os

# ==========================================
# 1. LOAD DATA
# ==========================================
# Try loading the requested filename, fallback to existing file if necessary
filename = 'nyc-housing-prices.csv'
if not os.path.exists(filename):
    if os.path.exists('nyc_housing_base.csv'):
        print(f"'{filename}' not found. Using 'nyc_housing_base.csv' instead.")
        filename = 'nyc_housing_base.csv'
    else:
        raise FileNotFoundError(f"Could not find '{filename}' or 'nyc_housing_base.csv'. Please ensure the dataset is in the directory.")

print(f"Loading data from {filename}...")
df = pd.read_csv(filename)

# ==========================================
# 2. DATA CLEANING & PREPROCESSING
# ==========================================
print("Cleaning data...")

# Column Mapping
column_mapping = {
    'sale_price': 'SALE PRICE',
    'bldgarea': 'GROSS SQUARE FEET',
    'yearbuilt': 'YEAR BUILT',
    'latitude': 'LATITUDE',
    'longitude': 'LONGITUDE',
    'borough_y': 'BOROUGH_CODE',
    'building_age': 'BUILDING_AGE',
    'bldgclass': 'BUILDING CLASS CATEGORY' 
}

# Rename columns
df = df.rename(columns=column_mapping)

# Ensure numeric columns 
numeric_cols = ['SALE PRICE', 'GROSS SQUARE FEET', 'YEAR BUILT', 'LATITUDE', 'LONGITUDE']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop N/A
df = df.dropna(subset=numeric_cols)

# SALE_YEAR Derivation
if 'BUILDING_AGE' in df.columns:
    df['SALE_YEAR'] = df['YEAR BUILT'] + df['BUILDING_AGE']
    df['SALE_YEAR'] = df['SALE_YEAR'].fillna(2025).astype(int)
else:
    df['SALE_YEAR'] = 2025 

# Strict Filtering
df = df[(df['SALE PRICE'] >= 100000) & 
        (df['GROSS SQUARE FEET'] > 0) &
        (df['LATITUDE'] != 0) & (df['LONGITUDE'] != 0)]

# Neighborhood creation
if 'NEIGHBORHOOD' not in df.columns:
    if 'BOROUGH_CODE' in df.columns:
        borough_map = {
            'MN': 'Manhattan', 'BK': 'Brooklyn', 'QN': 'Queens', 'BX': 'Bronx', 'SI': 'Staten Island'
        }
        df['NEIGHBORHOOD'] = df['BOROUGH_CODE'].map(borough_map).fillna('NYC')
    else:
        df['NEIGHBORHOOD'] = 'NYC'

if 'BUILDING CLASS CATEGORY' not in df.columns:
    df['BUILDING CLASS CATEGORY'] = 'Unknown'

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
print("Feature Engineering...")
df['PRICE_PER_SQFT'] = df['SALE PRICE'] / df['GROSS SQUARE FEET']

# Cap outliers for better scaling
df = df[df['PRICE_PER_SQFT'] < 100000]

# CRITICAL FIX: Ensure Types
df['LATITUDE'] = df['LATITUDE'].astype(float)
df['LONGITUDE'] = df['LONGITUDE'].astype(float)
df['PRICE_PER_SQFT'] = df['PRICE_PER_SQFT'].astype(float)
df['YEAR BUILT'] = df['YEAR BUILT'].astype(float)

# High-Value threshold for Scatterplot 
high_value_df = df[df['SALE PRICE'] > 3_000_000].copy()

# ------------------------------------------
# CRITICAL FIX: Schema Alignment
# ------------------------------------------
# The HexagonLayer produces 'elevationValue' and 'colorValue' automatically.
# The ScatterplotLayer uses raw data. To use the SAME tooltip for both,
# we artificially create these columns in the scatter data.
high_value_df['elevationValue'] = high_value_df['PRICE_PER_SQFT']
high_value_df['colorValue'] = high_value_df['YEAR BUILT']

# ==========================================
# 4. ANALYTICAL CLAMPING & DOMAIN
# ==========================================
# Argument: "Clamp at 90th-95th percentile to prevent outliers from dominating"
price_p95 = df['PRICE_PER_SQFT'].quantile(0.95)
print(f"95th Percentile Price/SqFt: ${price_p95:.2f} (Clamping Limit)")

# ==========================================
# 5. CYBERPUNK CONFIGURATION (Inequality Theme)
# ==========================================
# Argument: "Low pressure = Dim Cyan/Teal, High pressure = Bright Magenta/Purple"
INEQUALITY_COLOR_RANGE = [
    [0, 80, 80],      # Brighter Teal (Easier to see)
    [0, 120, 120],    # Dim Cyan
    [0, 180, 180],    # Bright Cyan (Mid Pressure)
    [120, 0, 120],    # Deep Purple
    [255, 0, 255],    # Neon Magenta (High Pressure)
    [255, 255, 255]   # Pure White (Extreme Pressure)
]

# ==========================================
# 6. PYDECK LAYERS
# ==========================================
print("Configuring Layers with BOOSTED VISIBILITY...")

# ----------------------------
# Layer 1: Heatmap (The Systemic Glow)
# ----------------------------
heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    df,
    get_position=["LONGITUDE", "LATITUDE"],
    opacity=0.3, # Reduced opacity to not wash out the map
    aggregation="MEAN", # Changed to MEAN to show pressure, not volume
    get_weight="PRICE_PER_SQFT", 
    radius_pixels=80, 
    intensity=1.0, # Reduced intensity
    threshold=0.1,
    color_range=[
        [0, 0, 0, 0],
        [0, 50, 50, 100],   # Faint Teal
        [0, 150, 150, 150], # Cyan
        [150, 0, 150, 200]  # Purple
    ]
)

# ----------------------------
# Layer 2: Hexagon (The Structure)
# ----------------------------
# Argument: "Encode price per sqft as heat intensity"
hexagon_layer = pdk.Layer(
    "HexagonLayer",
    df,
    get_position=["LONGITUDE", "LATITUDE"],
    auto_highlight=True,
    elevation_scale=1.0, # FLATTER TERRAIN (Easier to read)
    pickable=True,
    elevation_range=[0, 3000],
    extruded=True,
    coverage=0.95, # Almost solid blocks
    radius=500,   # WIDER Districts (500m) for map-like legibility

    # METRICS: PRICE PRESSURE ONLY
    get_elevation_weight="PRICE_PER_SQFT",
    elevation_aggregation="MEAN",
    elevation_domain=[0, price_p95], # CLAMPED

    get_color_weight="PRICE_PER_SQFT", # COLOR BY PRICE
    color_aggregation="MEAN",
    color_domain=[0, price_p95], # CLAMPED
    color_range=INEQUALITY_COLOR_RANGE,
    
    material={
        "ambient": 0.2, # Darker, more oppressive
        "diffuse": 0.8,
        "shininess": 10,
        "specularColor": [100, 100, 100]
    }
)

# ----------------------------
# Layer 3: Scatterplot (The Spikes)
# ----------------------------
scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    high_value_df,
    get_position=["LONGITUDE", "LATITUDE"],
    get_radius=25, # Reduced from 80 to 25 to prevent overlap "blobs"
    get_fill_color=[255, 255, 255],
    get_line_color=[0, 255, 255],
    pickable=True, # Re-enabled now that schema matches!
    stroked=True,
    filled=True,
    line_width_min_pixels=2,
    opacity=0.9
)

# ----------------------------
# RENDER
# ----------------------------

# Calculate view state dynamically
mean_lat = df['LATITUDE'].mean()
mean_lon = df['LONGITUDE'].mean()

view_state = pdk.ViewState(
    latitude=mean_lat,  
    longitude=mean_lon,
    zoom=11, 
    pitch=30, # Lower angle (30 deg) prevents occlusion
    bearing=0
)

# Updated Tooltip for Analytical Argument
tooltip = {
    "html": "<div style='font-family: Arial, sans-serif; padding: 10px; background: rgba(10, 10, 10, 0.95); border: 1px solid #ff00ff; border-radius: 0px; color: white;'>"
            "<h4 style='margin:0; border-bottom: 2px solid #00ffcc; padding-bottom:5px; color: #00ffcc; text-transform: uppercase;'>Structural Inequality</h4>"
            "<span style='color: #ff00ff;'>Price Pressure (PPSF):</span> <b>${elevationValue}</b><br/>"
            "<span style='font-size: 0.8em; color: gray;'>Relative Cost normalized by size</span>"
            "</div>",
    "style": {
        "backgroundColor": "transparent"
    }
}

deck = pdk.Deck(
    layers=[heatmap_layer, hexagon_layer, scatter_layer], 
    initial_view_state=view_state,
    map_style=pdk.map_styles.DARK,
    tooltip=tooltip
)

# ==========================================
# 8. EXPORT WITH CUSTOM HUD
# ==========================================
output_file = "cyberpunk_nyc.html"
print(f"Generating {output_file}...")

# 1. Generate Raw HTML File
deck.to_html(output_file)

# 2. Read the Generated File
with open(output_file, "r") as f:
    html_content = f.read()

# 3. Define Custom CSS for the HUD
custom_css = """
<style>
    body { margin: 0; padding: 0; overflow: hidden; background-color: #000; }
    #deckgl-overlay { mix-blend-mode: hard-light; } /* Optional blend mode tweak */
    
    .cyber-hud {
        position: absolute;
        bottom: 30px;
        right: 30px;
        width: 250px;
        background: rgba(10, 10, 15, 0.85);
        border: 1px solid #00ffcc;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
        padding: 15px;
        font-family: 'Courier New', monospace;
        color: #e0e0e0;
        z-index: 9999; /* Ensure it's on top */
        pointer-events: none; /* Let clicks pass through */
        border-radius: 4px;
    }
    .cyber-hud h3 {
        margin: 0 0 10px 0;
        color: #ff00ff;
        text-transform: uppercase;
        font-size: 14px;
        border-bottom: 1px solid #ff00ff;
        padding-bottom: 5px;
        text-shadow: 0 0 5px #ff00ff;
    }
    .control-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
        font-size: 12px;
    }
    .key { color: #00ffcc; font-weight: bold; }
    .action { text-align: right; }
</style>
"""

# 4. Define Custom HTML for the HUD
custom_html = """
<div class="cyber-hud">
    <h3>System Controls</h3>
    <div class="control-row">
        <span class="key">Rotate / Tilt</span>
        <span class="action">Shift + Drag</span>
    </div>
    <div class="control-row">
        <span class="key">Pan</span>
        <span class="action">Left Click + Drag</span>
    </div>
    <div class="control-row">
        <span class="key">Zoom</span>
        <span class="action">Scroll Wheel</span>
    </div>
    <div style="margin-top: 10px; font-size: 10px; color: #666; text-align: center;">
        <span style="color: #ff00ff;">&bull;</span> DATA VZ 2024 <span style="color: #00ffcc;">&bull;</span>
    </div>
</div>
"""

# 5. Inject into the Raw HTML
# Inject CSS before </head>
html_content = html_content.replace("</head>", f"{custom_css}</head>")
# Inject HTML before </body>
html_content = html_content.replace("</body>", f"{custom_html}</body>")

# 6. Save Modified HTML
with open(output_file, "w") as f:
    f.write(html_content)

print("Done! Visualization generated with Navigation HUD.")
