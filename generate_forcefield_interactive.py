import pandas as pd
import numpy as np
import pydeck as pdk
from scipy.interpolate import RegularGridInterpolator
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
GRID_RES = 150 # Resolution of the vector field grid
NUM_PARTICLES = 6000 # Number of streamlines
STEPS = 20 # Length of streamlines
DT = 0.001 # Integration step size
OUTPUT_FILE = "force_field_interactive.html"

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_data():
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
    
    # Cleaning
    df = df[df['SALE PRICE'] > 100_000] 
    df = df[df['GROSS SQUARE FEET'] > 0]
    df['PRICE_PER_SQFT'] = df['SALE PRICE'] / df['GROSS SQUARE FEET']
    
    # Outliers
    p99 = df['PRICE_PER_SQFT'].quantile(0.99)
    df = df[df['PRICE_PER_SQFT'] <= p99]
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'PRICE_PER_SQFT'])
    
    return df

# ==========================================
# 3. PHYSICS ENGINE (Particle Tracer)
# ==========================================
def generate_streamlines(df):
    print("Generating Vector Field...")
    
    # 1. Create Grid
    x = df['LONGITUDE'].values
    y = df['LATITUDE'].values
    z = np.log1p(df['PRICE_PER_SQFT'].values) # Log scalar field
    
    # Define bounds with buffer
    pad = 0.02
    x_min, x_max = x.min() - pad, x.max() + pad
    y_min, y_max = y.min() - pad, y.max() + pad
    
    # Create regular grid
    xi = np.linspace(x_min, x_max, GRID_RES)
    yi = np.linspace(y_min, y_max, GRID_RES)
    
    # Interpolate Z onto grid (using linear binning or nearest for speed/robustness)
    from scipy.interpolate import griddata
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
    zi = np.nan_to_num(zi, nan=np.nanmin(z)) # Fill voids
    
    # Calculate Gradient (Vector Field)
    # Gradient of Price = Direction of Wealth Flow
    # np.gradient returns (d/axis0, d/axis1) -> (dy, dx) basically
    dy, dx = np.gradient(zi)
    
    # U (Longitude change), V (Latitude change)
    # We need to normalize to lat/lon scale approximately to make traces look circular
    # But sticking to index space is easier for interpolation? 
    # Let's keep in grid space for lookup, convert output to coord space.
    
    # Create Interpolator Function for U and V
    # RegularGridInterpolator expects (x, y) points
    # Note: RGI expects points in verifying order (y, x) if data is (ny, nx)
    rgi_u = RegularGridInterpolator((yi, xi), dx, bounds_error=False, fill_value=0)
    rgi_v = RegularGridInterpolator((yi, xi), dy, bounds_error=False, fill_value=0)
    
    print(f"Tracing {NUM_PARTICLES} particles...")
    
    # Random Seed Points
    seeds_x = np.random.uniform(x_min, x_max, NUM_PARTICLES)
    seeds_y = np.random.uniform(y_min, y_max, NUM_PARTICLES)
    
    paths = []
    colors = []
    
    for i in range(NUM_PARTICLES):
        cx, cy = seeds_x[i], seeds_y[i]
        path = [[cx, cy]]
        valid = True
        
        # Integration Loop
        for _ in range(STEPS):
            # Get velocity at current pos
            u = rgi_u((cy, cx)).item() # Note (y, x) order
            v = rgi_v((cy, cx)).item()
            
            vec_mag = np.sqrt(u**2 + v**2)
            if vec_mag < 0.0001: # Stagnation
                break
                
            # Step forward
            # dx is gradient on grid indices. We need to scale to world coords roughly
            # Just applying a factor helps
            ncx = cx + u * DT
            ncy = cy + v * DT
            
            # Check bounds
            if not (x_min <= ncx <= x_max and y_min <= ncy <= y_max):
                break
                
            path.append([ncx, ncy])
            cx, cy = ncx, ncy
            
        if len(path) > 2:
            paths.append(path)
            # Color by final velocity magnitude (Flow Intensity)
            u_final = rgi_u((cy, cx)).item()
            v_final = rgi_v((cy, cx)).item()
            speed = np.sqrt(u_final**2 + v_final**2)
            colors.append(speed)
            
    return paths, colors

# ==========================================
# 4. VISUALIZATION
# ==========================================
def create_visualization(df, paths, speeds):
    print("Configuring Deck.gl...")
    
    # Normalize speeds for coloring
    speeds = np.array(speeds)
    if len(speeds) > 0:
        s_min, s_max = np.percentile(speeds, 5), np.percentile(speeds, 95)
        # Avoid div by zero
        if s_max == s_min: s_max += 1
    else:
        s_min, s_max = 0, 1

    # Prepare Path Data
    # PathLayer expects a list of dictionaries with 'path' and 'color'
    path_data = []
    for p, s in zip(paths, speeds):
        # Color Map: Low=Teal, High=Magenta
        # Normalize s 0..1
        norm = np.clip((s - s_min) / (s_max - s_min), 0, 1)
        
        # Interpolate Color
        # Teal: [0, 255, 255], Magenta: [255, 0, 255]
        r = int(0 + norm * 255)
        g = int(255 - norm * 255)
        b = 255
        color = [r, g, b, 200]
        
        color = [r, g, b, 200]
        
        path_data.append({
            "path": p, 
            "color": color,
            "tooltip": "Capital Flow Streamline\n(Wealth Gradient)"
        })
        
    # Layer 1: Streamlines (Paths)
    layer_paths = pdk.Layer(
        "PathLayer",
        path_data,
        get_path="path",
        get_color="color",
        width_min_pixels=1.5,
        width_max_pixels=3,
        pickable=True
    )
    
    # Layer 2: Gravity Wells (Scatterplot)
    # Top 2%
    high_val = df[df['PRICE_PER_SQFT'] > df['PRICE_PER_SQFT'].quantile(0.98)].copy()
    high_val['tooltip'] = "Market Gravity Center\nPrice: $" + high_val['PRICE_PER_SQFT'].map('{:,.0f}'.format) + "/sqft"
    
    layer_scatter = pdk.Layer(
        "ScatterplotLayer",
        high_val,
        get_position=["LONGITUDE", "LATITUDE"],
        get_radius=30,
        get_fill_color=[255, 255, 255], # Pure White Stars
        pickable=True,
        opacity=0.8,
        stroked=False
    )
    
    # View State
    mean_lat = df['LATITUDE'].mean()
    mean_lon = df['LONGITUDE'].mean()
    view_state = pdk.ViewState(
        latitude=mean_lat,
        longitude=mean_lon,
        zoom=11,
        pitch=0, # Top down for vector field usually better, but let's do slight tilt
        bearing=0
    )
    
    # Render
    deck = pdk.Deck(
        layers=[layer_paths, layer_scatter],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={"text": "{tooltip}"}
    )
    
    # Generate HTML File directly
    deck.to_html(OUTPUT_FILE)
    return OUTPUT_FILE

# ==========================================
# 5. HUD INJECTION & SAVE
# ==========================================
def scale_and_save(output_file):
    print(f"Injecting HUD into {output_file}...")
    
    with open(output_file, "r") as f:
        html_str = f.read()
    
    custom_css = """
    <style>
        body { margin: 0; padding: 0; background-color: #000; }
        .hud {
            position: absolute; top: 20px; left: 20px;
            background: rgba(0,0,0,0.8); color: #00ffcc;
            padding: 15px; border: 1px solid #ff00ff;
            font-family: monospace; z-index: 1000;
            max-width: 300px;
        }
        .hud h2 { margin: 0 0 10px 0; color: #ff00ff; font-size: 18px; }
        .hud p { font-size: 12px; color: #fff; line-height: 1.4; }
        .legend-gradient {
            height: 10px; width: 100%;
            background: linear-gradient(to right, #00ffff, #ff00ff);
            margin-top: 5px;
        }
        .labels { display: flex; justify-content: space-between; font-size: 10px; color: #aaa; }
    </style>
    """
    
    custom_html = """
    <div class="hud">
        <h2>MARKET FORCE FIELD</h2>
        <p><strong>Streamlines:</strong> Visualize the uphill flow of capital. The "Current" moves towards higher prices.</p>
        <p><strong>White Stars:</strong> Gravity Wells (Top 2% Prices).</p>
        <div class="legend-gradient"></div>
        <div class="labels"><span>Low Gradient</span><span>Steep Price Rise</span></div>
    </div>
    """
    
    final_html = html_str.replace("</head>", f"{custom_css}</head>")
    final_html = final_html.replace("</body>", f"{custom_html}</body>")
    
    with open(output_file, "w") as f:
        f.write(final_html)
    print("Done.")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    df = load_data()
    paths, speeds = generate_streamlines(df)
    output_filename = create_visualization(df, paths, speeds)
    scale_and_save(output_filename)
