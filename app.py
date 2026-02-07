import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ==========================================
# 1. PAGE CONFIGURATION (Cyberpunk Theme)
# ==========================================
st.set_page_config(
    page_title="NYC Market Force-Field",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Pitch Black" Aesthetic
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #00ffcc !important;
        font-family: 'Courier New', monospace;
    }
    div[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #333;
    }
    .caption {
        font-size: 1.2em;
        color: #ff00ff;
        font-style: italic;
        border-bottom: 1px solid #ff00ff;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & CLEANING
# ==========================================
@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv("nyc_housing_base.csv")
    
    # Cleaning Logic (Same as Cyberpunk Map)
    df = df[df['SALE PRICE'] > 100_000] # Remove low-end noise
    df = df[df['GROSS SQUARE FEET'] > 0]
    
    # Feature Engineering
    df['PRICE_PER_SQFT'] = df['SALE PRICE'] / df['GROSS SQUARE FEET']
    
    # Remove extreme outliers for grid stability
    p99 = df['PRICE_PER_SQFT'].quantile(0.99)
    df = df[df['PRICE_PER_SQFT'] <= p99]
    
    # Coordinate Cleaning
    if 'LATITUDE' not in df.columns:
        # Fallback if names differ (check your CSV)
        pass 
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'PRICE_PER_SQFT'])
    
    return df

df = load_data()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("Market Physics Engine")
st.sidebar.markdown("---")

# Borough Filter
boroughs = ['All'] + sorted(df['BOROUGH'].unique().tolist())
selected_borough = st.sidebar.selectbox("Select Zone (Borough)", boroughs)

# Grid Resolution (Smoothing)
grid_res = st.sidebar.slider("Interpolation Grid Resolution", 50, 300, 100, step=10)

# ==========================================
# 4. PHYSICS ENGINE (Interpolation & Gradient)
# ==========================================
# Filter Data
if selected_borough != 'All':
    filtered_df = df[df['BOROUGH'] == selected_borough]
else:
    filtered_df = df

# Narrative
st.title("NYC Housing: The Gravity of Wealth")
st.markdown('<div class="caption">"Housing value does not diffuse; it concentrates. This Force-Field map shows how high-density zones act as Gravity Wells, pulling market value inward rather than lowering costs for the surrounding area."</div>', unsafe_allow_html=True)

if filtered_df.empty:
    st.error("No data available for this selection.")
    st.stop()

# 1. Create Meshgrid
x = filtered_df['LONGITUDE'].values
y = filtered_df['LATITUDE'].values
z = filtered_df['PRICE_PER_SQFT'].values

# Define grid boundaries
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

# Log-scale Z for better gradient visualization (Wealth scales exponentially)
z_log = np.log1p(z)

# Generate Grid
grid_x, grid_y = np.mgrid[x_min:x_max:complex(0, grid_res), y_min:y_max:complex(0, grid_res)]

# 2. Interpolate Surface (Price Field)
# 'linear' is sharper, 'cubic' is smoother. Linear often looks more "structural".
grid_z = griddata((x, y), z_log, (grid_x, grid_y), method='linear')

# Fill NaNs with mean (or min) to prevent holes, but 0 makes edges weird.
# Strategy: Fill with local min to represent "voids"
grid_z = np.nan_to_num(grid_z, nan=np.nanmin(z_log))

# 3. Calculate Gradient (Market Force Vectors)
# V (dy), U (dx) -- Numpy gradient returns (gradient along axis 0, gradient along axis 1)
dy, dx = np.gradient(grid_z)
# Setup U, V mapping (U is dx, V is dy)
U = dx
V = dy

# Calculate Magnitude for coloring
speed = np.sqrt(U**2 + V**2)

# ==========================================
# 5. VISUALIZATION (Streamplot)
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
ax.set_facecolor('black')

# Plot Streamlines
# cmap='inferno' or 'plasma' for Neon
strm = ax.streamplot(
    grid_x[:, 0], grid_y[0, :], U, V,
    color=speed,
    linewidth=1.5 + (2 * speed / speed.max()), # Thicker lines for stronger forces
    cmap='plasma',
    density=2, # Density of lines
    arrowstyle='->',
    arrowsize=1.5
)

# Overlay High Value "Gravity Centers" (Top 5% in view)
high_val = filtered_df[filtered_df['PRICE_PER_SQFT'] > filtered_df['PRICE_PER_SQFT'].quantile(0.95)]
ax.scatter(high_val['LONGITUDE'], high_val['LATITUDE'], 
           color='white', alpha=0.3, s=10, marker='*', label='Gravity Centers (Top 5%)')

# Aesthetics
ax.set_title(f"Market Force Vector Field: {selected_borough}", color='#00ffcc', fontsize=16, pad=20)
ax.axis('off') # Hide axes for cleaner look

# Colorbar (Optional, but helps explain "Speed")
# cbar = plt.colorbar(strm.lines)
# cbar.set_label('Price Gradient Magnitude', color='white')
# cbar.ax.yaxis.set_tick_params(color='white')
# plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

st.pyplot(fig)

# Metrics
st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.metric("Max Price/SqFt", f"${filtered_df['PRICE_PER_SQFT'].max():,.0f}")
c2.metric("Median Price/SqFt", f"${filtered_df['PRICE_PER_SQFT'].median():,.0f}")
c3.metric("Data Points", len(filtered_df))
