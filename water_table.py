# water_table.py
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import warnings
import html
from datetime import datetime
import json
import requests # <--- PATCH: ADDED
import io # <--- PATCH: ADDED
warnings.filterwarnings('ignore')

print("=== COLCHESTER WELL DRYING RISK ANALYSIS ===")
print("GeoPandas-enhanced version — HTML report output\n")

# ---------------------------
# Helper / config
# ---------------------------
well_files = ["well_logs_with_coords.csv", "well_logs.csv", "wells.csv"]
obs_file = "obs_well_timeseries.csv"
bedrock_shp = "h428nsgb.shp"
surficial_shp = "h428nsgs.shp"
output_csv = "colchester_well_risk_analysis.csv"
output_html = "risk_report.html"

# If your UTM zone is different, change this. EPSG:26920 is NAD83 / UTM zone 20N (Nova Scotia).
utm_crs = "EPSG:26920"
wgs84 = "EPSG:4326"

# Pareto Principle Addition: Drought Stress Test Configuration
# Assumes a 2.0 meter worst-case drawdown due to drought for wells without detailed time series.
DROUGHT_DRAWDOWN_M = 2.0 # <--- PATCH: ADDED

# ECCC Hydrometric Data Integration Configuration (Optional, but highly recommended)
# Station ID for Stewiacke River at Newton Mills (NS) - reliable flow data
# WSC_STATION_ID = "01DH005"  # <-- Previous station ID (not found in file)
# ** FINAL PATCH: Using Salmon River near Truro (01EO001) as the secondary, more central station **
WSC_STATION_ID = "01EO001" 
# ** PATCH: Use the stable NS_daily_hydrometric.csv file with the correct path **
WSC_API_URL = "https://dd.weather.gc.ca/hydrometric/csv/NS/daily/NS_daily_hydrometric.csv"
# ---------------------------
# 1. Load well logs (try a few filenames)
# ---------------------------
print("Loading Colchester well logs...")
wells = None
for fname in well_files:
    try:
        wells = pd.read_csv(fname)
        print(f"Loaded: {fname}")
        break
    except Exception as e:
        # continue trying
        continue

if wells is None:
    raise FileNotFoundError("No well logs file found in working directory. Place a CSV named one of: " + ", ".join(well_files))

# Normalize column names
wells.columns = [str(c).upper().strip() for c in wells.columns]

# Map common schema variations to standard names
col_map = {
    # depth
    "TOTALORFINISHEDDEPTH": "DEPTH",
    "DEPTH": "DEPTH",
    "WYDEPTHENDOFTEST": "DEPTH",
    # static water level
    "WYSTATICLEVEL": "STATIC_WATER_LEVEL",
    "STATIC_WATER_LEVEL": "STATIC_WATER_LEVEL",
    "WYDEPTHTOWATERBEFOREPUMP": "STATIC_WATER_LEVEL",
    "WYDEPTHTOWATERAFTERPUMP": "STATIC_WATER_LEVEL",
    # yield
    "WYESTIMATEDYIELD": "YIELD",
    "WYRATE": "YIELD",
    "YIELD": "YIELD",
    # county
    "COUNTYL": "COUNTY",
    "COUNTY": "COUNTY",
    # coords/easting northing
    "EASTING": "X",
    "NORTHING": "Y",
    "X": "X",
    "Y": "Y",
    # alternative id
    "WELLNUMBER": "WELL_ID",
    "WELL_NO": "WELL_ID",
    "WELL_ID": "WELL_ID",
    # owner/driller
    "DRILLERSNAME": "DRILLER",
    "DRILLERCOMPANY": "DRILLER",
    # civic address
    "CIVICADDRESS": "CIVIC_ADDRESS",
    "ADDRESS": "CIVIC_ADDRESS",
    "LOCATION": "CIVIC_ADDRESS",
    "STREET": "CIVIC_ADDRESS",
    "MUNICIPALITY": "MUNICIPALITY",
    "COMMUNITY": "MUNICIPALITY",
}

# Apply mapping for any matching columns
available_cols = set(wells.columns)
rename_map = {old: new for old, new in col_map.items() if old in available_cols and new not in wells.columns}
if rename_map:
    wells = wells.rename(columns=rename_map)
    print(f"Normalized columns: {rename_map}")

# Check for duplicate column names after renaming
if wells.columns.duplicated().any():
    print("Warning: duplicate column names detected after renaming. Resolving...")

    # For each duplicated column name, collapse into one
    for col in wells.columns[wells.columns.duplicated()].unique():
        dupes = wells.loc[:, wells.columns == col]
        # collapse row-wise: take first non-null across duplicate columns
        wells[col] = dupes.bfill(axis=1).iloc[:, 0]
        # drop the extra duplicate columns (keep one)
        wells = wells.loc[:, ~wells.columns.duplicated()]

# Try to filter to Colchester if a COUNTY-like column exists
if "COUNTY" in wells.columns:
    try:
        mask = wells["COUNTY"].astype(str).str.lower() == "colchester"
        if mask.any():
            wells = wells[mask].copy()
            print(f"Filtered to Colchester County wells: {len(wells)}")
        else:
            print("COUNTY column present but no rows equal 'Colchester' (case-insensitive); analyzing all rows.")
    except Exception:
        print("COUNTY column present but could not filter; analyzing all rows.")
else:
    print("Warning: COUNTY column not found — analyzing all wells")

# ---------------------------
# 2. Required columns check
# ---------------------------
# After mapping, check for required fields and show helpful suggestions if missing
required = ["DEPTH", "STATIC_WATER_LEVEL"]
missing = [c for c in required if c not in wells.columns]
if missing:
    print(f"ERROR: Missing required columns: {missing}")
    print("Available columns (sample):", list(wells.columns)[:40])
    # Try to suggest likely column names
    suggestions = {}
    for want in missing:
        # look for near matches
        for col in wells.columns:
            if want.split("_")[0] in col or col in want:
                suggestions.setdefault(want, []).append(col)
    if suggestions:
        print("Possible matching columns:", suggestions)
    raise SystemExit("Please rename or map your well log columns so DEPTH and STATIC_WATER_LEVEL exist (or use the mapping table in the script).")

# ---------------------------
# 3. Observation wells (if available)
# ---------------------------
updated_count = 0
try:
    obs = pd.read_csv(obs_file, parse_dates=["date"])
    print(f"Loaded observation timeseries: {obs_file}")
    # try to find well id column in obs
    obs_id_cols = [c for c in obs.columns if c.upper() in ("WELL_ID", "WELLNUMBER", "WELL_NUMBER", "WELLNO")]
    if not obs_id_cols:
        raise ValueError("No WELL ID column found in observation file")
    obs_id_col = obs_id_cols[0]
    # try to find water level column
    wl_cols = [c for c in obs.columns if "water" in c.lower() and ("level" in c.lower() or "m" in c.lower())]
    if not wl_cols:
        raise ValueError("No water level column found in observation file")
    wl_col = wl_cols[0]
    # get last reading per well
    current_levels = obs.sort_values("date").groupby(obs_id_col)[wl_col].last().reset_index()
    current_levels.columns = ["WELL_ID", "water_level_m"]
    # normalize wells WELL_ID to string for matching
    if "WELL_ID" in wells.columns:
        wells["WELL_ID"] = wells["WELL_ID"].astype(str)
    else:
        # create WELL_ID from WELLNUMBER if present
        if "WELLNUMBER" in wells.columns:
            wells["WELL_ID"] = wells["WELLNUMBER"].astype(str)
        elif "WELL_ID" not in wells.columns:
            wells["WELL_ID"] = wells.index.astype(str)

    # assign static level to current_water_level_m
    if "STATIC_WATER_LEVEL" in wells.columns:
        wells["current_water_level_m"] = pd.to_numeric(
            wells["STATIC_WATER_LEVEL"].squeeze(), errors="coerce"
        )
    else:
        wells["current_water_level_m"] = np.nan

    # try to update from current_levels - matching as strings
    current_levels["WELL_ID"] = current_levels["WELL_ID"].astype(str)
    for _, r in current_levels.iterrows():
        wid = r["WELL_ID"]
        mask = wells["WELL_ID"].astype(str) == wid
        if mask.any():
            wells.loc[mask, "current_water_level_m"] = pd.to_numeric(r["water_level_m"], errors="coerce")
            updated_count += mask.sum()
    print(f"Updated {updated_count} well rows with observation current levels (matches by WELL_ID)")
except FileNotFoundError:
    print("Observation file not found — using static water levels only")
    # assign static level to current_water_level_m
    if "STATIC_WATER_LEVEL" in wells.columns:
        wells["current_water_level_m"] = pd.to_numeric(
            wells["STATIC_WATER_LEVEL"].squeeze(), errors="coerce"
        )
    else:
        wells["current_water_level_m"] = np.nan
except Exception as e:
    print(f"Warning: could not load/parse observation data ({e}) — using static levels only")
    # assign static level to current_water_level_m
    if "STATIC_WATER_LEVEL" in wells.columns:
        wells["current_water_level_m"] = pd.to_numeric(
            wells["STATIC_WATER_LEVEL"].squeeze(), errors="coerce"
        )
    else:
        wells["current_water_level_m"] = np.nan

# ---------------------------
# 4. Pump depth estimate & buffer
# ---------------------------
def estimate_pump_depth(depth):
    try:
        depth = float(depth)
        return min(depth * 0.8, depth - 2.5)
    except Exception:
        return np.nan

wells["DEPTH"] = pd.to_numeric(wells["DEPTH"], errors="coerce")
wells["pump_depth_m"] = wells["DEPTH"].apply(estimate_pump_depth)
wells["buffer_m"] = wells["pump_depth_m"] - wells["current_water_level_m"]

def classify_risk(buffer):
    if pd.isna(buffer):
        return "No data"
    if buffer < 0:
        return "CRITICAL - Well may be dry"
    if buffer < 2:
        return "High risk - <2m buffer"
    if buffer < 5:
        return "Moderate risk - 2-5m buffer"
    return "Low risk - >5m buffer"

wells["drying_risk"] = wells["buffer_m"].apply(classify_risk)

# ---------------------------
# Helper Function: WSC Data Fetcher
# ---------------------------
def fetch_real_time_discharge(station_id, api_url):
    """Fetches the current discharge (flow) from the WSC API using the provincial daily file."""
    try:
        url = api_url
        
        print(f"Attempting to fetch provincial daily summary file from: {url}")
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error fetching WSC provincial data (Status: {response.status_code}).")
            return None

        # This CSV format has a clean header at row 0
        data = pd.read_csv(io.StringIO(response.text))
        
        # ** FINAL PATCH: Use the exact, messy column names from the error report **
        # The station ID column has a leading space
        station_id_col = ' ID' 
        
        # The Discharge column has French characters (likely corrupted on download)
        discharge_col = None
        for col in data.columns:
            if 'Discharge' in col and '(cms)' in col:
                discharge_col = col
                break
        
        if station_id_col not in data.columns or discharge_col is None:
            print(f"Error: Could not find required columns in the WSC CSV. ID column: {' ID' in data.columns}, Discharge column: {discharge_col}")
            return None

        # 1. Filter by the desired station (ID)
        station_data = data[data[station_id_col] == station_id]
        
        if station_data.empty:
            print(f"Station {station_id} not found in the latest provincial daily data file.")
            return None

        # 2. Extract the Flow value from the Discharge column
        # Note: Daily file does not have a 'TYPE' column, so we just use the discharge column
        
        # The last row should be the most recent day
        latest_flow = pd.to_numeric(station_data[discharge_col], errors='coerce').dropna().iloc[-1]
        
        # The value is the daily mean flow in m^3/s (cms = cubic meters per second).
        print(f"Successfully retrieved latest daily mean flow for station {station_id}: {latest_flow} m³/s")
        return latest_flow

    except Exception as e:
        print(f"Error processing WSC data: {e}. Check API URL and Station ID.")
        return None

# ---------------------------
# 5. Yield adjustment if present (ORIGINALLY 5)
# ---------------------------
if "YIELD" in wells.columns:
    wells["YIELD"] = pd.to_numeric(wells["YIELD"], errors="coerce")
    wells["yield_category"] = wells["YIELD"].apply(
        lambda x: "Low yield (<10 L/min)" if pd.notna(x) and x < 10
        else "Adequate yield (≥10 L/min)" if pd.notna(x) and x >= 10
        else "Unknown yield"
    )
    low_yield_mask = (wells["YIELD"] < 5) & (wells["YIELD"].notna())
    wells.loc[low_yield_mask & (wells["drying_risk"].str.contains("Moderate")), "drying_risk"] = "High risk - Low yield well"

# ---------------------------
# 6. Aquifer classification using GeoPandas (if coords and shapefiles exist)
# ---------------------------
wells["aquifer_type"] = "Unknown"
wells["latitude"] = np.nan
wells["longitude"] = np.nan

if ("X" in wells.columns and "Y" in wells.columns) and pd.notna(wells["X"]).any() and pd.notna(wells["Y"]).any():
    try:
        # ** PATCH: Ensure X and Y are numeric, coercing any non-numeric values to NaN **
        wells["X"] = pd.to_numeric(wells["X"], errors='coerce') 
        wells["Y"] = pd.to_numeric(wells["Y"], errors='coerce') 
        
        # ** FINAL GEO FIX: Create a temporary DataFrame for spatial analysis from clean rows **
        wells_clean = wells.dropna(subset=["X", "Y"]).copy()
        
        # Skip if no rows are left with valid coordinates
        if wells_clean.empty:
            print("Warning: All wells lacked valid X/Y coordinates for GeoPandas analysis.")
            raise Exception("No valid coordinates")
            
        # Create GeoDataFrame from the clean data
        wells_gdf = gpd.GeoDataFrame(
            wells_clean,
            geometry=gpd.points_from_xy(wells_clean["X"], wells_clean["Y"]),
            crs=utm_crs
        ).to_crs(wgs84)
        
        # Extract latitude and longitude from geometry
        wells_gdf["longitude"] = wells_gdf.geometry.x
        wells_gdf["latitude"] = wells_gdf.geometry.y

        # Load shapefiles and reproject to WGS84
        bedrock = gpd.read_file(bedrock_shp).to_crs(wgs84)
        surficial = gpd.read_file(surficial_shp).to_crs(wgs84)

        # Spatial join: bedrock
        wb = gpd.sjoin(wells_gdf, bedrock[["geometry"]], how="left", predicate="within")
        wb["aquifer_type"] = np.where(wb["index_right"].notna(), "Bedrock", np.nan)
        # drop index_right then join to surficial
        wb = wb.drop(columns=[c for c in wb.columns if c == "index_right"])
        wb = gpd.sjoin(wb, surficial[["geometry"]], how="left", predicate="within")
        wb.loc[wb["index_right"].notna(), "aquifer_type"] = "Surficial"
        
        # ** FINAL GEO FIX: Create a temporary result DataFrame and merge back by index **
        spatial_results = pd.DataFrame(wb).set_index(wells_clean.index)[["aquifer_type", "latitude", "longitude"]]
        
        # Merge the new spatial columns back into the original 'wells' DataFrame
        wells["aquifer_type"] = spatial_results["aquifer_type"].combine_first(wells["aquifer_type"])
        wells["latitude"] = spatial_results["latitude"].combine_first(wells["latitude"])
        wells["longitude"] = spatial_results["longitude"].combine_first(wells["longitude"])

        print(f"Aquifer classification applied (known types: {wells['aquifer_type'].nunique(dropna=True)})")
    except Exception as e:
        # Catch and report any remaining errors
        print(f"Warning: final aquifer join failed ({e}). Aquifer type set to 'Unknown'.")
        wells["aquifer_type"] = wells.get("aquifer_type", "Unknown")
else:
    print("No X/Y coordinates found for aquifer spatial join — aquifer_type set to 'Unknown' for all wells.")
# ---------------------------
# 7. Apply Drought Stress Test (NEW)
# ---------------------------
# Note: DROUGHT_DRAWDOWN_M and WSC_STATION_ID are defined in the config section.
DEFAULT_DRAWDOWN = DROUGHT_DRAWDOWN_M # Default drawdown (2.0m)

# 1. Fetch Real-Time Surface Water Data
current_flow_rate = fetch_real_time_discharge(WSC_STATION_ID, WSC_API_URL)

if current_flow_rate is not None:
    # Simplified logic for initial integration 
    # Placeholder: Assuming a critical threshold of 1.0 m³/s for the Salmon River near Truro.
    CRITICAL_FLOW_THRESHOLD = 1.0 
    
    if current_flow_rate < CRITICAL_FLOW_THRESHOLD:
        # If streamflow is critically low, apply a higher, worst-case drawdown.
        wells["drought_drawdown_m"] = DEFAULT_DRAWDOWN * 1.5 
        print(f"CRITICAL surface water flow detected ({current_flow_rate} m³/s). Applying {DEFAULT_DRAWDOWN * 1.5}m extreme drought stress.")
    else:
        # If flow is near normal, use the default conservative drawdown.
        wells["drought_drawdown_m"] = DEFAULT_DRAWDOWN
        print(f"Normal surface water flow ({current_flow_rate} m³/s). Applying {DEFAULT_DRAWDOWN}m default stress test.")
else:
    # If API call fails, fall back to the safest default.
    wells["drought_drawdown_m"] = DEFAULT_DRAWDOWN
    print(f"WSC data unavailable or failed to parse. Applying {DEFAULT_DRAWDOWN}m safe default stress test.")

# 2. Calculate Stressed Water Level
wells["drought_water_level_m"] = wells["current_water_level_m"] + wells["drought_drawdown_m"]

# 3. Recalculate Buffer and Risk (The final risk output is now the STRESSED risk)
wells["buffer_m_drought"] = wells["pump_depth_m"] - wells["drought_water_level_m"]
wells["drying_risk_drought"] = wells["buffer_m_drought"].apply(classify_risk)

# 4. Overwrite Final Columns
wells["buffer_m"] = wells["buffer_m_drought"]
wells["drying_risk"] = wells["drying_risk_drought"]

# 5. Rename columns for clarity in output
wells = wells.rename(columns={"current_water_level_m": "current_water_level_m_observed"})
wells = wells.rename(columns={"drought_water_level_m": "stressed_water_level_m"})


# ---------------------------
# 8. Create Google Maps links and location info (RENAMED FROM 7)
# ---------------------------
def create_google_maps_link(row):
    """Create Google Maps link from address or coordinates"""
    # First try civic address
    if "CIVIC_ADDRESS" in row and pd.notna(row.get("CIVIC_ADDRESS")):
        address = str(row["CIVIC_ADDRESS"]).strip()
        if address and address != "nan":
            # Add municipality if available
            if "MUNICIPALITY" in row and pd.notna(row.get("MUNICIPALITY")):
                municipality = str(row["MUNICIPALITY"]).strip()
                if municipality and municipality != "nan":
                    address += f", {municipality}"
            address += ", Colchester County, Nova Scotia"
            return f"https://www.google.com/maps/search/?api=1&query={html.escape(address)}"
    
    # Fall back to coordinates if available
    if (pd.notna(row.get("latitude")) and pd.notna(row.get("longitude"))):
        return f"https://www.google.com/maps/?q={row['latitude']},{row['longitude']}"
    
    return ""

def format_location_display(row):
    """Format location for display in the report"""
    location_parts = []
    
    if "CIVIC_ADDRESS" in row and pd.notna(row.get("CIVIC_ADDRESS")):
        address = str(row["CIVIC_ADDRESS"]).strip()
        if address and address != "nan":
            location_parts.append(address)
    
    if "MUNICIPALITY" in row and pd.notna(row.get("MUNICIPALITY")):
        municipality = str(row["MUNICIPALITY"]).strip()
        if municipality and municipality != "nan":
            location_parts.append(municipality)
    
    if not location_parts:
        if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
            location_parts.append(f"Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}")
        else:
            location_parts.append("Location unknown")
    
    return ", ".join(location_parts)

# Apply location functions
wells["google_maps_link"] = wells.apply(create_google_maps_link, axis=1)
wells["location_display"] = wells.apply(format_location_display, axis=1)

# ---------------------------
# 9. Summary stats and CSV export (RENAMED FROM 8)
# ---------------------------
print("\n=== ANALYSIS SUMMARY ===")
print(f"Total wells analyzed: {len(wells)}")
print(f"Wells updated from observation data: {updated_count}")
risk_counts = wells["drying_risk"].value_counts(dropna=False)
print("\nRisk distribution:")
print(risk_counts.to_string())

# Export CSV (include useful columns)
out_cols = ["WELL_ID", "CIVIC_ADDRESS", "MUNICIPALITY", "location_display", "google_maps_link", 
            "latitude", "longitude", "DEPTH", "STATIC_WATER_LEVEL", 
            "current_water_level_m_observed", "drought_drawdown_m", "stressed_water_level_m", # <--- PATCH: ADDED/RENAMED
            "pump_depth_m", "buffer_m", "drying_risk", "YIELD", "yield_category", "aquifer_type"]
# only keep existing
out_cols = [c for c in out_cols if c in wells.columns]
results = wells[out_cols].sort_values("buffer_m", ascending=True)
results.to_csv(output_csv, index=False)
print(f"\nDetailed results saved to {output_csv}")

# ---------------------------
# 10. Create HTML report (Dashboard Style with DataTables) (RENAMED FROM 9)
# ---------------------------
print("Generating HTML dashboard...")

# === DATA PREPARATION FOR HTML ===

# Prepare the main table for client-side rendering
all_wells_display = results.copy()
if 'google_maps_link' in all_wells_display.columns:
    # ... (code to create 'Map Link' and define display_cols remains the same) ...
    all_wells_display['Map Link'] = all_wells_display['google_maps_link'].apply(
        lambda x: f'<a href="{x}" target="_blank" class="map-link">📍 Map</a>' if x else '—'
    )
    # Define columns for the final table, removing originals used for the map link
    drop_cols = ['google_maps_link', 'CIVIC_ADDRESS', 'MUNICIPALITY', 'latitude', 'longitude']
    
    # --- THIS IS THE CORRECTED LINE from the previous request ---
    display_cols = ['WELL_ID', 'location_display', 'Map Link'] + [c for c in all_wells_display.columns if c not in drop_cols + ['WELL_ID', 'location_display', 'Map Link']]
    
    all_wells_display = all_wells_display[display_cols]

# *** Convert the full dataset to JSON ***
all_wells_json = all_wells_display.to_json(orient="records")
# *** Create the column definitions for DataTables ***
datatables_columns = json.dumps([{"data": col, "title": col} for col in all_wells_display.columns])

# ** PATCH: Save the data payload separately to keep the HTML file small **
data_json_file = "wells_data.json"
with open(data_json_file, "w", encoding="utf-8") as f:
    f.write(all_wells_json)
print(f"Data payload saved separately to {data_json_file}")
# *** Convert the full dataset to JSON for DataTables ***
all_wells_json = all_wells_display.to_json(orient="records")
# *** Create the column definitions for DataTables ***
datatables_columns = json.dumps([{"data": col, "title": col} for col in all_wells_display.columns])


# Prepare data for smaller, pre-rendered tables
top20_display = all_wells_display.head(20)

# Risk by aquifer pivot
if "aquifer_type" in wells.columns:
    risk_by_aq = wells.groupby(["aquifer_type", "drying_risk"]).size().unstack(fill_value=0)
else:
    risk_by_aq = pd.DataFrame()

# KPIs
total_wells = len(wells)
num_critical = len(wells[wells["drying_risk"].str.contains("CRITICAL", na=False)])
num_high = len(wells[wells["drying_risk"].str.contains("High risk", na=False)])
avg_buffer = wells["buffer_m"].mean(skipna=True)


# === HTML STRING BUILDING ===

html_parts = []
html_parts.append("<!doctype html>")
html_parts.append("<html lang='en'><head><meta charset='utf-8'><title>Colchester Well Drying Risk Report</title>")

# CSS and JS libraries
html_parts.append("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.bootstrap5.min.css">
<style>
body { background:#f8fafc; color:#111; }
h1,h2 { color:#0b4d78; margin-top:20px; }
.card { border-radius:10px; box-shadow:0 1px 6px rgba(0,0,0,0.08); margin-bottom:20px; }
.kpi-card { text-align:center; padding:20px; }
.kpi-value { font-size:1.5rem; font-weight:bold; color:#0b4d78; }
.nav-tabs .nav-link.active { background:#0b4d78; color:#fff; }
.map-link { color:#0b4d78; text-decoration:none; }
.map-link:hover { text-decoration:underline; }
.dataTables_wrapper .dt-buttons { margin-bottom:10px; }
table.dataTable thead th { white-space: nowrap; }
/* --- NEW CSS FOR HELP WINDOW --- */
#help-window .card-body { font-size: 0.9rem; }
#help-window .card-body ul { padding-left: 20px; }
</style>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.bootstrap5.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.print.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</head><body>
<div class="container-fluid py-4">
""")

# Title + timestamp
html_parts.append(f"<h1 class='mb-3'>Colchester County Well Drying Risk Dashboard</h1>")
html_parts.append(f"<p class='text-muted'>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>")

# --- NEW: EXPANDABLE HELP WINDOW HTML ---
html_parts.append("""
<p>
  <button class="btn btn-outline-info btn-sm" type="button" data-bs-toggle="collapse" data-bs-target="#help-window" aria-expanded="false" aria-controls="help-window">
    How to Read This Report 📖
  </button>
</p>
<div class="collapse" id="help-window">
  <div class="card card-body bg-light mb-4">
    <h4>Understanding the Columns</h4>
    <p>This report analyzes the risk of private wells running dry based on their construction and water levels.</p>
    <ul>
        <li><strong>buffer_m (Buffer in Meters):</strong> This is the most important column. It shows the vertical distance between the current water level and the estimated pump depth. A small or negative number indicates a high risk of the pump drawing air.</li>
        <li><strong>drying_risk:</strong> A risk category assigned based on the <strong>buffer_m</strong> value:
            <ul>
                <li><strong>CRITICAL:</strong> Buffer is negative. The water level is likely below the pump.</li>
                <li><strong>High risk:</strong> Buffer is less than 2 meters.</li>
                <li><strong>Moderate risk:</strong> Buffer is between 2 and 5 meters.</li>
                <li><strong>Low risk:</strong> Buffer is greater than 5 meters.</li>
            </ul>
        </li>
        <li><strong>stressed_water_level_m:</strong> The estimated depth to the water from the ground surface, in meters, **adjusted for potential drought conditions** based on Environment Canada hydrometric data. A larger number means the water is deeper down.</li>
        <li><strong>current_water_level_m_observed:</strong> The original depth to water (static level or latest observation well reading).</li>
        <li><strong>drought_drawdown_m:</strong> The amount of additional drawdown (in meters) applied to the water level to stress-test the well under drought conditions.</li>
        <li><strong>pump_depth_m:</strong> An <em>estimated</em> depth of the submersible pump. This is calculated as 80% of the well's total depth, or 2.5m from the bottom, whichever is shallower.</li>
        <li><strong>DEPTH:</strong> The total drilled depth of the well in meters.</li>
        <li><strong>YIELD:</strong> The well's flow rate in Liters per Minute. Wells with very low yield (< 5 L/min) may have their risk category upgraded.</li>
    </ul>
  </div>
</div>
""") # <--- PATCH: UPDATED HTML HELP TEXT

# KPI cards
html_parts.append("""
<div class="row">
  <div class="col-md-3"><div class="card kpi-card"><div>Total Wells</div><div class="kpi-value">{}</div></div></div>
  <div class="col-md-3"><div class="card kpi-card"><div>Critical Wells</div><div class="kpi-value">{}</div></div></div>
  <div class="col-md-3"><div class="card kpi-card"><div>High Risk Wells</div><div class="kpi-value">{}</div></div></div>
  <div class="col-md-3"><div class="card kpi-card"><div>Avg Buffer (m)</div><div class="kpi-value">{:.2f}</div></div></div>
</div>
""".format(total_wells, num_critical, num_high, avg_buffer))

# Tabs
html_parts.append("""
<ul class="nav nav-tabs" id="reportTabs" role="tablist">
  <li class="nav-item" role="presentation">
    <button class="nav-link active" id="top20-tab" data-bs-toggle="tab" data-bs-target="#top20" type="button" role="tab">Top 20 At-Risk</button>
  </li>
  <li class="nav-item" role="presentation">
    <button class="nav-link" id="all-tab" data-bs-toggle="tab" data-bs-target="#all" type="button" role="tab">All Wells</button>
  </li>
  <li class="nav-item" role="presentation">
    <button class="nav-link" id="dist-tab" data-bs-toggle="tab" data-bs-target="#dist" type="button" role="tab">Risk Distribution</button>
  </li>
  <li class="nav-item" role="presentation">
    <button class="nav-link" id="aq-tab" data-bs-toggle="tab" data-bs-target="#aq" type="button" role="tab">Risk by Aquifer</button>
  </li>
</ul>
<div class="tab-content mt-3">
""")

# Tab: Top 20
html_parts.append("<div class='tab-pane fade show active' id='top20' role='tabpanel'>")
html_parts.append(top20_display.to_html(classes="table table-striped table-bordered", escape=False, index=False, table_id="top20_table"))
html_parts.append("</div>")

# Tab: All Wells
html_parts.append("<div class='tab-pane fade' id='all' role='tabpanel'>")
html_parts.append('<div class="table-responsive"><table id="all_wells_table" class="table table-striped table-bordered" style="width:100%"></table></div>')
html_parts.append("</div>")

# Tab: Risk Distribution
html_parts.append("<div class='tab-pane fade' id='dist' role='tabpanel'>")
html_parts.append(risk_counts.reset_index().rename(columns={0: "count"}).to_html(classes="table table-striped", index=False, table_id="risk_table"))
html_parts.append("</div>")

# Tab: Risk by Aquifer
html_parts.append("<div class'tab-pane fade' id='aq' role='tabpanel'>")
if not risk_by_aq.empty:
    html_parts.append(risk_by_aq.to_html(classes="table table-striped", table_id="aq_table"))
else:
    html_parts.append("<p class='text-muted'>No aquifer classification available.</p>")
html_parts.append("</div>")

html_parts.append("</div>") # end tab-content
html_parts.append("</div>") # end container


# Inject column definitions (small) and define the data file path
html_parts.append(f"<script> const dtColumns = {datatables_columns}; const dataJsonFile = '{data_json_file}';</script>")

# Updated DataTables initialization script
# Updated DataTables initialization script
html_parts.append("""
<script>
$(document).ready(function() {
  // Use Fetch API to load the data payload separately
  fetch(dataJsonFile)
    .then(response => {
        if (!response.ok) {
            // Handle HTTP errors (e.g., 404)
            throw new Error('Failed to load data: ' + response.statusText);
        }
        return response.json();
    })
    .then(allWellsData => {
        // Initialize the large table from JSON data after fetching
        $('#all_wells_table').DataTable({
            data: allWellsData,
            columns: dtColumns,
            pageLength: 25,
            lengthMenu: [10, 25, 50, 100, {label: "All", value: -1}],
            dom: 'Bfrtip',
            buttons: ['copy', 'csv', 'excel', 'print'],
            scrollX: true,
            responsive: true
        });
    })
    .catch(error => {
        console.error("Error initializing dashboard:", error);
        // Display an error message to the user if the data fails to load
        $('#all_wells_table').html("<p style='color:red;'>Error: Could not load well data. Ensure 'wells_data.json' is present alongside the HTML report.</p>");
    });

  // Initialize the smaller, pre-rendered tables (they don't use the large allWellsData payload)
  $('#top20_table, #risk_table, #aq_table').DataTable({
    pageLength: 20,
    dom: 'Bfrtip',
    buttons: ['copy', 'csv', 'print'],
    scrollX: true
  });
});
</script>
""")

html_parts.append("</body></html>")

# Write the final HTML file
with open(output_html, "w", encoding="utf-8") as f:
    f.write("\n".join(html_parts))

print(f"Dashboard report written to {output_html}")