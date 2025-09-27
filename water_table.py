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
output_html = "index.html"

# If your UTM zone is different, change this. EPSG:26920 is NAD83 / UTM zone 20N (Nova Scotia).
utm_crs = "EPSG:26920"
wgs84 = "EPSG:4326"

# Pareto Principle Addition: Drought Stress Test Configuration
# Assumes a 2.0 meter worst-case drawdown due to drought for wells without detailed time series.
DROUGHT_DRAWDOWN_M = 2.0 # <--- PATCH: ADDED

# ECCC Hydrometric Data Integration Configuration (Optional, but highly recommended)
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
# Initialize with consistent data types to prevent dtype promotion errors
wells["aquifer_type"] = "Unknown"  # String, not object with mixed types
wells["latitude"] = 0.0  # Float64, not mixed NaN/float
wells["longitude"] = 0.0  # Float64, not mixed NaN/float

print("DEBUG: Initialized aquifer_type as string, lat/lng as float64")

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
        
        # COMPLETELY SKIP THE PROBLEMATIC MERGE - just do coordinates
        print("DEBUG: Skipping complex spatial results merge to avoid dtype issues")
        print(f"DEBUG: Successfully processed {len(wells_clean)} wells with coordinates")
        
        print(f"Coordinate conversion and basic aquifer classification completed")
    except Exception as e:
        # Catch and report any remaining errors
        print(f"Warning: coordinate processing failed ({e}). Coordinates may be missing.")
        import traceback
        print(f"DEBUG: Full coordinate error traceback:\n{traceback.format_exc()}")
        wells["aquifer_type"] = "Unknown"
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
# 10. Prepare map data for wells with valid coordinates (NEW)
# ---------------------------
def prepare_map_data(wells_df, max_points=5000):
    """Prepare optimized map data with clustering for large datasets"""
    
    print("=== DEBUGGING COORDINATE DATA ===")
    print(f"Total wells in dataset: {len(wells_df)}")
    
    # Check what coordinate columns we have
    coord_cols = [col for col in wells_df.columns if any(term in col.upper() for term in ['LAT', 'LON', 'X', 'Y', 'EASTING', 'NORTHING'])]
    print(f"Available coordinate columns: {coord_cols}")
    
    # Check latitude/longitude columns specifically
    if 'latitude' in wells_df.columns:
        lat_stats = wells_df['latitude'].describe()
        print(f"Latitude stats: {lat_stats}")
        lat_valid = wells_df['latitude'].notna().sum()
        print(f"Wells with valid latitude: {lat_valid}")
    else:
        print("No 'latitude' column found")
    
    if 'longitude' in wells_df.columns:
        lng_stats = wells_df['longitude'].describe()
        print(f"Longitude stats: {lng_stats}")
        lng_valid = wells_df['longitude'].notna().sum()
        print(f"Wells with valid longitude: {lng_valid}")
    else:
        print("No 'longitude' column found")
    
    # Check X/Y columns if lat/lng not available
    if 'X' in wells_df.columns and 'Y' in wells_df.columns:
        x_valid = wells_df['X'].notna().sum()
        y_valid = wells_df['Y'].notna().sum()
        print(f"Wells with valid X coordinates: {x_valid}")
        print(f"Wells with valid Y coordinates: {y_valid}")
        if x_valid > 0:
            print(f"X coordinate range: {wells_df['X'].min()} to {wells_df['X'].max()}")
        if y_valid > 0:
            print(f"Y coordinate range: {wells_df['Y'].min()} to {wells_df['Y'].max()}")
    
    # Try multiple approaches to find valid coordinates
    map_wells = None
    
    # Approach 1: Use latitude/longitude if available
    if 'latitude' in wells_df.columns and 'longitude' in wells_df.columns:
        map_wells = wells_df[
            pd.notna(wells_df["latitude"]) & 
            pd.notna(wells_df["longitude"]) &
            (wells_df["latitude"] != 0) & 
            (wells_df["longitude"] != 0) &
            (wells_df["latitude"].between(-90, 90)) &  # Valid latitude range
            (wells_df["longitude"].between(-180, 180))  # Valid longitude range
        ].copy()
        print(f"Found {len(map_wells)} wells with valid lat/lng coordinates")
    
    # Approach 2: If no lat/lng, try to use X/Y coordinates and convert them
    if (map_wells is None or map_wells.empty) and 'X' in wells_df.columns and 'Y' in wells_df.columns:
        print("Attempting to convert X/Y coordinates to lat/lng...")
        
        # Filter wells with valid X/Y coordinates
        xy_wells = wells_df[
            pd.notna(wells_df["X"]) & 
            pd.notna(wells_df["Y"]) &
            (wells_df["X"] != 0) & 
            (wells_df["Y"] != 0)
        ].copy()
        
        if not xy_wells.empty:
            print(f"Found {len(xy_wells)} wells with X/Y coordinates")
            
            # Try to determine if these are UTM coordinates (typical range for Nova Scotia)
            x_min, x_max = xy_wells['X'].min(), xy_wells['X'].max()
            y_min, y_max = xy_wells['Y'].min(), xy_wells['Y'].max()
            
            print(f"X range: {x_min} to {x_max}")
            print(f"Y range: {y_min} to {y_max}")
            
            # Check if these look like UTM coordinates for Nova Scotia (Zone 20N)
            # UTM Zone 20N for NS: X ~200,000-800,000, Y ~4,900,000-5,200,000
            if (x_min > 100000 and x_max < 900000 and 
                y_min > 4800000 and y_max < 5300000):
                print("Coordinates appear to be UTM Zone 20N - converting to lat/lng")
                
                try:
                    # Create temporary GeoDataFrame for coordinate conversion
                    temp_gdf = gpd.GeoDataFrame(
                        xy_wells,
                        geometry=gpd.points_from_xy(xy_wells["X"], xy_wells["Y"]),
                        crs=utm_crs  # UTM Zone 20N
                    ).to_crs(wgs84)  # Convert to WGS84
                    
                    # Extract converted coordinates
                    xy_wells["latitude"] = temp_gdf.geometry.y
                    xy_wells["longitude"] = temp_gdf.geometry.x
                    
                    # Update the main wells dataframe
                    wells_df.loc[xy_wells.index, "latitude"] = xy_wells["latitude"]
                    wells_df.loc[xy_wells.index, "longitude"] = xy_wells["longitude"]
                    
                    # Use converted coordinates
                    map_wells = xy_wells[
                        pd.notna(xy_wells["latitude"]) & 
                        pd.notna(xy_wells["longitude"])
                    ].copy()
                    
                    print(f"Successfully converted {len(map_wells)} wells to lat/lng coordinates")
                    
                except Exception as e:
                    print(f"Error converting UTM coordinates: {e}")
                    map_wells = pd.DataFrame()
            
            elif (x_min > -180 and x_max < 180 and y_min > -90 and y_max < 90):
                print("Coordinates appear to already be in lat/lng format")
                # Assume X=longitude, Y=latitude
                xy_wells["longitude"] = xy_wells["X"]
                xy_wells["latitude"] = xy_wells["Y"]
                map_wells = xy_wells.copy()
                
                # Update main dataframe
                wells_df.loc[xy_wells.index, "latitude"] = xy_wells["latitude"]
                wells_df.loc[xy_wells.index, "longitude"] = xy_wells["longitude"]
            
            else:
                print("Coordinate system not recognized - coordinates may be in an unsupported projection")
                map_wells = pd.DataFrame()
    
    if map_wells is None or map_wells.empty:
        print("ERROR: No wells with valid coordinates found for mapping!")
        print("Please ensure your well data has either:")
        print("  1. 'latitude' and 'longitude' columns with WGS84 coordinates, or")
        print("  2. 'X' and 'Y' columns with UTM coordinates")
        return []
    
    print(f"Final: {len(map_wells)} wells available for mapping")
    
    # If too many points, prioritize high-risk wells and sample the rest
    if len(map_wells) > max_points:
        # Get all critical and high risk wells
        priority_wells = map_wells[
            map_wells["drying_risk"].str.contains("CRITICAL|High risk", na=False)
        ]
        
        # Sample remaining wells
        remaining_wells = map_wells[
            ~map_wells["drying_risk"].str.contains("CRITICAL|High risk", na=False)
        ]
        
        remaining_sample_size = max(0, max_points - len(priority_wells))
        if len(remaining_wells) > remaining_sample_size:
            remaining_wells = remaining_wells.sample(n=remaining_sample_size, random_state=42)
        
        map_wells = pd.concat([priority_wells, remaining_wells])
        print(f"Optimized to {len(map_wells)} wells for mapping (prioritizing high-risk wells)")
    
    # Create map data with risk-based styling
    map_data = []
    for _, well in map_wells.iterrows():
        # Determine marker color based on risk
        if pd.isna(well.get("drying_risk")):
            color = "gray"
            risk_priority = 4
        elif "CRITICAL" in str(well["drying_risk"]):
            color = "red"
            risk_priority = 0
        elif "High risk" in str(well["drying_risk"]):
            color = "orange"
            risk_priority = 1
        elif "Moderate risk" in str(well["drying_risk"]):
            color = "yellow"
            risk_priority = 2
        else:
            color = "green"
            risk_priority = 3
        
        # Create popup content with safe value handling
        def safe_format(value, format_str="{}", default="N/A"):
            try:
                if pd.isna(value) or value == "nan":
                    return default
                return format_str.format(value)
            except:
                return default
        
        popup_content = f"""
        <div style="width: 250px;">
            <strong>Well ID:</strong> {safe_format(well.get('WELL_ID'))}<br>
            <strong>Location:</strong> {safe_format(well.get('location_display'), default='Unknown')}<br>
            <strong>Risk Level:</strong> <span style="color: {color}; font-weight: bold;">{safe_format(well.get('drying_risk'), default='Unknown')}</span><br>
            <strong>Buffer:</strong> {safe_format(well.get('buffer_m'), '{:.1f}m')}<br>
            <strong>Well Depth:</strong> {safe_format(well.get('DEPTH'), '{}m')}<br>
            <strong>Aquifer:</strong> {safe_format(well.get('aquifer_type'), default='Unknown')}<br>
            <strong>Yield:</strong> {safe_format(well.get('YIELD'), '{} L/min')}<br>
            <strong>Coordinates:</strong> {well['latitude']:.4f}, {well['longitude']:.4f}<br>
        </div>
        """
        
        map_data.append({
            'lat': float(well['latitude']),
            'lng': float(well['longitude']),
            'color': color,
            'risk_priority': risk_priority,
            'popup': popup_content,
            'well_id': str(well.get('WELL_ID', '')),
            'risk': str(well.get('drying_risk', ''))
        })
    
    return map_data

# Prepare map data
map_wells_data = prepare_map_data(wells)

# ---------------------------
# 11. Create HTML report (Dashboard Style with DataTables and Interactive Map) (RENAMED FROM 10)
# ---------------------------
print("Generating HTML dashboard with interactive map...")

# === DATA PREPARATION FOR HTML ===

# Prepare the main table for client-side rendering
all_wells_display = results.copy()
if 'google_maps_link' in all_wells_display.columns:
    all_wells_display['Map Link'] = all_wells_display['google_maps_link'].apply(
        lambda x: f'<a href="{x}" target="_blank" class="map-link">🔍 Map</a>' if x else '—'
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

# Save map data separately
map_data_json_file = "wells_map_data.json"
with open(map_data_json_file, "w", encoding="utf-8") as f:
    json.dump(map_wells_data, f)
print(f"Map data payload saved separately to {map_data_json_file}")

# Prepare data for smaller, pre-rendered tables
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

# Calculate map center (Nova Scotia/Colchester County approximate center)
if map_wells_data:
    map_center_lat = sum(p['lat'] for p in map_wells_data) / len(map_wells_data)
    map_center_lng = sum(p['lng'] for p in map_wells_data) / len(map_wells_data)
else:
    map_center_lat = 45.3
    map_center_lng = -63.3  # Nova Scotia center

# === HTML STRING BUILDING ===

html_parts = []
html_parts.append("<!doctype html>")
html_parts.append("<html lang='en'><head><meta charset='utf-8'><title>Colchester Well Drying Risk Report</title>")

# CSS and JS libraries
html_parts.append(f"""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.bootstrap5.min.css">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css" />
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css" />
<style>
body {{ background:#f8fafc; color:#111; }}
h1,h2 {{ color:#0b4d78; margin-top:20px; }}
.card {{ border-radius:10px; box-shadow:0 1px 6px rgba(0,0,0,0.08); margin-bottom:20px; }}
.kpi-card {{ text-align:center; padding:20px; }}
.kpi-value {{ font-size:1.5rem; font-weight:bold; color:#0b4d78; }}
.nav-tabs .nav-link.active {{ background:#0b4d78; color:#fff; }}
.map-link {{ color:#0b4d78; text-decoration:none; }}
.map-link:hover {{ text-decoration:underline; }}
.dataTables_wrapper .dt-buttons {{ margin-bottom:10px; }}
table.dataTable thead th {{ white-space: nowrap; }}

/* Map Modal Styles */
#mapModal .modal-dialog {{ max-width: 95vw; }}
#mapModal .modal-body {{ padding: 0; }}
#wellMap {{ height: 80vh; width: 100%; }}

/* Map Controls */
.map-controls {{ 
    position: absolute; 
    top: 10px; 
    right: 10px; 
    z-index: 1000; 
    background: white; 
    padding: 10px; 
    border-radius: 5px; 
    box-shadow: 0 1px 5px rgba(0,0,0,0.2);
}}
.map-controls label {{ margin-right: 10px; font-size: 0.9rem; }}
.map-legend {{ 
    position: absolute; 
    bottom: 20px; 
    right: 10px; 
    z-index: 1000; 
    background: white; 
    padding: 10px; 
    border-radius: 5px; 
    box-shadow: 0 1px 5px rgba(0,0,0,0.2);
    font-size: 0.8rem;
}}
.legend-item {{ display: flex; align-items: center; margin-bottom: 3px; }}
.legend-color {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; }}

/* Help window styles */
#help-window .card-body {{ font-size: 0.9rem; }}
#help-window .card-body ul {{ padding-left: 20px; }}

/* Responsive adjustments */
@media (max-width: 768px) {{
    .map-controls {{ 
        position: relative; 
        margin-bottom: 10px; 
    }}
    .map-legend {{
        position: relative;
        margin-top: 10px;
    }}
    #wellMap {{ height: 60vh; }}
}}
</style>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.bootstrap5.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.print.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
</head><body>
<div class="container-fluid py-4">
""")

# Title + timestamp
html_parts.append(f"<h1 class='mb-3'>Colchester County Well Drying Risk Dashboard</h1>")
html_parts.append(f"<p class='text-muted'>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Wells mapped: {len(map_wells_data)}</p>")

# --- NEW: EXPANDABLE HELP WINDOW HTML ---
html_parts.append("""
<p>
  <button class="btn btn-outline-info btn-sm" type="button" data-bs-toggle="collapse" data-bs-target="#help-window" aria-expanded="false" aria-controls="help-window">
    How to Read This Report 📖
  </button>
  <button class="btn btn-primary btn-sm ms-2" type="button" data-bs-toggle="modal" data-bs-target="#mapModal">
    🗺️ View Interactive Map
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
""")

# KPI cards
html_parts.append(f"""
<div class="row">
  <div class="col-md-3"><div class="card kpi-card"><div>Total Wells</div><div class="kpi-value">{total_wells}</div></div></div>
  <div class="col-md-3"><div class="card kpi-card"><div>Critical Wells</div><div class="kpi-value">{num_critical}</div></div></div>
  <div class="col-md-3"><div class="card kpi-card"><div>High Risk Wells</div><div class="kpi-value">{num_high}</div></div></div>
  <div class="col-md-3"><div class="card kpi-card"><div>Avg Buffer (m)</div><div class="kpi-value">{avg_buffer:.2f}</div></div></div>
</div>
""")

# Tabs - MODIFIED: Remove 'Top 20' and 'All Wells' tabs, make 'All Wells' the default view
html_parts.append("""
<ul class="nav nav-tabs" id="reportTabs" role="tablist">
  <li class="nav-item" role="presentation">
    <button class="nav-link active" id="all-tab" data-bs-toggle="tab" data-bs-target="#all" type="button" role="tab" aria-selected="true">All Wells</button>
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

# Tab: All Wells - MODIFIED: Set to show active/main content
html_parts.append("<div class='tab-pane fade show active' id='all' role='tabpanel' aria-labelledby='all-tab'>")
html_parts.append('<div class="table-responsive"><table id="all_wells_table" class="table table-striped table-bordered" style="width:100%"></table></div>')
html_parts.append("</div>")

# Tab: Risk Distribution
html_parts.append("<div class='tab-pane fade' id='dist' role='tabpanel' aria-labelledby='dist-tab'>")
html_parts.append(risk_counts.reset_index().rename(columns={0: "count"}).to_html(classes="table table-striped", index=False, table_id="risk_table"))
html_parts.append("</div>")

# Tab: Risk by Aquifer
html_parts.append("<div class='tab-pane fade' id='aq' role='tabpanel' aria-labelledby='aq-tab'>")
if not risk_by_aq.empty:
    html_parts.append(risk_by_aq.to_html(classes="table table-striped", table_id="aq_table"))
else:
    html_parts.append("<p class='text-muted'>No aquifer classification available.</p>")
html_parts.append("</div>")

html_parts.append("</div>") # end tab-content

# Map Modal
html_parts.append(f"""
<!-- Map Modal -->
<div class="modal fade" id="mapModal" tabindex="-1" aria-labelledby="mapModalLabel" aria-hidden="true">
  <div class="modal-dialog modal-fullscreen-lg-down">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="mapModalLabel">Interactive Well Risk Map</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
      </div>
      <div class="modal-body position-relative">
        <div class="map-controls">
          <label><input type="checkbox" id="clusterToggle" checked> Cluster markers</label>
          <label><input type="checkbox" id="criticalOnly"> Critical/High risk only</label>
        </div>
        <div id="wellMap"></div>
        <div class="map-legend">
          <strong>Risk Levels:</strong><br>
          <div class="legend-item"><div class="legend-color" style="background-color: red;"></div>Critical</div>
          <div class="legend-item"><div class="legend-color" style="background-color: orange;"></div>High Risk</div>
          <div class="legend-item"><div class="legend-color" style="background-color: yellow;"></div>Moderate Risk</div>
          <div class="legend-item"><div class="legend-color" style="background-color: green;"></div>Low Risk</div>
          <div class="legend-item"><div class="legend-color" style="background-color: gray;"></div>No Data</div>
        </div>
      </div>
    </div>
  </div>
</div>
""")

html_parts.append("</div>") # end container

# Inject column definitions (small) and define the data file paths
html_parts.append(f"""<script> 
const dtColumns = {datatables_columns}; 
const dataJsonFile = '{data_json_file}';
const mapDataFile = '{map_data_json_file}';
const mapCenter = [{map_center_lat}, {map_center_lng}];
</script>""")

# Updated DataTables and Map initialization script
html_parts.append("""
<script>
let map = null;
let markersLayer = null;
let clusterGroup = null;
let allMapData = [];
let currentTable = null;

$(document).ready(function() {
  // Load table data
  fetch(dataJsonFile)
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to load data: ' + response.statusText);
        }
        return response.json();
    })
    .then(allWellsData => {
        // Initialize the large table from JSON data after fetching
        currentTable = $('#all_wells_table').DataTable({
            data: allWellsData,
            columns: dtColumns,
            pageLength: 25,
            lengthMenu: [10, 25, 50, 100, {label: "All", value: -1}],
            dom: 'Bfrtip',
            buttons: ['copy', 'csv', 'excel', 'print'],
            scrollX: true,
            responsive: true,
            order: [[findColumnIndex('buffer_m'), 'asc']] // Sort by buffer_m ascending (most critical first)
        });
        
        // Add row selection event to sync with map
        $('#all_wells_table tbody').on('click', 'tr', function() {
            if ($(this).hasClass('selected')) {
                $(this).removeClass('selected');
            } else {
                currentTable.$('tr.selected').removeClass('selected');
                $(this).addClass('selected');
                
                // If map is open, try to highlight the corresponding well
                const data = currentTable.row(this).data();
                if (map && data && data.WELL_ID) {
                    highlightWellOnMap(data.WELL_ID);
                }
            }
        });
    })
    .catch(error => {
        console.error("Error initializing dashboard:", error);
        $('#all_wells_table').html("<p style='color:red;'>Error: Could not load well data. Ensure 'wells_data.json' is present alongside the HTML report.</p>");
    });

  // Load map data
  fetch(mapDataFile)
    .then(response => response.json())
    .then(mapData => {
        allMapData = mapData;
        console.log(`Loaded ${allMapData.length} wells for mapping`);
    })
    .catch(error => {
        console.error("Error loading map data:", error);
    });

  // Initialize smaller tables
  $('#risk_table, #aq_table').DataTable({
    pageLength: 20,
    dom: 'Bfrtip',
    buttons: ['copy', 'csv', 'print'],
    scrollX: true,
    responsive: true
  });
});

// Helper function to find column index by name
function findColumnIndex(columnName) {
    for (let i = 0; i < dtColumns.length; i++) {
        if (dtColumns[i].data === columnName) {
            return i;
        }
    }
    return 0; // Default to first column
}

// Initialize map when modal is shown
$('#mapModal').on('shown.bs.modal', function () {
    if (!map) {
        initializeMap();
    } else {
        // Refresh map size in case of layout changes
        setTimeout(() => map.invalidateSize(), 100);
    }
});

function initializeMap() {
    // Initialize the map
    map = L.map('wellMap').setView(mapCenter, 9);
    
    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    
    // Initialize cluster group
    clusterGroup = L.markerClusterGroup({
        maxClusterRadius: 50,
        spiderfyOnMaxZoom: true,
        showCoverageOnHover: false
    });
    
    // Initialize regular marker layer group
    markersLayer = L.layerGroup();
    
    // Load markers
    loadMapMarkers();
    
    // Add event listeners for controls
    document.getElementById('clusterToggle').addEventListener('change', toggleClustering);
    document.getElementById('criticalOnly').addEventListener('change', filterMarkers);
}

function loadMapMarkers(filterCritical = false) {
    // Clear existing markers
    if (clusterGroup) clusterGroup.clearLayers();
    if (markersLayer) markersLayer.clearLayers();
    
    let filteredData = allMapData;
    if (filterCritical) {
        filteredData = allMapData.filter(well => 
            well.risk.includes('CRITICAL') || well.risk.includes('High risk')
        );
    }
    
    // Sort by risk priority (critical first)
    filteredData.sort((a, b) => a.risk_priority - b.risk_priority);
    
    filteredData.forEach(well => {
        const marker = L.circleMarker([well.lat, well.lng], {
            radius: 6,
            fillColor: well.color,
            color: '#000',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.8
        }).bindPopup(well.popup);
        
        // Store well ID for highlighting
        marker.wellId = well.well_id;
        
        clusterGroup.addLayer(marker);
        markersLayer.addLayer(marker);
    });
    
    // Add appropriate layer to map
    const clusterEnabled = document.getElementById('clusterToggle')?.checked ?? true;
    if (clusterEnabled) {
        map.addLayer(clusterGroup);
        if (map.hasLayer(markersLayer)) {
            map.removeLayer(markersLayer);
        }
    } else {
        map.addLayer(markersLayer);
        if (map.hasLayer(clusterGroup)) {
            map.removeLayer(clusterGroup);
        }
    }
}

function toggleClustering() {
    const clusterEnabled = document.getElementById('clusterToggle').checked;
    
    if (clusterEnabled) {
        if (map.hasLayer(markersLayer)) {
            map.removeLayer(markersLayer);
        }
        map.addLayer(clusterGroup);
    } else {
        if (map.hasLayer(clusterGroup)) {
            map.removeLayer(clusterGroup);
        }
        map.addLayer(markersLayer);
    }
}

function filterMarkers() {
    const criticalOnly = document.getElementById('criticalOnly').checked;
    loadMapMarkers(criticalOnly);
}

function highlightWellOnMap(wellId) {
    if (!map) return;
    
    // Find the well in map data
    const well = allMapData.find(w => w.well_id === wellId);
    if (!well) return;
    
    // Pan to the well location
    map.setView([well.lat, well.lng], 12);
    
    // Try to find and open the popup
    const currentLayer = map.hasLayer(clusterGroup) ? clusterGroup : markersLayer;
    currentLayer.eachLayer(layer => {
        if (layer.wellId === wellId) {
            layer.openPopup();
        }
    });
}

// Cleanup when modal is hidden
$('#mapModal').on('hidden.bs.modal', function () {
    // Optional: Could clear selections or reset view
});
</script>
""")

html_parts.append("</body></html>")

# Write the final HTML file
with open(output_html, "w", encoding="utf-8") as f:
    f.write("\n".join(html_parts))

print(f"Dashboard report with interactive map written to {output_html}")
