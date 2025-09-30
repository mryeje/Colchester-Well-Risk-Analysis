# water_table.py - MULTI-STATION PATCHED VERSION
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import warnings
import html
from datetime import datetime
import json
import requests
import os
import pathlib
import io
import re
warnings.filterwarnings('ignore')

print("=== COLCHESTER WELL DRYING RISK ANALYSIS ===")
print("GeoPandas-enhanced version with Multi-Station Hydrometric Analysis\n")

# ---------------------------
# Helper / config
# ---------------------------
well_files = ["wells_with_county_added.csv", "well_logs_with_coords.csv", "well_logs.csv", "wells.csv"]

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
DROUGHT_DRAWDOWN_M = 2.0

# MULTI-STATION HYDROMETRIC DATA CONFIGURATION
# Major river systems in Colchester County with WSC stations
COLCHESTER_STATIONS = {
    "01EO001": {  # Salmon River near Truro
        "name": "Salmon River near Truro", 
        "critical_threshold": 1.0,  # m³/s
        "region": "Central Colchester"
    },
    "01EB001": {  # Economy River near Economy  
        "name": "Economy River near Economy",
        "critical_threshold": 0.5,  # m³/s (smaller river)
        "region": "North Colchester"
    },
    "01ED002": {  # Great Village River at Great Village
        "name": "Great Village River at Great Village", 
        "critical_threshold": 0.8,  # m³/s
        "region": "North-Central Colchester"
    }
}

# Backup configuration if primary stations unavailable
BACKUP_STATIONS = {
    "01EF001": "Portapique River near Portapique",  # Backup north coast
    "01EG001": "French River near French River"    # Backup central
}

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
    
    "WYDEPTHENDOFTEST": "DEPTH",
    # static water level
    "WYSTATICLEVEL": "STATIC_WATER_LEVEL",
    
    "WYDEPTHTOWATERBEFOREPUMP": "STATIC_WATER_LEVEL",
    "WYDEPTHTOWATERAFTERPUMP": "STATIC_WATER_LEVEL",
    # yield
    "WYESTIMATEDYIELD": "YIELD",
    "WYRATE": "YIELD",
    
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

# --- INSERT THIS BLOCK BEFORE COLUMN RENAMING ---
# 1. Identify which reliable, specific columns are present
available_specific_cols = {
    "WYSTATICLEVEL": "STATIC_WATER_LEVEL",
    "WYRATE": "YIELD"
}

# 2. If the reliable source exists, drop the generic, less reliable target column if it also exists.
# This ensures the rename will succeed and use the specific source data.
for specific_col, standard_col in available_specific_cols.items():
    if specific_col in wells.columns and standard_col in wells.columns:
        print(f"Conflicting generic column '{standard_col}' found. Dropping to ensure '{specific_col}' is used.")
        # Drop the incorrect generic column. The correct specific column will be renamed next.
        wells = wells.drop(columns=[standard_col])

# --- YOUR ORIGINAL RENAMING BLOCK FOLLOWS HERE ---
# Apply mapping for any matching columns
available_cols = set(wells.columns)
rename_map = {old: new for old, new in col_map.items() if old in available_cols and new not in wells.columns}
if rename_map:
    wells = wells.rename(columns=rename_map)
    # ... (rest of your original code)
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


# --- CORRECTED UNIT CONVERSION PATCH ---
if 'WYSTATICLEVEL' in wells.columns:
    # Force imperial SWL to the reliable source (WYSTATICLEVEL is in FEET)
    wells['STATIC_WATER_LEVEL'] = pd.to_numeric(wells['WYSTATICLEVEL'], errors='coerce')
    wells['STATIC_WATER_LEVEL_ORIGINAL'] = pd.to_numeric(wells['WYSTATICLEVEL'], errors='coerce')
    
    # Convert feet to meters for the metric column (1 foot = 0.3048 meters)
    wells['current_water_level_m_observed'] = wells['STATIC_WATER_LEVEL'] * 0.3048
    
    print("✅ Corrected SWL: STATIC_WATER_LEVEL (ft) from WYSTATICLEVEL, converted to current_water_level_m_observed (m)")

if 'WYRATE' in wells.columns:
    # WYRATE is in GPM, convert to L/min (1 GPM = 3.78541 L/min)
    wells['YIELD'] = pd.to_numeric(wells['WYRATE'], errors='coerce')
    wells['YIELD_LMIN'] = wells['YIELD'] * 3.78541
    print("✅ Converted YIELD from GPM to L/min")


# PATCH 2: STATIC WATER LEVEL VALIDATION AND CORRECTION
# ---------------------------
print("\n=== STATIC WATER LEVEL VALIDATION ===")
coord_columns = [col for col in wells.columns if any(term in col.upper() for term in ['X', 'Y', 'EAST', 'NORTH', 'LAT', 'LON'])]
print(f"Coordinate-related columns found: {coord_columns}")

# Focus on the key coordinate columns
key_coords = ['X', 'Y']
for col in key_coords:
    if col in wells.columns:
        # Convert to numeric first to avoid the error
        numeric_col = pd.to_numeric(wells[col], errors='coerce')
        valid_coords = numeric_col.notna().sum()
        print(f"{col}: {valid_coords} valid numeric values out of {len(wells)}")
        if valid_coords > 0:
            print(f"  Range: {numeric_col.min():.1f} to {numeric_col.max():.1f}")
            print(f"  Sample values: {numeric_col.dropna().head(5).tolist()}")

# ---------------------------
# PATCH 1: IMPROVED COLCHESTER COUNTY FILTERING

# Define all Colchester County municipalities and communities
colchester_places = {
    # Incorporated Municipalities
    'municipality of the county of colchester', 'county of colchester', 'colchester county',
    'town of truro', 'truro',
    'town of stewiacke', 'stewiacke', 
    'village of bible hill', 'bible hill',
    'village of tatamagouche', 'tatamagouche',
    'millbrook first nation',
    
    # County Subdivisions
    'colchester subdivision a', 'colchester subdivision b', 'colchester subdivision c',
    
    # Rural Communities and Districts
    'alton', 'brentwood', 'brookfield', 'hilden', 'onslow', 'masstown', 
    'glenholme', 'little dyke', 'great village', 'highland village', 
    'portapique', 'five houses', 'bass river', 'upper economy', 'cove road', 
    'economy', 'carrs brook', 'lower economy', 'five islands', 'folly lake',
    'kemptown', 'bayhead', 'brule', 'gays river', 'green oaks', 'beaver brook',
    'old barns', 'truro heights', 'oliver', 'west new annan', 'central new annan',
    'the falls', 'balmoral mills', 'east earltown', 'green creek', 'eastville',
    'north river', 'nuttby', 'earltown', 'denmark', 'newton mills',
    'upper onslow', 'onslow mountain', 'mccallum settlement', 'upper north river',
    'central north river', 'upper brookside', 'upper kemptown', 'riversdale',
    
    # Additional Communities
    'acadian mines', 'belmont', 'black rock', 'burnside', 'camden', 'castlereagh',
    'cloverdale', 'coldstream', 'debert', 'east mines station', 'east stewiacke',
    'east village', 'french river', 'greenfield', 'harmony', 'lanesville',
    'londonderry', 'lornevale', 'lynn', 'montrose', 'pleasant hills', 'princeport',
    'salmon river', 'sand point', 'south branch', 'union', 'valley', 
    'west st. andrews', 'wittenburg'
}

print(f"Total wells before filtering: {len(wells)}")

# Try multiple filtering approaches
filtered_wells = None
filter_method = "none"

# Method 1: Check COUNTY column
if "COUNTY" in wells.columns:
    county_mask = wells["COUNTY"].astype(str).str.contains("colchester", case=False, na=False)
    if county_mask.any():
        filtered_wells = wells[county_mask].copy()
        filter_method = "COUNTY column"
        print(f"Found {county_mask.sum()} wells using COUNTY column filter")

# Method 2: Check MUNICIPALITY column  
if filtered_wells is None and "MUNICIPALITY" in wells.columns:
    muni_mask = pd.Series([False] * len(wells))
    for place in colchester_places:
        place_mask = wells["MUNICIPALITY"].astype(str).str.contains(place, case=False, na=False)
        muni_mask = muni_mask | place_mask
    
    if muni_mask.any():
        filtered_wells = wells[muni_mask].copy()
        filter_method = "MUNICIPALITY column"
        print(f"Found {muni_mask.sum()} wells using MUNICIPALITY column filter")

# Method 3: Check CIVIC_ADDRESS or ADDRESS for Colchester places
if filtered_wells is None:
    for addr_col in ["CIVIC_ADDRESS", "ADDRESS", "LOCATION"]:
        if addr_col in wells.columns:
            addr_mask = pd.Series([False] * len(wells))
            for place in colchester_places:
                place_mask = wells[addr_col].astype(str).str.contains(place, case=False, na=False)
                addr_mask = addr_mask | place_mask
            
            if addr_mask.any():
                filtered_wells = wells[addr_mask].copy()
                filter_method = f"{addr_col} column"
                print(f"Found {addr_mask.sum()} wells using {addr_col} filter")
                break

# Method 4: Manual exclusion of obvious non-Colchester places
if filtered_wells is None:
    print("No direct Colchester filtering worked. Attempting exclusion filter...")
    
    # Known non-Colchester places that appeared in your JSON
    exclude_places = [
        'halls harbour', 'beaver bank', 'hrm', 'halifax', 'havre boucher', 
        'prospect bay', 'antigonish', 'kings county', 'halifax county',
        'pictou', 'cumberland', 'hants', 'digby', 'yarmouth', 'shelburne',
        'queens', 'lunenburg', 'annapolis', 'cape breton'
    ]
    
    exclude_mask = pd.Series([False] * len(wells))
    for addr_col in ["CIVIC_ADDRESS", "ADDRESS", "LOCATION", "MUNICIPALITY"]:
        if addr_col in wells.columns:
            for exclude_place in exclude_places:
                exclude_place_mask = wells[addr_col].astype(str).str.contains(exclude_place, case=False, na=False)
                exclude_mask = exclude_mask | exclude_place_mask
    
    if exclude_mask.any():
        filtered_wells = wells[~exclude_mask].copy()
        filter_method = "exclusion filter"
        print(f"Excluded {exclude_mask.sum()} wells from known non-Colchester locations")

# Apply the filter
if filtered_wells is not None:
    wells = filtered_wells
    print(f"Successfully filtered using {filter_method}: {len(wells)} wells remain")
    
    # Show what places were found
    if "MUNICIPALITY" in wells.columns:
        print("Top municipalities found:")
        print(wells["MUNICIPALITY"].value_counts().head(10))
    elif "CIVIC_ADDRESS" in wells.columns:
        print("Sample addresses found:")
        sample_addresses = wells["CIVIC_ADDRESS"].dropna().head(10).tolist()
        for addr in sample_addresses:
            print(f"  - {addr}")
        
else:
    print("WARNING: Could not filter to Colchester County wells!")
    print("Available columns:", wells.columns.tolist())
    
    # Show sample data to help debug
    for col in ["COUNTY", "MUNICIPALITY", "CIVIC_ADDRESS", "ADDRESS"]:
        if col in wells.columns:
            print(f"\nSample {col} values:")
            sample_values = wells[col].dropna().head(10).tolist()
            for val in sample_values:
                print(f"  - {val}")

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
# PATCH 2: STATIC WATER LEVEL VALIDATION AND CORRECTION
# ---------------------------
print("\n=== STATIC WATER LEVEL VALIDATION ===")
wells["DEPTH"] = pd.to_numeric(wells["DEPTH"], errors="coerce")
wells["STATIC_WATER_LEVEL"] = pd.to_numeric(wells["STATIC_WATER_LEVEL"], errors="coerce")



print(f"Static water level stats:")
swl_stats = wells["STATIC_WATER_LEVEL"].describe()
print(swl_stats)
print(f"Wells with static level > 100m: {(wells['STATIC_WATER_LEVEL'] > 100).sum()}")
print(f"Wells with static level > 50m: {(wells['STATIC_WATER_LEVEL'] > 50).sum()}")
print(f"Wells with static level > depth (impossible): {(wells['STATIC_WATER_LEVEL'] > wells['DEPTH']).sum()}")

print(f"\nWell depth stats for comparison:")
depth_stats = wells["DEPTH"].describe()
print(depth_stats)

# Identify and flag problematic data
problematic_swl = wells["STATIC_WATER_LEVEL"] > 100
impossible_swl = wells["STATIC_WATER_LEVEL"] > wells["DEPTH"]

if problematic_swl.any():
    print(f"\n*** DATA QUALITY WARNING ***")
    print(f"Found {problematic_swl.sum()} wells with static water levels > 100m")
    print(f"This is geologically unrealistic for Nova Scotia groundwater")
    print(f"These values may represent:")
    print(f"  - Elevation above sea level (instead of depth to water)")
    print(f"  - Corrupted/mislabeled data")
    print(f"  - Wrong units")
    
    # Show some examples
    print(f"\nExamples of problematic wells:")
    problem_wells = wells[problematic_swl][["WELL_ID", "DEPTH", "STATIC_WATER_LEVEL"]].head(5)
    print(problem_wells.to_string(index=False))

# Data correction strategy
print(f"\n=== APPLYING DATA CORRECTIONS ===")

# Strategy 1: Cap unrealistic values at 50m (typical maximum for NS)
wells["STATIC_WATER_LEVEL_ORIGINAL"] = wells["STATIC_WATER_LEVEL"].copy()
capped_count = 0

# Cap values that are impossibly deep
unrealistic_mask = wells["STATIC_WATER_LEVEL"] > 50
if unrealistic_mask.any():
    # For very deep reported static levels, assume they're elevation or corrupted
    # Use a more realistic estimate based on well depth
    wells.loc[unrealistic_mask, "STATIC_WATER_LEVEL"] = np.minimum(
        wells.loc[unrealistic_mask, "DEPTH"] * 0.3,  # Assume water at 30% of depth
        25.0  # Cap at 25m depth to water
    )
    capped_count = unrealistic_mask.sum()
    print(f"Corrected {capped_count} wells with unrealistic static water levels")

# Strategy 2: Fix impossible cases (static level > total depth)
impossible_mask = wells["STATIC_WATER_LEVEL"] > wells["DEPTH"]
if impossible_mask.any():
    # For these cases, assume static level is reasonable fraction of depth
    wells.loc[impossible_mask, "STATIC_WATER_LEVEL"] = wells.loc[impossible_mask, "DEPTH"] * 0.2
    impossible_count = impossible_mask.sum()
    print(f"Fixed {impossible_count} wells where static level > total depth")

print(f"Corrected static water level stats:")


# =============================================================================
# ADD UNIT CONVERSIONS HERE - INSERT THIS BLOCK
# =============================================================================

print("\n=== APPLYING UNIT CONVERSIONS ===")

# Convert all depth measurements from feet to meters
# DEPTH is in feet, convert to meters
if 'DEPTH' in wells.columns:
    wells['DEPTH_M'] = wells['DEPTH'] * 0.3048
    print(f"✅ Converted DEPTH from feet to meters")

# Ensure static water level metric conversion is correct
if 'STATIC_WATER_LEVEL' in wells.columns:
    if 'current_water_level_m_observed' not in wells.columns:
        wells['current_water_level_m_observed'] = wells['STATIC_WATER_LEVEL'] * 0.3048
        print(f"✅ Created metric static water level from feet")
    else:
        # Double-check the conversion
        wells['current_water_level_m_observed'] = wells['STATIC_WATER_LEVEL'] * 0.3048
        print(f"✅ Verified metric static water level conversion")

# Convert yield from GPM to L/min if we have yield data
if 'YIELD' in wells.columns:
    wells['YIELD_LMIN'] = wells['YIELD'] * 3.78541
    print(f"✅ Converted YIELD from GPM to L/min")

# Update yield categorization to use L/min
if 'YIELD_LMIN' in wells.columns:
    wells['yield_category'] = wells['YIELD_LMIN'].apply(
        lambda x: "Low yield (<10 L/min)" if pd.notna(x) and x < 10
        else "Adequate yield (≥10 L/min)" if pd.notna(x) and x >= 10
        else "Unknown yield"
    )
    print("✅ Updated yield categories based on L/min values")
elif 'YIELD' in wells.columns:
    # Fallback: if no L/min conversion, use GPM with approximate threshold (10 L/min ≈ 2.64 GPM)
    wells['yield_category'] = wells['YIELD'].apply(
        lambda x: "Low yield (<2.64 GPM)" if pd.notna(x) and x < 2.64
        else "Adequate yield (≥2.64 GPM)" if pd.notna(x) and x >= 2.64
        else "Unknown yield"
    )
    print("✅ Updated yield categories based on GPM values")

print("Unit conversions completed successfully")

# =============================================================================
# END OF UNIT CONVERSION BLOCK
# =============================================================================
# AQUIFER CLASSIFICATION DEBUGGING
print("\n=== AQUIFER CLASSIFICATION DEBUG ===")
print(f"Looking for shapefile files:")
print(f"  Bedrock shapefile: {bedrock_shp} - exists: {os.path.exists(bedrock_shp)}")  
print(f"  Surficial shapefile: {surficial_shp} - exists: {os.path.exists(surficial_shp)}")

if ("X" in wells.columns and "Y" in wells.columns):
    wells_with_coords = wells.dropna(subset=["X", "Y"])
    print(f"Wells with valid X/Y coordinates: {len(wells_with_coords)}")
    
    if len(wells_with_coords) > 0:
        print(f"Sample coordinates:")
        for i, (_, row) in enumerate(wells_with_coords.head(3).iterrows()):
            print(f"  Well {row.get('WELL_ID', 'Unknown')}: X={row['X']}, Y={row['Y']}")

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
    wells["current_water_level_m"] = wells["STATIC_WATER_LEVEL"].copy()

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
    print("Observation file not found – using corrected static water levels only")
    wells["current_water_level_m"] = wells["STATIC_WATER_LEVEL"].copy()
except Exception as e:
    print(f"Warning: could not load/parse observation data ({e}) – using corrected static levels only")
    wells["current_water_level_m"] = wells["STATIC_WATER_LEVEL"].copy()

# ---------------------------
# 4. Pump depth estimate & buffer
# ---------------------------
def estimate_pump_depth(depth_ft):
    """Estimate pump depth in meters from well depth in feet"""
    try:
        depth_ft = float(depth_ft)
        # Convert feet to meters first
        depth_m = depth_ft * 0.3048
        # Pump is typically at 80% of depth or 2.5m from bottom, whichever is shallower
        return min(depth_m * 0.8, depth_m - 2.5)
    except Exception:
        return np.nan

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
# MULTI-STATION HYDROMETRIC FUNCTIONS
# ---------------------------
def fetch_multi_station_discharge(stations_dict, api_url):
    """Fetches discharge data for multiple stations and returns aggregated drought risk"""
    try:
        print(f"Fetching data for {len(stations_dict)} stations...")
        response = requests.get(api_url)
        
        if response.status_code != 200:
            print(f"Error fetching WSC data (Status: {response.status_code})")
            return None

        data = pd.read_csv(io.StringIO(response.text))
        station_id_col = ' ID'  # Column with leading space
        
        # Find discharge column
        discharge_col = None
        for col in data.columns:
            if 'Discharge' in col and '(cms)' in col:
                discharge_col = col
                break
        
        if station_id_col not in data.columns or discharge_col is None:
            print(f"Error: Required columns not found in WSC data")
            return None

        station_results = {}
        critical_stations = 0
        total_stations = 0
        
        for station_id, station_info in stations_dict.items():
            station_data = data[data[station_id_col] == station_id]
            
            if station_data.empty:
                print(f"Station {station_id} ({station_info['name']}) not found in data")
                continue
                
            try:
                latest_flow = pd.to_numeric(station_data[discharge_col], errors='coerce').dropna().iloc[-1]
                threshold = station_info['critical_threshold']
                is_critical = latest_flow < threshold
                
                station_results[station_id] = {
                    'flow': latest_flow,
                    'threshold': threshold, 
                    'critical': is_critical,
                    'name': station_info['name']
                }
                
                if is_critical:
                    critical_stations += 1
                total_stations += 1
                
                status = "CRITICAL" if is_critical else "Normal"
                print(f"  {station_info['name']}: {latest_flow:.2f} mÂ³/s ({status})")
                
            except Exception as e:
                print(f"Error processing station {station_id}: {e}")
                continue
        
        if total_stations == 0:
            return None
            
        # Calculate regional drought risk
        critical_ratio = critical_stations / total_stations
        print(f"\nRegional Analysis: {critical_stations}/{total_stations} stations critical ({critical_ratio:.1%})")
        
        return {
            'stations': station_results,
            'critical_ratio': critical_ratio,
            'total_stations': total_stations
        }
        
    except Exception as e:
        print(f"Error in multi-station analysis: {e}")
        return None

def apply_regional_drought_stress(wells_df, station_data):
    """Apply drought stress based on regional hydrometric conditions"""
    
    if station_data is None:
        # Fallback to default
        wells_df["drought_drawdown_m"] = DROUGHT_DRAWDOWN_M
        print(f"No station data - applying default {DROUGHT_DRAWDOWN_M}m stress test")
        return "DEFAULT"
    
    critical_ratio = station_data['critical_ratio']
    
    # Graduated drought stress based on regional conditions
    if critical_ratio >= 0.67:  # 67%+ of stations critical
        drought_multiplier = 2.0  # Severe regional drought
        stress_level = "SEVERE"
    elif critical_ratio >= 0.33:  # 33-66% of stations critical  
        drought_multiplier = 1.5  # Moderate regional drought
        stress_level = "MODERATE"
    else:  # <33% critical
        drought_multiplier = 1.0  # Normal conditions
        stress_level = "NORMAL"
    
    wells_df["drought_drawdown_m"] = DROUGHT_DRAWDOWN_M * drought_multiplier
    
    print(f"{stress_level} regional drought conditions detected")
    print(f"Applying {DROUGHT_DRAWDOWN_M * drought_multiplier:.1f}m drought stress test")
    
    return stress_level

# ---------------------------
# 5. Yield adjustment if present
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

# AQUIFER CLASSIFICATION DEBUGGING
print("\n=== AQUIFER CLASSIFICATION DEBUG ===")
print(f"Looking for shapefile files:")
print(f"  Bedrock shapefile: {bedrock_shp} - exists: {os.path.exists(bedrock_shp)}")  
print(f"  Surficial shapefile: {surficial_shp} - exists: {os.path.exists(surficial_shp)}")

# Initialize with consistent data types to prevent dtype promotion errors
wells["aquifer_type"] = "Unknown"  # String, not object with mixed types
wells["latitude"] = 0.0  # Float64, not mixed NaN/float
wells["longitude"] = 0.0  # Float64, not mixed NaN/float

print("DEBUG: Initialized aquifer_type as string, lat/lng as float64")

if ("X" in wells.columns and "Y" in wells.columns) and pd.notna(wells["X"]).any() and pd.notna(wells["Y"]).any():
    wells_with_coords = wells.dropna(subset=["X", "Y"])
    print(f"Wells with valid X/Y coordinates: {len(wells_with_coords)}")
    
    if len(wells_with_coords) > 0:
        print(f"Sample coordinates:")
        for i, (_, row) in enumerate(wells_with_coords.head(3).iterrows()):
            print(f"  Well {row.get('WELL_ID', 'Unknown')}: X={row['X']}, Y={row['Y']}")
    
    try:
        print("DEBUG: Starting aquifer spatial analysis...")
        
        # ** PATCH: Ensure X and Y are numeric, coercing any non-numeric values to NaN **
        wells["X"] = pd.to_numeric(wells["X"], errors='coerce') 
        wells["Y"] = pd.to_numeric(wells["Y"], errors='coerce') 
        
        # ** FINAL GEO FIX: Create a temporary DataFrame for spatial analysis from clean rows **
        wells_clean = wells.dropna(subset=["X", "Y"]).copy()
        print(f"DEBUG: Wells with clean coordinates: {len(wells_clean)}")
        
        # Skip if no rows are left with valid coordinates
        if wells_clean.empty:
            print("Warning: All wells lacked valid X/Y coordinates for GeoPandas analysis.")
            raise Exception("No valid coordinates")
            
        print("DEBUG: Creating GeoDataFrame...")
        # Create GeoDataFrame from the clean data
        wells_gdf = gpd.GeoDataFrame(
            wells_clean,
            geometry=gpd.points_from_xy(wells_clean["X"], wells_clean["Y"]),
            crs=utm_crs
        ).to_crs(wgs84)
        print(f"DEBUG: Created GeoDataFrame with {len(wells_gdf)} wells")
        
        # Extract latitude and longitude from geometry
        wells_gdf["longitude"] = wells_gdf.geometry.x
        wells_gdf["latitude"] = wells_gdf.geometry.y
        print("DEBUG: Extracted lat/lng coordinates")

        print("DEBUG: Loading shapefile data...")
        # Load shapefiles and reproject to WGS84
        bedrock = gpd.read_file(bedrock_shp).to_crs(wgs84)
        surficial = gpd.read_file(surficial_shp).to_crs(wgs84)
        print(f"DEBUG: Loaded bedrock polygons: {len(bedrock)}, surficial polygons: {len(surficial)}")

        print("DEBUG: Performing spatial joins...")
        # Spatial join: bedrock
        wb = gpd.sjoin(wells_gdf, bedrock[["geometry"]], how="left", predicate="within")
        bedrock_matches = wb["index_right"].notna().sum()
        print(f"DEBUG: Wells matching bedrock polygons: {bedrock_matches}")
        
        # -----------------------------------------------------------------------------------------
        # --- Section 6b: Fallback Textual & Numeric Aquifer Classification ---
        # -----------------------------------------------------------------------------------------

        print("\n--- Applying Fallback Textual & Numeric Aquifer Classification ---")

        # Define keywords indicative of bedrock for case-insensitive searching
        BEDROCK_KEYWORDS = [
            'SLATE', 'SHALE', 'GRANITE', 'BEDROCK', 'METAMORPHIC', 'IGNEOUS', 
            'SEDIMENTARY', 'GNEISS', 'SCHIST', 'QUARTZITE', 'LITHIFIED'
        ]
        SEARCH_COLS = ["COMMENTS", "LITHOLOGY"] # Columns to search for keywords

        # 1. Filter for wells that failed spatial classification (Unknown or None)
        # This FIX ensures we capture both "Unknown" (from init/no-coords) and "None" (from spatial miss)
        unknown_or_none_mask = wells["aquifer_type"].isin(["Unknown", "None"])

        # --- 2. Check for Textual keyword matches ---
        text_match_mask = pd.Series(False, index=wells.index)
        keyword_pattern = '|'.join(BEDROCK_KEYWORDS)

        if any(col in wells.columns for col in SEARCH_COLS):
            for col in SEARCH_COLS:
                if col in wells.columns:
                    # Convert column to string (to safely handle NaN/missing data) and search, case-insensitive
                    col_match = wells[col].astype(str).str.contains(
                        keyword_pattern, 
                        case=False, 
                        na=False
                    )
                    # Combine matches from all search columns
                    text_match_mask = text_match_mask | col_match

        # --- 3. Check for Numeric DEPTH_TO_BEDROCK (New Definitive Fallback) ---
        numeric_bedrock_mask = pd.Series(False, index=wells.index)
        if "DEPTHTOBEDROCK" in wells.columns:
            # A positive, recorded depth to bedrock is definitive proof of a bedrock well.
            # We must ensure the column is numeric and check for non-NA and value > 0.
            wells["DEPTHTOBEDROCK"] = pd.to_numeric(wells["DEPTHTOBEDROCK"], errors='coerce')
            numeric_bedrock_mask = wells["DEPTHTOBEDROCK"].notna() & (wells["DEPTHTOBEDROCK"] > 0)


        # --- 4. Apply the Fallback ---
        # Wells must be UNCLASSIFIED AND (have a text match OR a numeric depth to bedrock)
        fallback_condition = unknown_or_none_mask & (text_match_mask | numeric_bedrock_mask)
        fallback_count = fallback_condition.sum()

        if fallback_count > 0:
            wells.loc[fallback_condition, "aquifer_type"] = "Bedrock"
            print(f"Successfully classified {fallback_count} well(s) as 'Bedrock' using fallback logic.")
        else:
            print("No additional wells classified using fallback logic.")

        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        
        
        wb["aquifer_type"] = np.where(
            wb["index_right"].notna(),
            "Bedrock",
            None  # use Python None instead of np.nan
        ).astype(object)
        
        # drop index_right then join to surficial
        wb = wb.drop(columns=[c for c in wb.columns if c == "index_right"])
        wb = gpd.sjoin(wb, surficial[["geometry"]], how="left", predicate="within")
        surficial_matches = wb["index_right"].notna().sum()
        print(f"DEBUG: Wells matching surficial polygons: {surficial_matches}")
        
        wb.loc[wb["index_right"].notna(), "aquifer_type"] = "Surficial"
        
        # Count final aquifer classifications
        aquifer_counts = wb["aquifer_type"].value_counts(dropna=False)
        print(f"DEBUG: Final aquifer classification counts:")
        print(aquifer_counts)
        
        # Now try to merge results back to main wells dataframe
        print("DEBUG: Attempting to merge results back to main dataframe...")
        # Use index-based merge to avoid dtype issues
        classified_count = 0
        for idx in wb.index:
            if idx in wells.index:
                wells.loc[idx, "aquifer_type"] = wb.loc[idx, "aquifer_type"]
                wells.loc[idx, "latitude"] = wb.loc[idx, "latitude"] 
                wells.loc[idx, "longitude"] = wb.loc[idx, "longitude"]
                classified_count += 1
        
        print(f"DEBUG: Successfully merged {classified_count} wells back to main dataframe")
        
        final_aquifer_counts = wells["aquifer_type"].value_counts(dropna=False)
        print(f"DEBUG: Aquifer types in final wells dataframe:")
        print(final_aquifer_counts)
        
        print(f"Aquifer classification completed successfully")
        
    except Exception as e:
        # Catch and report any remaining errors
        print(f"Warning: aquifer classification failed ({e}). Setting all to 'Unknown'.")
        import traceback
        print(f"DEBUG: Full aquifer error traceback:\n{traceback.format_exc()}")
        wells["aquifer_type"] = "Unknown"
else:
    print("No X/Y coordinates found for aquifer spatial join – aquifer_type set to 'Unknown' for all wells.")

# ---------------------------
# 7. Apply Multi-Station Drought Stress Test
# ---------------------------

# Fetch Multi-Station Surface Water Data
station_data = fetch_multi_station_discharge(COLCHESTER_STATIONS, WSC_API_URL)
drought_level = apply_regional_drought_stress(wells, station_data)

# Calculate Stressed Water Level
wells["drought_water_level_m"] = wells["current_water_level_m"] + wells["drought_drawdown_m"]

# Recalculate Buffer and Risk (The final risk output is now the STRESSED risk)
wells["buffer_m_drought"] = wells["pump_depth_m"] - wells["drought_water_level_m"]
wells["drying_risk_drought"] = wells["buffer_m_drought"].apply(classify_risk)

# Overwrite Final Columns
wells["buffer_m"] = wells["buffer_m_drought"]
wells["drying_risk"] = wells["drying_risk_drought"]

# Rename columns for clarity in output
wells = wells.rename(columns={"current_water_level_m": "current_water_level_m_observed"})
wells = wells.rename(columns={"drought_water_level_m": "stressed_water_level_m"})

# ---------------------------
# 8. Create Google Maps links and location info
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
# 9. Summary stats and CSV export
# ---------------------------
print("\n=== MULTI-STATION ANALYSIS SUMMARY ===")
print(f"Total wells analyzed: {len(wells)}")
print(f"Wells updated from observation data: {updated_count}")
print(f"Regional drought level: {drought_level}")

# Add drought stress summary
drought_summary = wells["drought_drawdown_m"].describe()
print(f"\nDrought stress applied (meters):")
print(drought_summary)

# Add data quality summary
if "STATIC_WATER_LEVEL_ORIGINAL" in wells.columns:
    corrected_count = (wells["STATIC_WATER_LEVEL"] != wells["STATIC_WATER_LEVEL_ORIGINAL"]).sum()
    print(f"Wells with corrected static water levels: {corrected_count}")

risk_counts = wells["drying_risk"].value_counts(dropna=False)
print("\nRisk distribution (after drought stress):")
print(risk_counts.to_string())

# Show buffer statistics
buffer_stats = wells["buffer_m"].describe()
print(f"\nBuffer statistics (meters after drought stress):")
print(buffer_stats)

# Export CSV (include useful columns)
# Export CSV (include useful columns with proper units)
out_cols = ["WELL_ID", "CIVIC_ADDRESS", "MUNICIPALITY", "location_display", "google_maps_link", 
            "latitude", "longitude", "DEPTH", "DEPTH_M", "STATIC_WATER_LEVEL",
            "current_water_level_m_observed", "drought_drawdown_m", "stressed_water_level_m",
            "pump_depth_m", "buffer_m", "drying_risk", "YIELD", "YIELD_LMIN", "yield_category", "aquifer_type"]
# only keep existing
out_cols = [c for c in out_cols if c in wells.columns]
results = wells[out_cols].sort_values("buffer_m", ascending=True)
results.to_csv(output_csv, index=False)
print(f"\nDetailed results saved to {output_csv}")

# Replace the generate_detailed_well_report function and related code with this:

# Replace the generate_detailed_well_report function and related code with this:

import pathlib
from string import Template

# ---------------------------
# WELL REPORT TEMPLATE (Single HTML Template)
# ---------------------------

WELL_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Well Report - $well_id</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: Georgia, serif; background-color: #f8f9fa; }
        .risk-header { color: $risk_color; border-left: 4px solid $risk_color; padding-left: 15px; }
        .info-section { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .tech-details { background: #f8f9fa; border-radius: 8px; padding: 15px; margin-top: 20px; }
        .alert-custom { border-left: 4px solid $risk_color; }
        h1, h2, h3, h4 { color: #2c3e50; }
        .back-button { position: fixed; top: 20px; right: 20px; z-index: 1000; }
        @media print { .back-button { display: none; } }
        .key-finding { font-size: 1.1em; font-weight: bold; margin: 10px 0; }
        .explanation { line-height: 1.6; }
    </style>
</head>
<body>
    <a href="../index.html" class="btn btn-primary back-button">← Back to Dashboard</a>
    
    <div class="container py-4">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                
                <header class="text-center mb-4">
                    <h1>Well Risk Assessment Report</h1>
                    <h2>Well ID: $well_id</h2>
                    <p class="text-muted">Generated: $timestamp</p>
                </header>
                
                <div class="info-section alert alert-custom">
                    <div class="risk-header">
                        <h3>Your Well's Risk Status</h3>
                    </div>
                    <div class="key-finding" style="color: $risk_color;">$risk_level</div>
                    <div class="explanation">
                        $risk_explanation
                    </div>
                </div>
                
                <div class="info-section">
                    <h3>What This Means for You</h3>
                    <div class="explanation">
                        $risk_recommendations
                    </div>
                </div>
                
                <div class="info-section">
                    <h3>About Your Well</h3>
                    <div class="explanation">
                        <p><strong>Location:</strong> $location</p>
                        
                        <h4>Water Source</h4>
                        <p>$aquifer_explanation</p>
                        
                        <h4>Well Capacity</h4>
                        <p>$yield_explanation</p>
                        
                        <h4>Current Conditions Assessment</h4>
                        <p>$drought_explanation</p>
                    </div>
                </div>
                
                <div class="info-section">
                    <h3>Understanding Your Well's Safety Margin</h3>
                    <div class="explanation">
                        <p>Think of your well like a straw in a glass of water. The <strong>safety margin</strong> (or "buffer") 
                        is how much water is above the bottom of your straw (pump). If the water level drops below your straw, 
                        you'll suck air instead of water.</p>
                        
                        <p><strong>Your current safety margin: $buffer_display</strong></p>
                        
                        <ul>
                            <li><strong>Positive numbers are good</strong> - You have water above your pump</li>
                            <li><strong>Numbers near zero are concerning</strong> - Your pump is close to the water surface</li>
                            <li><strong>Negative numbers are critical</strong> - Your pump may be above the water level</li>
                        </ul>
                        
                        <p>We calculate this by looking at where your pump likely sits in the well (usually about 80% down 
                        the total depth) and comparing that to where the water level is now, including stress from drought conditions.</p>
                    </div>
                </div>
                
                <div class="info-section">
                    <h3>Warning Signs to Watch For</h3>
                    <div class="explanation">
                        <p>Contact a well professional if you notice any of these signs:</p>
                        <ul>
                            <li><strong>Reduced water pressure</strong> in your taps or shower</li>
                            <li><strong>Pump running more frequently</strong> or for longer periods</li>
                            <li><strong>Air spitting from faucets</strong> when you first turn them on</li>
                            <li><strong>Pump making unusual noises</strong> or cycling on and off rapidly</li>
                            <li><strong>Cloudy or muddy water</strong> (could indicate low water levels)</li>
                            <li><strong>Complete loss of water</strong> - turn off your pump immediately</li>
                        </ul>
                    </div>
                </div>
                
                <div class="info-section">
                    <h3>Water Conservation Tips</h3>
                    <div class="explanation">
                        <p>Whether your well is high-risk or low-risk, these conservation measures can help:</p>
                        
                        <h4>High-Impact Actions:</h4>
                        <ul>
                            <li><strong>Fix leaks immediately</strong> - A dripping tap can waste thousands of liters per year</li>
                            <li><strong>Install low-flow fixtures</strong> - Showerheads, toilets, and faucet aerators</li>
                            <li><strong>Run full loads</strong> - Only run dishwasher and washing machine when full</li>
                            <li><strong>Take shorter showers</strong> - Even 2 minutes less makes a big difference</li>
                        </ul>
                        
                        <h4>During Dry Periods:</h4>
                        <ul>
                            <li><strong>Space out water usage</strong> - Don't run multiple appliances at once</li>
                            <li><strong>Collect rainwater</strong> for gardens and outdoor use</li>
                            <li><strong>Limit lawn watering</strong> - Grass will recover, wells may not</li>
                            <li><strong>Use greywater</strong> where safe (dishwater for plants, etc.)</li>
                        </ul>
                    </div>
                </div>
                
                <div class="info-section">
                    <h3>Professional Help</h3>
                    <div class="explanation">
                        <p><strong>When to call a well professional:</strong></p>
                        <ul>
                            <li>Your well is rated as High Risk or Critical</li>
                            <li>You notice any warning signs listed above</li>
                            <li>Your water usage needs have increased significantly</li>
                            <li>It's been more than 10 years since a professional inspection</li>
                        </ul>
                        
                        <p><strong>What they can do:</strong></p>
                        <ul>
                            <li><strong>Water level measurement</strong> - Accurate current assessment</li>
                            <li><strong>Pump inspection and adjustment</strong> - Lower pump if possible</li>
                            <li><strong>Well rehabilitation</strong> - Cleaning and restoration techniques</li>
                            <li><strong>Well deepening</strong> - If geology allows</li>
                            <li><strong>Water quality testing</strong> - Ensure safety</li>
                            <li><strong>System efficiency improvements</strong> - Better pumps, pressure tanks</li>
                        </ul>
                        
                        <p><strong>Finding a professional:</strong> Look for certified well drillers or pump installers 
                        in Nova Scotia. Check with the Nova Scotia Ground Water Association or your local health authority 
                        for recommendations.</p>
                    </div>
                </div>
                
                $technical_summary
                
                <div class="info-section">
                    <h3>About This Analysis</h3>
                    <div class="explanation">
                        <p>This report is based on:</p>
                        <ul>
                            <li><strong>Well construction data</strong> from provincial records</li>
                            <li><strong>Regional drought conditions</strong> from Environment Canada monitoring stations</li>
                            <li><strong>Standard engineering assumptions</strong> about pump placement and drawdown</li>
                            <li><strong>Conservative risk assessment</strong> designed to err on the side of caution</li>
                        </ul>
                        
                        <p><strong>Limitations:</strong> This analysis makes estimates based on typical conditions and standard practices. 
                        Your specific situation may vary due to factors like actual pump depth, local geology, recent well work, 
                        or changes in water usage. For definitive assessment, consult with a qualified well professional.</p>
                        
                        <p><strong>Disclaimer:</strong> This report is for informational purposes and does not replace 
                        professional well assessment or water system inspection.</p>
                    </div>
                </div>
                
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

# ---------------------------
# HELPER FUNCTIONS FOR TEMPLATE DATA
# ---------------------------

def safe_get(row, key, default="N/A"):
    """Safely get a value from a row"""
    value = row.get(key, default)
    if pd.isna(value) or value == "nan" or value == "":
        return default
    return value

def format_number(value, unit="", decimals=1):
    """Format a number with units"""
    try:
        if pd.isna(value) or value == "nan":
            return "N/A"
        return f"{float(value):.{decimals}f}{unit}"
    except:
        return "N/A"

def get_risk_color(risk_level):
    """Get the color for a risk level"""
    if "CRITICAL" in str(risk_level):
        return "#d32f2f"
    elif "High risk" in str(risk_level):
        return "#f57c00"
    elif "Moderate risk" in str(risk_level):
        return "#fbc02d"
    return "#388e3c"

def get_risk_explanation(risk_level):
    """Get the explanation for a risk level"""
    explanations = {
        "CRITICAL": """
        <strong>CRITICAL RISK:</strong> Your well is at immediate risk of running dry. The water level may already be at or below 
        your pump depth, which means your pump may be drawing air instead of water. This requires immediate attention.
        """,
        "High": """
        <strong>HIGH RISK:</strong> Your well has a very small safety margin. During dry periods or with increased usage, 
        your well could run dry. The pump is close to the water level, leaving little room for error.
        """,
        "Moderate": """
        <strong>MODERATE RISK:</strong> Your well has some safety margin, but it's worth monitoring. During severe droughts 
        or if your water usage increases significantly, you might experience problems.
        """,
        "Low": """
        <strong>LOW RISK:</strong> Your well has a good safety margin. Even during dry periods, you should have adequate water supply.
        This doesn't mean unlimited water, but you're in a relatively secure position.
        """
    }
    
    for key, explanation in explanations.items():
        if key in str(risk_level):
            return explanation
    return "<strong>Unknown risk level.</strong>"

def get_risk_recommendations(risk_level):
    """Get recommendations based on risk level"""
    if "CRITICAL" in str(risk_level):
        return """
        <h4>Immediate Actions Needed:</h4>
        <ul>
            <li><strong>Contact a well professional immediately</strong> - Don't wait</li>
            <li><strong>Reduce water usage</strong> to absolute essentials only</li>
            <li><strong>Check your pump</strong> - If it's running but no water comes out, turn it off to prevent damage</li>
            <li><strong>Consider emergency water supply</strong> options (bottled water, neighbors, etc.)</li>
            <li><strong>Pump lowering may be needed</strong> if there's enough water deeper in the well</li>
            <li><strong>Well deepening</strong> might be required if the aquifer allows</li>
        </ul>
        """
    elif "High risk" in str(risk_level):
        return """
        <h4>Recommended Actions:</h4>
        <ul>
            <li><strong>Monitor water levels closely</strong> - Watch for changes in flow or pressure</li>
            <li><strong>Implement water conservation</strong> measures immediately</li>
            <li><strong>Get a professional assessment</strong> within the next few weeks</li>
            <li><strong>Consider pump lowering</strong> as a preventive measure</li>
            <li><strong>Install a low-water alarm</strong> to warn before the well runs dry</li>
            <li><strong>Have an emergency water plan</strong> ready</li>
        </ul>
        """
    elif "Moderate risk" in str(risk_level):
        return """
        <h4>Recommended Actions:</h4>
        <ul>
            <li><strong>Practice water conservation</strong> during dry periods</li>
            <li><strong>Monitor your well</strong> for signs of declining water levels</li>
            <li><strong>Plan for professional assessment</strong> within the next year</li>
            <li><strong>Be prepared</strong> with water conservation measures during droughts</li>
            <li><strong>Consider efficiency upgrades</strong> for appliances and fixtures</li>
        </ul>
        """
    return """
        <h4>Recommended Actions:</h4>
        <ul>
            <li><strong>Continue regular maintenance</strong> of your well system</li>
            <li><strong>Monitor periodically</strong> for any changes</li>
            <li><strong>Practice reasonable water conservation</strong> as good stewardship</li>
            <li><strong>Consider a professional checkup</strong> every 3-5 years</li>
        </ul>
        """

def get_aquifer_explanation(aquifer_type):
    """Get explanation for aquifer type"""
    if aquifer_type == "Bedrock":
        return """
        Your well draws water from <strong>bedrock aquifers</strong> - water stored in cracks and fractures in solid rock. 
        These wells can be very reliable but may be more susceptible to seasonal variations and take longer to recover 
        after heavy use.
        """
    elif aquifer_type == "Surficial":
        return """
        Your well draws water from <strong>surficial aquifers</strong> - water stored in soil, sand, and gravel near the surface. 
        These wells often recharge more quickly from rainfall but may be more sensitive to dry periods.
        """
    return """
        The aquifer type for your well is not determined in our records. This could be either bedrock (water from rock fractures) 
        or surficial (water from soil/sand layers). A well professional can help determine this.
        """

def get_yield_explanation(yield_val):
    """Get explanation for well yield"""
    if yield_val == "N/A":
        return "The flow rate (yield) of your well is not available in our records."
    
    try:
        yield_num = float(yield_val)
        if yield_num < 5:
            return f"""
            <strong>Low Yield Well:</strong> Your well produces {yield_num} L/min, which is considered low. This means 
            you need to be especially careful about water usage and may need to spread out high-demand activities 
            (like laundry, showers) throughout the day.
            """
        elif yield_num < 10:
            return f"""
            <strong>Moderate Yield Well:</strong> Your well produces {yield_num} L/min, which is adequate for most 
            household needs with reasonable usage patterns.
            """
        return f"""
            <strong>Good Yield Well:</strong> Your well produces {yield_num} L/min, which should meet typical 
            household water needs comfortably.
            """
    except:
        return f"Your well's yield is recorded as {yield_val} L/min."

def get_drought_explanation(drought_stress):
    """Get explanation for drought stress"""
    try:
        drought_value = float(drought_stress)
        if drought_value > 3.0:
            return f"""
            <strong>Severe Regional Drought Conditions:</strong> Our analysis shows that most area rivers and streams 
            are experiencing critically low flows. We've applied an additional {drought_value:.1f} meters of stress testing 
            to your well to simulate these harsh conditions.
            """
        elif drought_value > 2.5:
            return f"""
            <strong>Moderate Regional Drought Conditions:</strong> Some area waterways are showing stress. We've applied 
            an additional {drought_value:.1f} meters of stress testing to simulate continued dry conditions.
            """
        return f"""
            <strong>Normal Conditions:</strong> Area waterways are at normal levels. We've applied a standard 
            {drought_value:.1f} meters of drought stress testing as a precautionary measure.
            """
    except:
        return "We've applied standard drought stress testing to simulate dry conditions."

def get_technical_summary(row):
    """Generate technical details section"""
    return f"""
    <div class="tech-details">
        <h4>Technical Details (for reference)</h4>
        <div class="row">
            <div class="col-md-6">
                <p><strong>Total Well Depth:</strong> {format_number(row.get('DEPTH'), 'm')}</p>
                <p><strong>Original Static Water Level:</strong> {format_number(row.get('STATIC_WATER_LEVEL'), 'm')}</p>
                <p><strong>Current Water Level:</strong> {format_number(row.get('current_water_level_m_observed'), 'm')}</p>
                <p><strong>Stressed Water Level:</strong> {format_number(row.get('stressed_water_level_m'), 'm')}</p>
            </div>
            <div class="col-md-6">
                <p><strong>Estimated Pump Depth:</strong> {format_number(row.get('pump_depth_m'), 'm')}</p>
                <p><strong>Safety Buffer:</strong> {format_number(row.get('buffer_m'), 'm')}</p>
                <p><strong>Drought Stress Applied:</strong> {format_number(row.get('drought_drawdown_m'), 'm')}</p>
                <p><strong>Well Yield:</strong> {format_number(row.get('YIELD'), ' L/min')}</p>
            </div>
        </div>
        <p><em>Note: Water levels are measured as depth below ground surface. Larger numbers mean deeper water.</em></p>
    </div>
    """

def generate_well_report_from_template(row, template_str):
    """Generate a well report using the template"""
    
    # Clean well_id
    well_id = str(row.get("WELL_ID", "Unknown")).replace('<a href="well_reports/well_', '').replace('.html" target="_blank">', '').replace(' 📊</a>', '')
    
    # Prepare all template variables
    template_data = {
        'well_id': well_id,
        'timestamp': datetime.now().strftime('%B %d, %Y at %I:%M %p'),
        'risk_level': safe_get(row, 'drying_risk'),
        'risk_color': get_risk_color(row.get('drying_risk')),
        'risk_explanation': get_risk_explanation(row.get('drying_risk')),
        'risk_recommendations': get_risk_recommendations(row.get('drying_risk')),
        'location': safe_get(row, 'location_display', 'Location not specified'),
        'aquifer_explanation': get_aquifer_explanation(safe_get(row, 'aquifer_type', 'Unknown')),
        'yield_explanation': get_yield_explanation(safe_get(row, 'YIELD')),
        'drought_explanation': get_drought_explanation(safe_get(row, 'drought_drawdown_m')),
        'buffer_display': format_number(row.get('buffer_m'), ' meters'),
        'technical_summary': get_technical_summary(row)
    }
    
    # Use Template for safe substitution
    template = Template(template_str)
    return template.safe_substitute(template_data)

# ---------------------------
# GENERATE REPORTS USING TEMPLATE
# ---------------------------

# Create directory for reports
report_dir = pathlib.Path("well_reports")
report_dir.mkdir(exist_ok=True)

# Generate reports for all wells using the template
print("\n=== GENERATING WELL REPORTS FROM TEMPLATE ===")
report_count = 0

# ---------------------------
# 10. Prepare map data for wells with valid coordinates
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
            <strong>Drought Stress:</strong> {safe_format(well.get('drought_drawdown_m'), '{:.1f}m')}<br>
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
# ENHANCED WELL ID LINK CREATION WITH REPORT ICON
# ---------------------------

def create_well_report_link(well_id):
    """Create a clickable well ID with report icon"""
    clean_id = str(well_id).strip()
    
    # Remove any existing HTML if it's already been processed
    if '<a href=' in clean_id:
        match = re.search(r'well_([^\.]+)\.html', clean_id)
        if match:
            clean_id = match.group(1)
        else:
            clean_id = re.sub(r'<[^>]+>', '', clean_id).strip()
    
    # Link to single template with URL parameter
    return f'<a href="well_report_template.html?id={clean_id}" target="_blank" class="well-link" title="View detailed report">{clean_id} 📊</a>'

# ---------------------------
# UPDATED HTML TABLE PREPARATION
# ---------------------------

# Prepare the main table for client-side rendering
all_wells_display = results.copy()

# === CHECK FOR DUPLICATES IN SOURCE DATA ===
print("Checking for duplicate columns in source data...")
print(f"Initial columns in results: {list(results.columns)}")
if results.columns.duplicated().any():
    print("❌ RESULTS has duplicate columns!")
    duplicate_cols = results.columns[results.columns.duplicated()].tolist()
    print(f"Duplicate columns in results: {duplicate_cols}")
    # Remove duplicates from results
    results = results.loc[:, ~results.columns.duplicated()]
    all_wells_display = results.copy()
    print("✅ Removed duplicate columns from results")

# Apply the new link creation function
print("Creating well ID links with report icons...")
if 'WELL_ID' in all_wells_display.columns:
    all_wells_display['WELL_ID'] = all_wells_display['WELL_ID'].apply(create_well_report_link)
    print(f"Created report links for {len(all_wells_display)} wells")

# Rest of table preparation...
if 'google_maps_link' in all_wells_display.columns:
    all_wells_display['Map Link'] = all_wells_display['google_maps_link'].apply(
        lambda x: f'<a href="{x}" target="_blank" class="map-link">🗺️ Map</a>' if x else '—'
    )
    
    # Define columns for the final table
    drop_cols = ['google_maps_link', 'CIVIC_ADDRESS', 'MUNICIPALITY', 'latitude', 'longitude']
    display_cols = ['WELL_ID', 'location_display', 'Map Link'] + [
        c for c in all_wells_display.columns 
        if c not in drop_cols + ['WELL_ID', 'location_display', 'Map Link']
    ]
    
    # Remove duplicates from display_cols
    unique_display_cols = []
    seen_cols = set()
    for col in display_cols:
        if col not in seen_cols:
            unique_display_cols.append(col)
            seen_cols.add(col)
        else:
            print(f"Removing duplicate column from display: {col}")
    display_cols = unique_display_cols
    
    all_wells_display = all_wells_display[display_cols]

# === FINAL DUPLICATE CHECK BEFORE JSON ===
print("Final check for duplicate columns before JSON conversion...")
print(f"Columns before JSON: {list(all_wells_display.columns)}")

if all_wells_display.columns.duplicated().any():
    print("❌ DUPLICATE COLUMNS STILL EXIST! Using emergency cleanup...")
    # Emergency cleanup - rebuild with unique columns only
    unique_cols = []
    seen = set()
    for col in all_wells_display.columns:
        if col not in seen:
            unique_cols.append(col)
            seen.add(col)
        else:
            print(f"Emergency removal of duplicate: {col}")
    all_wells_display = all_wells_display[unique_cols]

# Ensure all column names are strings
all_wells_display.columns = [str(col) for col in all_wells_display.columns]

print(f"✅ Final column count: {len(all_wells_display.columns)}")
print(f"✅ Final columns: {list(all_wells_display.columns)}")


# Convert the full dataset to JSON
try:
    all_wells_json = all_wells_display.to_json(orient="records")
    print("✅ JSON conversion successful")
except Exception as e:
    print(f"❌ JSON conversion failed: {e}")
    # Last resort: create a simple DataFrame with just essential columns
    essential_cols = ['WELL_ID', 'location_display', 'Map Link', 'DEPTH', 'DEPTH_M', 'STATIC_WATER_LEVEL', 'current_water_level_m_observed', 'YIELD', 'YIELD_LMIN', 'drying_risk', 'yield_category', 'aquifer_type']
    essential_cols = [col for col in essential_cols if col in all_wells_display.columns]
    all_wells_display_simple = all_wells_display[essential_cols]
    all_wells_json = all_wells_display_simple.to_json(orient="records")
    print("✅ Used fallback JSON conversion with essential columns only")

# Create column definitions with proper units for DataTables
def format_column_title(col_name):
    """Add units to column titles for better readability"""
    unit_map = {
        'DEPTH': 'Well Depth',
        'DEPTH_M': 'Well Depth',
        'STATIC_WATER_LEVEL': 'Static Water Level',
        'current_water_level_m_observed': 'Current Water Level',
        'drought_drawdown_m': 'Drought Stress',
        'stressed_water_level_m': 'Stressed Water Level',
        'pump_depth_m': 'Pump Depth',
        'buffer_m': 'Safety Buffer',
        'YIELD': 'Yield',
        'YIELD_LMIN': 'Yield',
        'drying_risk': 'Risk Level',
        'yield_category': 'Yield Category',
        'aquifer_type': 'Aquifer Type',
        'location_display': 'Location',
        'WELL_ID': 'Well ID',
        'Map Link': 'Map',
    }
    return unit_map.get(col_name, col_name)

# Create the column definitions for DataTables
datatables_columns = json.dumps([{"data": col, "title": format_column_title(col)} for col in all_wells_display.columns])

# Save the data payload separately to keep the HTML file small
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

# ---------------------------
# 11. Create HTML report (Dashboard Style with DataTables and Interactive Map)
# ---------------------------
print("Generating HTML dashboard with multi-station analysis...")

html_parts = []
html_parts.append("<!doctype html>")
html_parts.append("<html lang='en'><head><meta charset='utf-8'><title>Colchester Multi-Station Well Risk Report</title>")

# CSS and JS libraries
html_parts.append("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.bootstrap5.min.css">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css" />
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css" />
<style><style>
body {{ background:#f8fafc; color:#111; }}
h1,h2 {{ color:#0b4d78; margin-top:20px; }}
.card {{ border-radius:10px; box-shadow:0 1px 6px rgba(0,0,0,0.08); margin-bottom:20px; }}
.kpi-card {{ text-align:center; padding:20px; }}
.kpi-value {{ font-size:1.5rem; font-weight:bold; color:#0b4d78; }}
.drought-status {{ padding:10px; border-radius:5px; margin-bottom:20px; }}
.drought-severe {{ background-color:#ffebee; border-left:4px solid #f44336; }}
.drought-moderate {{ background-color:#fff3e0; border-left:4px solid #ff9800; }}
.drought-normal {{ background-color:#e8f5e8; border-left:4px solid #4caf50; }}
.drought-default {{ background-color:#f5f5f5; border-left:4px solid #9e9e9e; }}
.nav-tabs .nav-link.active {{ background:#0b4d78; color:#fff; }}
.map-link {{ color:#0b4d78; text-decoration:none; }}
.map-link:hover {{ text-decoration:underline; }}
.dataTables_wrapper .dt-buttons {{ margin-bottom:10px; }}
table.dataTable thead th {{ white-space: nowrap; }}

.well-link {{
    color: #0b4d78;
    text-decoration: none;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}}

.well-link:hover {{
    color: #1976d2;
    text-decoration: underline;
}}

table.dataTable td .well-link {{
    padding: 2px 4px;
    border-radius: 3px;
    transition: all 0.2s ease;
}}

table.dataTable td .well-link:hover {{
    background-color: rgba(11, 77, 120, 0.1);
}}

/* Map Modal Styles - FIXED */
#mapModal .modal-dialog {{ 
    max-width: 95vw;
    margin: 1.75rem auto;
}}

#mapModal .modal-content {{ 
    height: 85vh;
}}

#mapModal .modal-body {{ 
    padding: 0; 
    height: calc(100% - 60px);
    position: relative;
}}

#wellMap {{ 
    height: 100% !important; 
    width: 100% !important;
    min-height: 400px;
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
}}

/* Mobile optimizations */
@media (max-width: 768px) {{
    .map-controls {{ 
        position: relative; 
        margin-bottom: 10px; 
    }}
    .map-legend {{
        position: relative;
        margin-top: 10px;
    }}
    #wellMap {{ 
        height: 70vh !important;
        min-height: 400px;
    }}
    
    /* Make DataTables search larger and more centered on mobile */
    .dataTables_wrapper .dataTables_filter {{
        text-align: center !important;
        float: none !important;
        margin: 15px 0 !important;
    }}
    
    .dataTables_wrapper .dataTables_filter input {{
        width: 100% !important;
        max-width: 400px !important;
        margin: 10px auto !important;
        padding: 12px 15px !important;
        font-size: 16px !important;
        border: 2px solid #0b4d78 !important;
        border-radius: 8px !important;
        display: block !important;
    }}
    
    .dataTables_wrapper .dataTables_filter label {{
        display: block !important;
        width: 100% !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        margin-bottom: 5px !important;
    }}
    
    /* Center and stack DataTables controls on mobile */
    .dataTables_wrapper .dataTables_length {{
        text-align: center !important;
        float: none !important;
        margin: 10px 0 !important;
    }}
    
    .dataTables_wrapper .dataTables_info {{
        text-align: center !important;
        float: none !important;
        padding-top: 10px !important;
    }}
    
    .dataTables_wrapper .dataTables_paginate {{
        text-align: center !important;
        float: none !important;
        margin-top: 10px !important;
    }}
    
    /* Make buttons stack nicely on mobile */
    .dataTables_wrapper .dt-buttons {{
        text-align: center !important;
        margin: 10px 0 !important;
    }}
    
    .dataTables_wrapper .dt-buttons .btn {{
        margin: 3px !important;
        font-size: 14px !important;
    }}
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

# Title + timestamp with multi-station indicator
html_parts.append(f"<h1 class='mb-3'>Colchester County Well Drying Risk Dashboard</h1>")
html_parts.append(f"<p class='text-muted'>Multi-Station Hydrometric Analysis | Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Wells mapped: {len(map_wells_data)}</p>")

# Add drought status indicator
drought_class_map = {
    "SEVERE": "drought-severe",
    "MODERATE": "drought-moderate", 
    "NORMAL": "drought-normal",
    "DEFAULT": "drought-default"
}
drought_css_class = drought_class_map.get(drought_level, "drought-default")

drought_messages = {
    "SEVERE": "🚨 SEVERE DROUGHT CONDITIONS - 67%+ of monitoring stations show critical low flows",
    "MODERATE": "⚠️ MODERATE DROUGHT CONDITIONS - 33-66% of monitoring stations show critical low flows", 
    "NORMAL": "✅ NORMAL SURFACE WATER CONDITIONS - Wells tested under standard drought stress",
    "DEFAULT": "ℹ️ HYDROMETRIC DATA UNAVAILABLE - Conservative drought assumptions applied"
}
drought_message = drought_messages.get(drought_level, "ℹ️ HYDROMETRIC DATA UNAVAILABLE - Conservative drought assumptions applied")

html_parts.append(f"""
<div class="drought-status {drought_css_class}">
    <strong>Regional Drought Status:</strong> {drought_message}
</div>
""")

# Expandable help window
html_parts.append("""
<p>
  <button class="btn btn-outline-info btn-sm" type="button" data-bs-toggle="collapse" data-bs-target="#help-window" aria-expanded="false" aria-controls="help-window">
    How to Read This Report 📖
  </button>
  <button class="btn btn-primary btn-sm ms-2" type="button" data-bs-toggle="modal" data-bs-target="#mapModal">
    🗺️ View Interactive Map
  </button>
</p>

<p>
  
  <button class="btn btn-outline-secondary btn-sm ms-2" type="button" id="unitToggle" onclick="toggleUnits()">
    Switch to Imperial
  </button>
</p>

<div class="collapse" id="help-window">
  <div class="card card-body bg-light mb-4">
    <h4>Understanding the Multi-Station Analysis</h4>
    <p>This report uses real-time data from multiple Environment Canada hydrometric stations across Colchester County to assess regional drought conditions and apply appropriate stress testing to well risk calculations.</p>
    
    <h5>Monitoring Stations:</h5>
    <ul>
        <li><strong>Salmon River near Truro (01EO001):</strong> Central Colchester region</li>
        <li><strong>Economy River near Economy (01EB001):</strong> North coast region</li>
        <li><strong>Great Village River at Great Village (01ED002):</strong> North-central region</li>
    </ul>
    
    <h5>Drought Stress Levels:</h5>
    <ul>
        <li><strong>SEVERE:</strong> 67%+ stations critical → 4.0m additional drawdown applied</li>
        <li><strong>MODERATE:</strong> 33-66% stations critical → 3.0m additional drawdown applied</li>
        <li><strong>NORMAL:</strong> <33% stations critical → 2.0m standard drawdown applied</li>
    </ul>
    
    <h4>Understanding the Columns</h4>
    <ul>
        <li><strong>WELL_ID 📊:</strong> Click the chart icon to view a detailed, easy-to-understand report for that specific well</li>
        <li><strong>buffer_m (Buffer in Meters):</strong> Vertical distance between stressed water level and pump depth. Negative values indicate critical risk.</li>
        <li><strong>drying_risk:</strong> Risk category based on buffer after drought stress testing</li>
        <li><strong>stressed_water_level_m:</strong> Depth to water after applying regional drought drawdown</li>
        <li><strong>drought_drawdown_m:</strong> Additional drawdown applied based on current surface water conditions</li>
        <li><strong>current_water_level_m_observed:</strong> Original depth to water before drought stress</li>
        <li><strong>pump_depth_m:</strong> Estimated pump depth (80% of well depth or 2.5m from bottom)</li>
        <li><strong>DEPTH:</strong> Total drilled depth of the well</li>
        <li><strong>YIELD:</strong> Well flow rate in L/min</li>
    </ul>
    <p><strong>Note:</strong> This analysis provides a regional approach to drought risk assessment, giving more accurate results than single-station analysis for county-wide groundwater conditions.</p>
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

# Tabs
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

# Tab: All Wells
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
# Map Modal
# Map Modal
html_parts.append(f"""
<!-- Map Modal -->
<div class="modal fade" id="mapModal" tabindex="-1" aria-labelledby="mapModalLabel" aria-hidden="true">
  <div class="modal-dialog" style="max-width: 90vw; height: 80vh; margin: 2rem auto;">
    <div class="modal-content" style="height: 100%;">
      <div class="modal-header">
        <h5 class="modal-title" id="mapModalLabel">Interactive Well Risk Map - Multi-Station Analysis</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
      </div>
      <div class="modal-body" style="padding: 0; height: calc(100% - 60px); position: relative;">
        <div class="map-controls" style="position: absolute; top: 10px; right: 10px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
          <label style="margin-right: 10px; font-size: 0.9rem;">
            <input type="checkbox" id="clusterToggle" checked> Cluster markers
          </label>
          <label style="font-size: 0.9rem;">
            <input type="checkbox" id="criticalOnly"> Critical/High risk only
          </label>
        </div>
        <div id="wellMap" style="width: 100%; height: 100%; background: #e0e0e0;"></div>
        <div class="map-legend" style="position: absolute; bottom: 20px; right: 10px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); font-size: 0.85rem;">
          <strong>Risk Levels:</strong><br>
          <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: red; margin-right: 5px;"></div>
            Critical
          </div>
          <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: orange; margin-right: 5px;"></div>
            High Risk
          </div>
          <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: yellow; margin-right: 5px;"></div>
            Moderate Risk
          </div>
          <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: green; margin-right: 5px;"></div>
            Low Risk
          </div>
          <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: gray; margin-right: 5px;"></div>
            No Data
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
""")

html_parts.append("</div>") # end container

# Inject column definitions and define the data file paths
html_parts.append(f"""<script> 
const dtColumns = {datatables_columns}; 
const dataJsonFile = '{data_json_file}';
const mapDataFile = '{map_data_json_file}';
const mapCenter = [{map_center_lat}, {map_center_lng}];
</script>""")

# DataTables and Map initialization script
html_parts.append(f"""<script> 
const dtColumns = {datatables_columns}; 
const dataJsonFile = '{data_json_file}';
const mapDataFile = '{map_data_json_file}';
const mapCenter = [{map_center_lat}, {map_center_lng}];
</script>""")

# DataTables and Map initialization script
html_parts.append("""
<script>
let map = null;
let markersLayer = null;
let clusterGroup = null;
let allMapData = [];
let currentTable = null;

function findColumnIndex(columnName) {
    for (let i = 0; i < dtColumns.length; i++) {
        if (dtColumns[i].data === columnName) {
            return i;
        }
    }
    return 0;
}

$(document).ready(function() {
  fetch(dataJsonFile)
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to load data: ' + response.statusText);
        }
        return response.json();
    })
    .then(allWellsData => {
        currentTable = $('#all_wells_table').DataTable({
            data: allWellsData,
            columns: dtColumns,
            pageLength: 25,
            lengthMenu: [10, 25, 50, 100, {label: "All", value: -1}],
            dom: 'Bfrtip',
            buttons: ['copy', 'csv', 'excel', 'print'],
            scrollX: true,
            responsive: true,
            order: [[findColumnIndex('buffer_m'), 'asc']]
        });
        
        $('#all_wells_table tbody').on('click', 'tr', function() {
            if ($(this).hasClass('selected')) {
                $(this).removeClass('selected');
            } else {
                currentTable.$('tr.selected').removeClass('selected');
                $(this).addClass('selected');
                
                const data = currentTable.row(this).data();
                if (map && data && data.WELL_ID) {
                    highlightWellOnMap(data.WELL_ID);
                }
            }
        });
    })
    .catch(error => {
        console.error("Error initializing dashboard:", error);
        $('#all_wells_table').html("<p style='color:red;'>Error: Could not load well data.</p>");
    });

  fetch(mapDataFile)
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to load map data: ' + response.statusText);
        }
        return response.json();
    })
    .then(mapData => {
        allMapData = mapData;
        console.log(`Loaded ${allMapData.length} wells for mapping`);
    })
    .catch(error => {
        console.error("Error loading map data:", error);
    });

  $('#risk_table, #aq_table').DataTable({
    pageLength: 20,
    dom: 'Bfrtip',
    buttons: ['copy', 'csv', 'print'],
    scrollX: true,
    responsive: true
  });
});

$('#mapModal').on('shown.bs.modal', function () {
    console.log('=== MAP MODAL OPENED ===');
    console.log('Map object exists:', !!map);
    console.log('Map data loaded:', allMapData.length);
    
    const mapDiv = document.getElementById('wellMap');
    console.log('Map div found:', !!mapDiv);
    if (mapDiv) {
        console.log('Map div dimensions:', mapDiv.offsetWidth, 'x', mapDiv.offsetHeight);
    }
    
    if (!map) {
        setTimeout(() => {
            try {
                initializeMap();
            } catch (error) {
                console.error('Error:', error);
                alert('Map error: ' + error.message);
            }
        }, 500);
    } else {
        setTimeout(() => map.invalidateSize(), 500);
    }
});

function initializeMap() {
    const mapContainer = document.getElementById('wellMap');
    if (!mapContainer) {
        throw new Error('Map container not found');
    }
    
    console.log('Creating map...');
    
    map = L.map('wellMap').setView(mapCenter, 9);
    
    console.log('Adding tile layer...');
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);
    
    console.log('Creating marker layers...');
    
    clusterGroup = L.markerClusterGroup({
        maxClusterRadius: 50,
        spiderfyOnMaxZoom: true,
        showCoverageOnHover: false,
        disableClusteringAtZoom: 15
    });
    
    markersLayer = L.layerGroup();
    
    console.log('Loading markers...');
    loadMapMarkers();
    
    const clusterToggle = document.getElementById('clusterToggle');
    const criticalOnlyToggle = document.getElementById('criticalOnly');
    
    if (clusterToggle) {
        clusterToggle.addEventListener('change', toggleClustering);
    }
    if (criticalOnlyToggle) {
        criticalOnlyToggle.addEventListener('change', filterMarkers);
    }
    
    setTimeout(() => map.invalidateSize(), 100);
}

function loadMapMarkers(filterCritical = false) {
    console.log(`Loading markers (filter critical: ${filterCritical})...`);
    
    if (!map) {
        console.error('Map not initialized yet');
        return;
    }
    
    if (allMapData.length === 0) {
        console.warn('No map data available');
        return;
    }
    
    if (clusterGroup) clusterGroup.clearLayers();
    if (markersLayer) markersLayer.clearLayers();
    
    let filteredData = allMapData;
    if (filterCritical) {
        filteredData = allMapData.filter(well => 
            well.risk.includes('CRITICAL') || well.risk.includes('High risk')
        );
        console.log(`Filtered to ${filteredData.length} critical/high risk wells`);
    }
    
    filteredData.sort((a, b) => a.risk_priority - b.risk_priority);
    
    let markersAdded = 0;
    filteredData.forEach(well => {
        try {
            const marker = L.circleMarker([well.lat, well.lng], {
                radius: 6,
                fillColor: well.color,
                color: '#000',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            }).bindPopup(well.popup);
            
            marker.wellId = well.well_id;
            
            clusterGroup.addLayer(marker);
            markersLayer.addLayer(marker);
            markersAdded++;
        } catch (error) {
            console.error('Error adding marker:', error, well);
        }
    });
    
    console.log(`Added ${markersAdded} markers to map`);
    
    const clusterToggle = document.getElementById('clusterToggle');
    const clusterEnabled = clusterToggle ? clusterToggle.checked : true;
    
    if (clusterEnabled) {
        map.addLayer(clusterGroup);
    } else {
        map.addLayer(markersLayer);
    }
}

function toggleClustering() {
    if (!map) return;
    
    const clusterEnabled = document.getElementById('clusterToggle').checked;
    console.log(`Toggling clustering: ${clusterEnabled}`);
    
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
    console.log(`Filtering markers (critical only: ${criticalOnly})`);
    loadMapMarkers(criticalOnly);
}

function highlightWellOnMap(wellId) {
    if (!map) return;
    
    const well = allMapData.find(w => w.well_id === wellId);
    if (!well) return;
    
    map.setView([well.lat, well.lng], 12);
    
    const currentLayer = map.hasLayer(clusterGroup) ? clusterGroup : markersLayer;
    currentLayer.eachLayer(layer => {
        if (layer.wellId === wellId) {
            layer.openPopup();
        }
    });
}

let currentUnits = 'metric';
let originalMetricData = null;

function toggleUnits() {
    if (!currentTable) {
        console.error('Table not initialized yet');
        return;
    }
    
    const button = document.getElementById('unitToggle');
    
    if (!originalMetricData) {
        const allData = currentTable.rows().data().toArray();
        originalMetricData = allData.map(row => JSON.parse(JSON.stringify(row)));
    }
    
    if (currentUnits === 'metric') {
        currentUnits = 'imperial';
        button.textContent = 'Switch to Metric';
        
        currentTable.rows().every(function() {
            const row = this.data();
            
            if (row.DEPTH_M && !isNaN(parseFloat(row.DEPTH_M))) {
                row.DEPTH_M = (parseFloat(row.DEPTH_M) * 3.28084).toFixed(1);
            }
            if (row.pump_depth_m && !isNaN(parseFloat(row.pump_depth_m))) {
                row.pump_depth_m = (parseFloat(row.pump_depth_m) * 3.28084).toFixed(1);
            }
            if (row.buffer_m && !isNaN(parseFloat(row.buffer_m))) {
                row.buffer_m = (parseFloat(row.buffer_m) * 3.28084).toFixed(1);
            }
            if (row.current_water_level_m_observed && !isNaN(parseFloat(row.current_water_level_m_observed))) {
                row.current_water_level_m_observed = (parseFloat(row.current_water_level_m_observed) * 3.28084).toFixed(1);
            }
            if (row.drought_drawdown_m && !isNaN(parseFloat(row.drought_drawdown_m))) {
                row.drought_drawdown_m = (parseFloat(row.drought_drawdown_m) * 3.28084).toFixed(1);
            }
            if (row.stressed_water_level_m && !isNaN(parseFloat(row.stressed_water_level_m))) {
                row.stressed_water_level_m = (parseFloat(row.stressed_water_level_m) * 3.28084).toFixed(1);
            }
            if (row.YIELD_LMIN && !isNaN(parseFloat(row.YIELD_LMIN))) {
                row.YIELD_LMIN = (parseFloat(row.YIELD_LMIN) / 3.78541).toFixed(1);
            }
            
            this.data(row);
        });
        
        updateHeader('DEPTH_M', 'Well Depth (ft)');
        updateHeader('pump_depth_m', 'Pump Depth (ft)');
        updateHeader('buffer_m', 'Safety Buffer (ft)');
        updateHeader('current_water_level_m_observed', 'Current Water Level (ft)');
        updateHeader('drought_drawdown_m', 'Drought Stress (ft)');
        updateHeader('stressed_water_level_m', 'Stressed Water Level (ft)');
        updateHeader('YIELD_LMIN', 'Yield (GPM)');
        
    } else {
        currentUnits = 'metric';
        button.textContent = 'Switch to Imperial';
        
        currentTable.rows().every(function(idx) {
            if (originalMetricData[idx]) {
                this.data(originalMetricData[idx]);
            }
        });
        
        updateHeader('DEPTH_M', 'Well Depth (m)');
        updateHeader('pump_depth_m', 'Pump Depth (m)');
        updateHeader('buffer_m', 'Safety Buffer (m)');
        updateHeader('current_water_level_m_observed', 'Current Water Level (m)');
        updateHeader('drought_drawdown_m', 'Drought Stress (m)');
        updateHeader('stressed_water_level_m', 'Stressed Water Level (m)');
        updateHeader('YIELD_LMIN', 'Yield (L/min)');
    }
    
    currentTable.draw(false);
}

function updateHeader(columnData, newTitle) {
    const colIdx = currentTable.column(columnData + ':name').index();
    if (colIdx >= 0) {
        $(currentTable.column(colIdx).header()).text(newTitle);
    }
}
</script>
""")

html_parts.append("</body></html>")

# Write the final HTML file
with open(output_html, "w", encoding="utf-8") as f:
    f.write("\n".join(html_parts))

print(f"Multi-station dashboard report written to {output_html}")
print(f"\n=== MULTI-STATION ANALYSIS COMPLETE ===")
if "STATIC_WATER_LEVEL_ORIGINAL" in wells.columns:
    corrected_wells = (wells["STATIC_WATER_LEVEL"] != wells["STATIC_WATER_LEVEL_ORIGINAL"]).sum()
    print(f"Wells with data quality corrections applied: {corrected_wells}")
    
    if corrected_wells > 0:
        print(f"Original vs. Corrected Static Water Level Comparison:")
        print(f"  Original mean: {wells['STATIC_WATER_LEVEL_ORIGINAL'].mean():.1f}m")
        print(f"  Corrected mean: {wells['STATIC_WATER_LEVEL'].mean():.1f}m")
        print(f"  Wells with original levels > 100m: {(wells['STATIC_WATER_LEVEL_ORIGINAL'] > 100).sum()}")
        print(f"  Wells with corrected levels > 100m: {(wells['STATIC_WATER_LEVEL'] > 100).sum()}")

print(f"\n=== REGIONAL DROUGHT ASSESSMENT SUMMARY ===")
print(f"Drought assessment method: Multi-station hydrometric analysis")
print(f"Stations monitored: {len(COLCHESTER_STATIONS)}")
print(f"Regional drought level: {drought_level}")

if station_data:
    print(f"Station results:")
    for station_id, result in station_data['stations'].items():
        status = "CRITICAL" if result['critical'] else "Normal"
        print(f"  {result['name']}: {result['flow']:.2f} mÂ³/s ({status})")
    print(f"Critical ratio: {station_data['critical_ratio']:.1%}")

drought_stats = wells["drought_drawdown_m"].describe()
print(f"\nDrought stress statistics:")
print(f"  Mean additional drawdown: {drought_stats['mean']:.1f}m")
print(f"  Max additional drawdown: {drought_stats['max']:.1f}m")
print(f"  Wells receiving enhanced stress: {(wells['drought_drawdown_m'] > DROUGHT_DRAWDOWN_M).sum()}")

print(f"\nAnalysis complete. Check {output_html} for the interactive multi-station dashboard.")