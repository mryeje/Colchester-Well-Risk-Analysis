# add_county_data.py
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path

# --- CONFIGURATION ---
# Your main well data CSV file
INPUT_WELL_CSV = "well_logs_with_coords.csv" 

# The new CSV file that will be created with the corrected data
OUTPUT_WELL_CSV = "wells_with_county_added.csv"

# Common shapefile names for NS municipal boundaries
POSSIBLE_SHAPEFILES = [
    "MTA_MUNICIPAL_BOUNDARIES_UTM.shp",
    "municipal_boundaries.shp", 
    "ns_municipal_boundaries.shp",
    "colchester_boundary.shp"
]

# The name of the municipality to filter for
TARGET_COUNTY = "Colchester"

# Colchester County communities (based on your shapefile diagnostic output)
COLCHESTER_COMMUNITIES = {
    'Alton', 'Balmoral Mills', 'Bass River', 'Bayhead', 'Beaver Brook', 'Belmont', 
    'Bible Hill', 'Black Rock', 'Brentwood', 'Brookfield', 'Brule', 'Brule Point',
    'Brule Shore', 'Burnside', 'Camden', 'Carrs Brook', 'Castlereagh', 'Central New Annan',
    'Central North River', 'Central Onslow', 'Cloverdale', 'Coldstream', 'Debert', 
    'Debert Lake', 'Denmark', 'Earltown', 'East Earltown', 'East New Annan', 
    'East Village', 'Eastville', 'Economy', 'Five Houses', 'Five Islands', 
    'Folly Lake', 'French River', 'Gays River', 'Glenholme', 'Great Village',
    'Green Creek', 'Green Oaks', 'Harmony', 'Hilden', 'Highland Village',
    'Kemptown', 'Londonderry', 'Londonderry Station', 'Lower Debert', 'Lower Economy',
    'Lower Five Islands', 'Lower Onslow', 'Lynn', 'Masstown', 'McCallum Settlement',
    'Newton Mills', 'North River', 'North River Bridge', 'North River Centre',
    'Nuttby', 'Old Barns', 'Oliver', 'Onslow Mountain', 'Portapique', 
    'Salmon River', 'Salmon River Bridge', 'The Falls', 'Truro', 'Truro Heights',
    'Upper Brookside', 'Upper Economy', 'Upper Kemptown', 'Upper North River',
    'Upper Onslow', 'Upper Stewiacke', 'Valley', 'West New Annan', 'Wittenburg'
}

# Coordinate reference systems to try
POSSIBLE_CRS = ["EPSG:26920", "EPSG:4326", "EPSG:2961"]
# --- END CONFIGURATION ---

def find_shapefile():
    """Find the municipal boundaries shapefile"""
    # First try the expected names
    for shapefile in POSSIBLE_SHAPEFILES:
        if os.path.exists(shapefile):
            return shapefile
    
    # If not found, look for any .shp files in the directory
    current_dir = Path('.')
    shp_files = list(current_dir.glob('*.shp'))
    
    if shp_files:
        print(f"Found these .shp files in the current directory:")
        for i, shp_file in enumerate(shp_files):
            print(f"  {i+1}. {shp_file.name}")
        
        if len(shp_files) == 1:
            print(f"Using the only shapefile found: {shp_files[0].name}")
            return str(shp_files[0])
        else:
            # Let user choose or use the first one that might be municipal boundaries
            for shp_file in shp_files:
                name_lower = shp_file.name.lower()
                if any(keyword in name_lower for keyword in ['municipal', 'boundary', 'admin', 'county']):
                    print(f"Auto-selecting: {shp_file.name} (appears to be municipal boundaries)")
                    return str(shp_file)
            
            print(f"Multiple shapefiles found. Using the first one: {shp_files[0].name}")
            print(f"If this is wrong, rename your municipal boundaries file to 'municipal_boundaries.shp'")
            return str(shp_files[0])
    
    return None

def detect_coordinate_columns(df):
    """Detect which columns contain coordinate data"""
    possible_x_cols = ['easting', 'x', 'X', 'EASTING', 'UTM_E', 'longitude', 'lon', 'lng']
    possible_y_cols = ['northing', 'y', 'Y', 'NORTHING', 'UTM_N', 'latitude', 'lat']
    
    x_col = None
    y_col = None
    
    for col in possible_x_cols:
        if col in df.columns:
            x_col = col
            break
    
    for col in possible_y_cols:
        if col in df.columns:
            y_col = col
            break
    
    return x_col, y_col

def detect_municipality_column(gdf):
    """Detect which column contains municipality names"""
    possible_cols = ['MUNICIPALITY', 'NAME', 'MUNICIP', 'MUN_NAME', 'COUNTY', 'REGION', 'GSA_NAME', 'MUN_CODE', 'GSA_CODE']
    
    for col in possible_cols:
        if col in gdf.columns:
            # For GSA_NAME or similar, check if it contains readable municipality names
            if col in ['GSA_NAME', 'MUN_NAME', 'NAME']:
                sample_values = gdf[col].dropna().head(5).tolist()
                print(f"Sample values in {col}: {sample_values}")
                return col
            else:
                return col
    return None

print("=== Well Data County Correction Script ===")
print("This script will use spatial analysis to identify wells in Colchester County")
print("and populate the 'countyl' column accordingly.\n")

# 1. Load the well data from your CSV
try:
    print(f"Reading well data from '{INPUT_WELL_CSV}'...")
    wells_df = pd.read_csv(INPUT_WELL_CSV, low_memory=False)
    print(f"Loaded {len(wells_df)} total well records.")
    
    # Show column names to help with debugging
    print(f"Available columns: {list(wells_df.columns)}")
    
except FileNotFoundError:
    print(f"ERROR: Could not find the input file '{INPUT_WELL_CSV}'.")
    print("Please make sure it's in the same directory as the script.")
    exit()

# 2. Detect coordinate columns
x_col, y_col = detect_coordinate_columns(wells_df)
if not x_col or not y_col:
    print(f"ERROR: Could not find coordinate columns in the data.")
    print(f"Looking for X coordinates in: {['easting', 'x', 'X', 'EASTING', 'UTM_E', 'longitude', 'lon', 'lng']}")
    print(f"Looking for Y coordinates in: {['northing', 'y', 'Y', 'NORTHING', 'UTM_N', 'latitude', 'lat']}")
    print(f"Available columns: {list(wells_df.columns)}")
    exit()

print(f"Using coordinate columns: X = '{x_col}', Y = '{y_col}'")

# 3. Load the municipal boundaries shapefile
shapefile_path = find_shapefile()
if not shapefile_path:
    print(f"ERROR: Could not find municipal boundaries shapefile.")
    print(f"Looking for: {POSSIBLE_SHAPEFILES}")
    print("\nTo fix this:")
    print("1. Download NS municipal boundaries from https://geonova.novascotia.ca/")
    print("2. Extract all files (.shp, .shx, .dbf, .prj) to the same folder as this script")
    print("3. Rename the main .shp file to one of the expected names above")
    exit()

try:
    print(f"Reading municipal boundaries from '{shapefile_path}'...")
    
    # Check if all required shapefile components exist
    base_name = shapefile_path.replace('.shp', '')
    required_files = ['.shp', '.shx', '.dbf']
    missing_files = []
    
    for ext in required_files:
        file_path = base_name + ext
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"ERROR: Missing required shapefile components: {missing_files}")
        print("A complete shapefile needs these files:")
        print(f"  - {base_name}.shp (geometry)")
        print(f"  - {base_name}.shx (index)")  
        print(f"  - {base_name}.dbf (attributes)")
        print(f"  - {base_name}.prj (projection - optional)")
        print("\nSolution: Re-download the complete shapefile package and extract ALL files.")
        exit()
    
    # Try to restore the .shx file if it's corrupted
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    
    municipalities_gdf = gpd.read_file(shapefile_path)
    print(f"Loaded {len(municipalities_gdf)} municipal boundaries.")
    
    # Show available columns and municipalities
    print(f"Shapefile columns: {list(municipalities_gdf.columns)}")
    
    # Detect municipality column
    muni_col = detect_municipality_column(municipalities_gdf)
    if not muni_col:
        print("ERROR: Could not find municipality name column in shapefile.")
        print(f"Available columns: {list(municipalities_gdf.columns)}")
        exit()
    
    print(f"Using municipality column: '{muni_col}'")
    print(f"Available municipalities: {sorted(municipalities_gdf[muni_col].unique())}")
    
except Exception as e:
    print(f"ERROR: Could not load the shapefile '{shapefile_path}'.")
    print(f"Details: {e}")
    
    if "shx" in str(e).lower():
        print("\nThis is a shapefile index (.shx) corruption issue.")
        print("Solutions:")
        print("1. Re-download the complete municipal boundaries package")
        print("2. Make sure you extracted ALL files from the ZIP")
        print("3. Don't rename individual files - keep them together")
        print("4. If the download was incomplete, try downloading again")
    
    exit()

# 4. Find wells in Colchester County communities
try:
    print(f"Looking for Colchester County communities in shapefile...")
    
    # Filter to only Colchester communities
    colchester_boundaries = municipalities_gdf[
        municipalities_gdf[muni_col].isin(COLCHESTER_COMMUNITIES)
    ]
    
    if colchester_boundaries.empty:
        print(f"ERROR: Could not find any Colchester County communities in the shapefile.")
        print(f"Expected communities: {sorted(list(COLCHESTER_COMMUNITIES))[:10]}... (showing first 10)")
        
        # Check for partial matches
        found_communities = []
        for community in COLCHESTER_COMMUNITIES:
            matches = municipalities_gdf[
                municipalities_gdf[muni_col].str.contains(community, case=False, na=False)
            ]
            if not matches.empty:
                found_communities.extend(matches[muni_col].tolist())
        
        if found_communities:
            print(f"Found some matching communities: {found_communities[:10]}...")
            # Use partial matches
            colchester_boundaries = municipalities_gdf[
                municipalities_gdf[muni_col].isin(found_communities)
            ]
        else:
            print("No matching communities found at all.")
            exit()
    
    print(f"Found {len(colchester_boundaries)} Colchester County community boundaries:")
    found_communities_list = sorted(colchester_boundaries[muni_col].tolist())
    print(f"Communities: {found_communities_list}")
    
except Exception as e:
    print(f"ERROR: Problem finding Colchester communities: {e}")
    exit()

# 5. Prepare well data for spatial analysis
print("Preparing well data for spatial analysis...")

# Convert coordinates to numeric, dropping invalid entries
wells_df[x_col] = pd.to_numeric(wells_df[x_col], errors='coerce')
wells_df[y_col] = pd.to_numeric(wells_df[y_col], errors='coerce')

# Filter to wells with valid coordinates
wells_df_clean = wells_df.dropna(subset=[x_col, y_col]).copy()
invalid_coords = len(wells_df) - len(wells_df_clean)

if invalid_coords > 0:
    print(f"Warning: {invalid_coords} wells have invalid coordinates and will be skipped for spatial analysis.")

if wells_df_clean.empty:
    print("ERROR: No wells have valid coordinates for spatial analysis.")
    exit()

print(f"Processing {len(wells_df_clean)} wells with valid coordinates...")

# 6. Create GeoDataFrame from well data
# Try different coordinate reference systems
wells_gdf = None
for crs in POSSIBLE_CRS:
    try:
        print(f"Trying coordinate system: {crs}")
        wells_gdf = gpd.GeoDataFrame(
            wells_df_clean, 
            geometry=gpd.points_from_xy(wells_df_clean[x_col], wells_df_clean[y_col]),
            crs=crs
        )
        
        # Test if the coordinates make sense by checking bounds
        bounds = wells_gdf.bounds
        print(f"  Well bounds: X({bounds.minx.min():.1f} to {bounds.maxx.max():.1f}), Y({bounds.miny.min():.1f} to {bounds.maxy.max():.1f})")
        
        # For Nova Scotia, reasonable bounds would be:
        # UTM Zone 20N: X(200k-800k), Y(4.9M-5.2M)
        # WGS84: X(-67 to -60), Y(44 to 47)
        if crs == "EPSG:26920":  # UTM
            if (200000 <= bounds.minx.min() <= 800000 and 
                4900000 <= bounds.miny.min() <= 5200000):
                print(f"  Coordinates look reasonable for {crs}")
                break
        elif crs == "EPSG:4326":  # WGS84
            if (-67 <= bounds.minx.min() <= -60 and 
                44 <= bounds.miny.min() <= 47):
                print(f"  Coordinates look reasonable for {crs}")
                break
        else:
            break  # Accept other CRS without validation
            
    except Exception as e:
        print(f"  Failed with {crs}: {e}")
        continue

if wells_gdf is None:
    print("ERROR: Could not create valid GeoDataFrame from well coordinates.")
    exit()

# 7. Ensure both datasets use the same coordinate system
print("Aligning coordinate systems...")
if wells_gdf.crs != colchester_boundaries.crs:
    print(f"Reprojecting data to common CRS: {wells_gdf.crs}")
    colchester_boundaries = colchester_boundaries.to_crs(wells_gdf.crs)

# 8. Perform spatial join
print("Performing spatial analysis... this may take a moment.")
try:
    wells_in_county = gpd.sjoin(wells_gdf, colchester_boundaries, how="inner", predicate="within")
    print(f"Found {len(wells_in_county)} wells located within Colchester County communities.")
    
    if len(wells_in_county) == 0:
        print("WARNING: No wells found within any Colchester County community boundaries.")
        print("This could mean:")
        print("1. Wrong coordinate system")
        print("2. Wells are actually outside Colchester County")
        print("3. Coordinate data is in wrong format")
        
        # Show some sample coordinates for debugging
        print("\nSample well coordinates:")
        for i, (_, row) in enumerate(wells_df_clean.head(5).iterrows()):
            print(f"  Well {i+1}: X={row[x_col]}, Y={row[y_col]}")
        
        # Show the community boundary extents for comparison
        bounds = colchester_boundaries.total_bounds
        print(f"\nColchester community boundaries extent:")
        print(f"  X: {bounds[0]:.1f} to {bounds[2]:.1f}")
        print(f"  Y: {bounds[1]:.1f} to {bounds[3]:.1f}")
        
    else:
        # Show which communities the wells were found in
        communities_with_wells = wells_in_county[muni_col].value_counts()
        print(f"\nWells found in these communities:")
        for community, count in communities_with_wells.head(10).items():
            print(f"  {community}: {count} wells")
        if len(communities_with_wells) > 10:
            print(f"  ... and {len(communities_with_wells) - 10} more communities")
        
except Exception as e:
    print(f"ERROR during spatial join: {e}")
    exit()

# 9. Update the original dataset
print("Updating county information...")

# Initialize or clean the countyl column
if 'countyl' not in wells_df.columns:
    wells_df['countyl'] = ''
else:
    wells_df['countyl'] = wells_df['countyl'].fillna('')

# Mark wells that are in Colchester County
if 'wellnumber' in wells_in_county.columns:
    colchester_well_ids = wells_in_county['wellnumber'].values
    wells_df.loc[wells_df['wellnumber'].isin(colchester_well_ids), 'countyl'] = TARGET_COUNTY
    print(f"Updated 'countyl' column using 'wellnumber' for matching.")
else:
    # Use index-based matching as fallback
    colchester_indices = wells_in_county.index.values
    wells_df.loc[colchester_indices, 'countyl'] = TARGET_COUNTY
    print(f"Updated 'countyl' column using row indices for matching.")

# 10. Save results
wells_df.to_csv(OUTPUT_WELL_CSV, index=False)

# 11. Summary
wells_marked = (wells_df['countyl'] == TARGET_COUNTY).sum()
print(f"\n=== SUMMARY ===")
print(f"✓ Successfully processed {len(wells_df)} total wells")
print(f"✓ {wells_marked} wells marked as being in {TARGET_COUNTY} County")
print(f"✓ Results saved to: '{OUTPUT_WELL_CSV}'")
print(f"\nNext steps:")
print(f"1. Use '{OUTPUT_WELL_CSV}' as input for your main analysis script")
print(f"2. Update the well_files list in water_table.py to prioritize this file:")
print(f'   well_files = ["{OUTPUT_WELL_CSV}", "well_logs_with_coords.csv", ...]')
print(f"3. Run your main analysis script")

if wells_marked == 0:
    print(f"\n⚠️  WARNING: No wells were marked as being in {TARGET_COUNTY} County.")
    print(f"This suggests a coordinate system or boundary data issue.")
    print(f"Check the sample coordinates shown above and verify they make sense.")