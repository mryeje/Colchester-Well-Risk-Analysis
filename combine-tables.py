import pandas as pd
import os

# Mapping of well IDs to their filenames
well_files = {
    "004": "004FraserBrookWaterLevelData.xls",
    "007": "007MurraySidingWaterLevelData.xls",
    "014": "014TruroWaterLevelData.xls",
    "068": "068DebertWaterLevelData.xlsx",
    "083": "083TatamagoucheWaterLevelData.xlsx"
}

# First, let's check what sheets are available in each file
print("🔍 Checking available sheets in each file:")
for well_id, filename in well_files.items():
    if os.path.exists(filename):
        try:
            excel_file = pd.ExcelFile(filename)
            print(f"{filename}: {excel_file.sheet_names}")
        except Exception as e:
            print(f"⚠️ Could not read {filename}: {e}")
    else:
        print(f"❌ File not found: {filename}")

print("\n" + "="*50 + "\n")

all_data = []

for well_id, filename in well_files.items():
    try:
        # Check if file exists
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            continue
            
        # Try different possible sheet names
        sheet_names_to_try = [
            "Water Levels - year by year",
            "Water Levels-year by year", 
            "Water Levels year by year",
            "Water Levels"
        ]
        
        df = None
        used_sheet = None
        
        for sheet_name in sheet_names_to_try:
            try:
                df = pd.read_excel(filename, sheet_name=sheet_name, skiprows=6)
                used_sheet = sheet_name
                print(f"✅ Found sheet: '{sheet_name}' in {filename}")
                break
            except:
                continue
        
        if df is None:
            # If none of the expected names work, try the first sheet
            try:
                df = pd.read_excel(filename, sheet_name=0, skiprows=6)
                used_sheet = "First sheet (index 0)"
                print(f"⚠️ Using first sheet for {filename}")
            except Exception as e:
                print(f"❌ Could not read any sheet from {filename}: {e}")
                continue
        
        # Rename first column as 'date'
        df = df.rename(columns={df.columns[0]: "date"})
        
        # Drop empty rows
        df = df.dropna(subset=["date"])
        
        # Melt wide (years as columns) → long format
        df_long = df.melt(id_vars=["date"], var_name="year", value_name="water_level_m")
        
        # Convert to datetime with specific format to avoid warnings
        df_long["date"] = pd.to_datetime(df_long["date"], format='%d-%b', errors='coerce')
        df_long["water_level_m"] = pd.to_numeric(df_long["water_level_m"], errors="coerce")
        
        # Drop invalid rows
        df_long = df_long.dropna(subset=["date", "water_level_m"])
        
        # Add WELL_ID
        df_long["WELL_ID"] = well_id
        
        # Keep only needed columns
        df_long = df_long[["WELL_ID", "date", "water_level_m"]]
        
        all_data.append(df_long)
        print(f"✅ Processed {filename} (Well {well_id}) with {len(df_long)} rows using sheet: {used_sheet}")
    
    except Exception as e:
        print(f"⚠️ Error processing {filename}: {e}")

# Merge everything
if all_data:
    merged = pd.concat(all_data, ignore_index=True)
    
    # Add year column for easier analysis
    merged["year"] = merged["date"].dt.year
    merged["month_day"] = merged["date"].dt.strftime("%m-%d")
    
    merged.to_csv("obs_well_timeseries.csv", index=False)
    print(f"\n💾 Saved merged dataset: obs_well_timeseries.csv ({len(merged)} rows total)")
    print(f"📊 Data covers years: {merged['year'].min()} to {merged['year'].max()}")
    print(f"🏭 Wells included: {merged['WELL_ID'].unique().tolist()}")
else:
    print("❌ No data processed — check file formats/paths")