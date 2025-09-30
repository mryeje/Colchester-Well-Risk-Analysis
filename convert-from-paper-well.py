# well_report_parser.py - Convert physical well reports to digital format
import pandas as pd
import re
from datetime import datetime
import json
import os
import argparse
import sys

def convert_feet_to_meters(df):
    """
    Convert all depth-related columns from feet to meters.
    Looks for common column patterns and converts them.
    """
    print("\n=== CONVERTING FEET TO METERS ===")
    
    # Mapping of common column names that should be in feet
    # Format: {pattern: (conversion_factor, new_suffix)}
    conversion_map = {
        'DEPTH': 0.3048,
        'STATIC_WATER_LEVEL': 0.3048,
        'WYSTATICLEVEL': 0.3048,
        'WYDEPTHTOWATERBEFOREPUMP': 0.3048,
        'WYDEPTHTOWATERAFTERPUMP': 0.3048,
        'TOTALORFINISHEDDEPTH': 0.3048,
        'CASING_FROM': 0.3048,
        'CASING_TO': 0.3048,
        'ROSE_TO': 0.3048,
        'DRAWDOWN': 0.3048,
        'RECOVERED_TO': 0.3048,
        'FINAL_WATER_LEVEL': 0.3048,
        'DEPTH_TO_BEDROCK': 0.3048,
        'PUMP_DEPTH': 0.3048
    }
    
    converted_cols = []
    
    for col in df.columns:
        col_upper = col.upper().strip()
        
        # Check if this column should be converted
        for pattern, factor in conversion_map.items():
            if pattern in col_upper:
                # Check if values look like they're in feet (typically > 10 for Nova Scotia wells)
                sample_values = pd.to_numeric(df[col], errors='coerce').dropna()
                
                if len(sample_values) > 0:
                    avg_value = sample_values.mean()
                    
                    # If average value suggests feet (most NS wells are 20-200 feet deep)
                    if avg_value > 10:
                        # Keep original in a _FT column
                        df[f'{col}_FT'] = df[col].copy()
                        
                        # Convert to meters
                        df[col] = pd.to_numeric(df[col], errors='coerce') * factor
                        df[col] = df[col].round(2)
                        
                        converted_cols.append(col)
                        print(f"  Converted {col}: avg {avg_value:.1f} ft → {avg_value * factor:.1f} m")
                    else:
                        print(f"  Skipped {col}: values already appear to be in meters (avg: {avg_value:.1f})")
                
                break
    
    if converted_cols:
        print(f"\nConverted {len(converted_cols)} columns from feet to meters")
    else:
        print("\nNo columns needed conversion (may already be in meters)")
    
    return df, converted_cols

def convert_gpm_to_lpm(df):
    """Convert yield/flow rates from GPM to L/min"""
    yield_cols = [col for col in df.columns if any(term in col.upper() for term in ['YIELD', 'RATE', 'FLOW', 'GPM'])]
    
    converted = []
    for col in yield_cols:
        sample_values = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(sample_values) > 0:
            avg_value = sample_values.mean()
            
            # If values look like GPM (typically 5-50 for residential wells)
            if avg_value < 100:  # Unlikely to have >100 GPM residential wells
                df[f'{col}_GPM'] = df[col].copy()
                df[col] = pd.to_numeric(df[col], errors='coerce') * 3.78541
                df[col] = df[col].round(2)
                converted.append(col)
                print(f"  Converted {col}: avg {avg_value:.1f} GPM → {avg_value * 3.78541:.1f} L/min")
    
    return df, converted

def process_csv_file(input_file, output_file=None, save_intermediate=False):
    """
    Read a CSV file with measurements in feet/imperial units and convert to metric.
    """
    print(f"\n=== PROCESSING CSV FILE: {input_file} ===")
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return None
    
    # Read the CSV
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} wells from {input_file}")
        print(f"Columns found: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    # Normalize column names
    df.columns = [str(c).upper().strip() for c in df.columns]
    
    # Ensure COUNTY column exists and set to Colchester if missing
    if 'COUNTY' not in df.columns:
        df['COUNTY'] = 'Colchester'
        print("Added COUNTY column (set to Colchester)")
    
    # Fill empty counties with Colchester
    df['COUNTY'] = df['COUNTY'].fillna('Colchester')
    
    # Convert measurements
    df, depth_cols = convert_feet_to_meters(df)
    df, yield_cols = convert_gpm_to_lpm(df)
    
    # Only save intermediate file if requested
    if save_intermediate:
        if not output_file:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_metric.csv"
        
        df.to_csv(output_file, index=False)
        print(f"\nConverted data saved to intermediate file: {output_file}")
        return df, output_file
    
    return df, None

def append_to_well_files(df):
    """
    Append the converted data to all standard well database files.
    """
    target_files = [
        'wells_with_county_added.csv',
        'well_logs_with_coords.csv', 
        'well_logs.csv',
        'wells.csv'
    ]
    
    print("\n=== UPDATING WELL DATABASE FILES ===")
    
    updated_files = []
    
    for filename in target_files:
        if os.path.exists(filename):
            try:
                # Load existing data
                existing_df = pd.read_csv(filename)
                original_count = len(existing_df)
                
                # Normalize column names
                existing_df.columns = [str(c).upper().strip() for c in existing_df.columns]
                
                # Add missing columns from new data with None values
                for col in df.columns:
                    if col not in existing_df.columns:
                        existing_df[col] = None
                
                # Add missing columns from existing data to new data
                df_to_append = df.copy()
                for col in existing_df.columns:
                    if col not in df_to_append.columns:
                        df_to_append[col] = None
                
                # Reorder columns to match existing file
                df_to_append = df_to_append[existing_df.columns]
                
                # Combine
                combined_df = pd.concat([existing_df, df_to_append], ignore_index=True)
                
                # Remove duplicates if WELL_ID exists
                if 'WELL_ID' in combined_df.columns:
                    before_dedup = len(combined_df)
                    combined_df = combined_df.drop_duplicates(subset=['WELL_ID'], keep='last')
                    after_dedup = len(combined_df)
                    if before_dedup > after_dedup:
                        print(f"  {filename}: Removed {before_dedup - after_dedup} duplicate wells")
                
                # Save
                combined_df.to_csv(filename, index=False)
                new_count = len(combined_df)
                added = new_count - original_count
                
                print(f"  {filename}: {original_count} → {new_count} wells (+{added})")
                updated_files.append(filename)
                
            except Exception as e:
                print(f"  Error updating {filename}: {e}")
        else:
            # Create new file with the data
            try:
                df.to_csv(filename, index=False)
                print(f"  {filename}: Created new file with {len(df)} wells")
                updated_files.append(filename)
            except Exception as e:
                print(f"  Error creating {filename}: {e}")
    
    if updated_files:
        print(f"\nSuccessfully updated {len(updated_files)} files")
    else:
        print("\nWarning: No files were updated")
    
    return updated_files

def parse_well_report_interactive():
    """
    Interactive script to manually enter well report data from physical documents.
    """
    print("=== WELL DRILLING REPORT DATA ENTRY ===")
    print("Enter measurements in FEET as shown on the report - they'll be converted to meters.\n")
    
    well_data = {}
    
    # Owner Information
    print("--- OWNER INFORMATION ---")
    well_data['OWNER_NAME'] = input("Owner name: ").strip()
    well_data['CIVIC_ADDRESS'] = input("Address: ").strip()
    well_data['MUNICIPALITY'] = input("Municipality/Community: ").strip()
    well_data['COUNTY'] = input("County (default: Colchester): ").strip() or "Colchester"
    
    # Location
    print("\n--- LOCATION ---")
    well_data['LOCATION_DESCRIPTION'] = input("Location of well (description): ").strip()
    
    # Coordinates (if available)
    print("\nCoordinates (leave blank if not available):")
    x_coord = input("  Easting/X coordinate: ").strip()
    y_coord = input("  Northing/Y coordinate: ").strip()
    well_data['X'] = float(x_coord) if x_coord else None
    well_data['Y'] = float(y_coord) if y_coord else None
    
    # Driller Information
    print("\n--- DRILLER INFORMATION ---")
    well_data['DRILLER'] = input("Driller name: ").strip()
    well_data['LICENSE_NO'] = input("License number: ").strip()
    
    # Date
    print("\n--- DATE ---")
    day = input("Day completed: ").strip()
    month = input("Month completed: ").strip()
    year = input("Year completed (2-digit or 4-digit): ").strip()
    
    if year and len(year) == 2:
        year_int = int(year)
        year = f"19{year}" if year_int > 50 else f"20{year}"
    
    if day and month and year:
        well_data['DATE_COMPLETED'] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    else:
        well_data['DATE_COMPLETED'] = None
    
    # Well Construction Details
    print("\n--- WELL CONSTRUCTION ---")
    well_data['WELL_TYPE'] = input("Type of well (dug/bored/drilled/rotary): ").strip()
    
    depth_ft = input("Total depth below surface (ft): ").strip()
    if depth_ft:
        depth_m = float(depth_ft) * 0.3048
        well_data['DEPTH'] = round(depth_m, 2)
        well_data['DEPTH_FT'] = float(depth_ft)
    else:
        well_data['DEPTH'] = None
    
    diameter = input("Diameter (inches): ").strip()
    well_data['DIAMETER'] = float(diameter) if diameter else None
    
    # Water Information
    print("\n--- WATER INFORMATION ---")
    static_from = input("Water at (ft) - starting depth: ").strip()
    
    if static_from:
        well_data['STATIC_WATER_LEVEL'] = round(float(static_from) * 0.3048, 2)
        well_data['STATIC_WATER_LEVEL_FT'] = float(static_from)
    else:
        well_data['STATIC_WATER_LEVEL'] = None
    
    rose_to = input("Rose to (ft below surface): ").strip()
    if rose_to:
        well_data['ROSE_TO'] = round(float(rose_to) * 0.3048, 2)
    
    # Pump test information
    print("\n--- PUMP TEST ---")
    pumped_rate = input("Pumped at (gpm): ").strip()
    
    if pumped_rate:
        yield_lpm = float(pumped_rate) * 3.78541
        well_data['YIELD'] = round(yield_lpm, 2)
    else:
        well_data['YIELD'] = None
    
    # Geology
    print("\n--- GEOLOGY (optional, press Enter to skip) ---")
    depth_to_bedrock = input("Depth to top of bedrock (ft): ").strip()
    if depth_to_bedrock:
        well_data['DEPTH_TO_BEDROCK'] = round(float(depth_to_bedrock) * 0.3048, 2)
    
    # Additional notes
    print("\n--- ADDITIONAL INFORMATION ---")
    well_data['NOTES'] = input("Any additional notes: ").strip()
    
    return well_data

def save_well_data(well_data):
    """Save manually entered well data"""
    if 'WELL_ID' not in well_data or not well_data['WELL_ID']:
        owner_part = re.sub(r'[^a-zA-Z0-9]', '', well_data.get('OWNER_NAME', 'UNKNOWN'))[:10]
        date_part = well_data.get('DATE_COMPLETED', 'NODATE').replace('-', '')
        well_data['WELL_ID'] = f"{owner_part}_{date_part}"
    
    df = pd.DataFrame([well_data])
    updated_files = append_to_well_files(df)
    
    return updated_files

def display_summary(well_data):
    """Display a summary of entered data"""
    print("\n" + "="*60)
    print("WELL DATA ENTRY SUMMARY")
    print("="*60)
    print(f"Owner: {well_data.get('OWNER_NAME')}")
    print(f"Location: {well_data.get('CIVIC_ADDRESS')}, {well_data.get('MUNICIPALITY')}")
    print(f"County: {well_data.get('COUNTY')}")
    
    if well_data.get('DEPTH'):
        print(f"Well Depth: {well_data['DEPTH']} m ({well_data.get('DEPTH_FT', 'N/A')} ft)")
    if well_data.get('STATIC_WATER_LEVEL'):
        print(f"Static Water Level: {well_data['STATIC_WATER_LEVEL']} m ({well_data.get('STATIC_WATER_LEVEL_FT', 'N/A')} ft)")
    if well_data.get('YIELD'):
        print(f"Yield: {well_data['YIELD']} L/min")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Convert well drilling reports from imperial to metric units and update well databases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - manually enter well data
  python well_report_parser.py
  
  # Batch mode - convert CSV file from feet to meters
  python well_report_parser.py --csv my_wells_in_feet.csv
  
  # Convert and specify output file
  python well_report_parser.py --csv wells_ft.csv --output wells_metric.csv
        """
    )
    
    parser.add_argument('--csv', type=str, help='CSV file to convert from imperial to metric units')
    parser.add_argument('--output', type=str, help='Output file for converted CSV (optional, saves intermediate file)')
    parser.add_argument('--no-update', action='store_true', help='Do not update standard well database files')
    parser.add_argument('--save-intermediate', action='store_true', help='Save converted CSV as intermediate file before appending')
    
    args = parser.parse_args()
    
    if args.csv:
        # Batch CSV conversion mode
        result = process_csv_file(args.csv, args.output, save_intermediate=args.save_intermediate)
        
        if result is not None:
            df, output_file = result
            
            if not args.no_update:
                append_to_well_files(df)
                print("\n=== CONVERSION COMPLETE ===")
                print(f"Converted {len(df)} wells and appended to database files")
                print("Run water_table-test.py to analyze updated well data")
            else:
                if output_file:
                    print(f"\n=== CONVERSION COMPLETE ===")
                    print(f"Converted data saved to: {output_file}")
                    print("(Database files not updated due to --no-update flag)")
        else:
            print("\nError: Conversion failed")
            sys.exit(1)
    else:
        # Interactive manual entry mode
        print("Starting interactive well data entry mode")
        print("(Use --csv flag for batch CSV conversion)\n")
        
        while True:
            well_data = parse_well_report_interactive()
            display_summary(well_data)
            
            confirm = input("\nIs this information correct? (y/n): ").strip().lower()
            if confirm == 'y':
                updated_files = save_well_data(well_data)
                print(f"\nSuccess! Updated {len(updated_files)} well database files")
                
                another = input("\nEnter another well report? (y/n): ").strip().lower()
                if another != 'y':
                    break
            else:
                retry = input("Re-enter this well's data? (y/n): ").strip().lower()
                if retry != 'y':
                    break
        
        print("\n=== DATA ENTRY COMPLETE ===")
        print("Run water_table-test.py to analyze updated well data")

if __name__ == "__main__":
    main()