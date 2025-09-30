# specific_entry_add.py
import os

csv_file = "wells_with_county_added.csv"

# Your specific manual entry formatted to match the CSV structure
well_entry = '999999,,Glen Harpley,,1,1977-09-29 00:00:00,Jeremy,Smith,,Colchester,,43 susan ct,,,,Colchester,,,,1,,,,none,,none,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,71.6,,28.0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,Added manually by Jeremy,481497.0,5022656.0,,,,,,,,,,,,,,,,'

if os.path.exists(csv_file):
    # Count current lines
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines_before = len(f.readlines())
    
    # Add the new entry
    with open(csv_file, 'a', encoding='utf-8') as f:
        f.write('\n' + well_entry)
    
    # Count lines after
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines_after = len(f.readlines())
    
    print("✅ Well 999999 added successfully!")
    print(f"File now has {lines_after} lines (was {lines_before})")
    print("\nEntry details:")
    print("Well: 999999")
    print("Driller: Glen Harpley") 
    print("Owner: Jeremy Smith")
    print("Address: 43 susan ct, Colchester")
    print("County: Colchester")
    print("Depth: 28.0 ft")
    print("Yield: 71.6 GPM")
    print("Coordinates: 481497.0, 5022656.0")
    print("Comments: Added manually by Jeremy")
else:
    print(f"Error: {csv_file} not found!")