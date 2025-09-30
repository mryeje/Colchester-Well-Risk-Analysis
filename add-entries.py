import pandas as pd
import csv
from datetime import datetime

def add_manual_entry_to_csv(main_csv_path, output_csv_path=None):
    """
    Add the manual well entry to the main CSV database
    
    Args:
        main_csv_path (str): Path to the main CSV file
        output_csv_path (str): Path for the output CSV (defaults to overwriting main file)
    """
    
    # The mapped entry for the main CSV format
    new_entry = {
        'wellnumber': '',  # Will be auto-generated or left blank
        'drillerregnumberl': '',
        'drillersname': 'Glen Harpley',
        'drillercompany': '',
        'dugordrilled': 1,  # 1 for drilled well
        'datewellcompleted': '1977-09-29 00:00:00',
        'welldrilledforfirst': 'Jeremy',
        'welldrilledforlast': 'Smith',
        'contractor': '',
        'nearestcommunity': 'Colchester',
        'nearestcommunityatlasormapl': '',
        'civicaddress': '43 susan ct',
        'lotnumber': '',
        'subdivision': '',
        'propertpid': '',
        'countyl': 'Colchester',
        'postalcode': '',
        'finalstatusofwelll': '',
        'waterusel': '',
        'methodofdrillingl': '',
        'drillingfluidsl': '',
        'wqcolourl': 'none',
        'wqtastel': '',
        'wqodourl': 'none',
        'wqotherl': '',
        'dwbottommateriall': '',
        'dwstonereservoirmateriall': '',
        'dwbackfillmateriall': '',
        'driveshoemakel': '',
        'grouttypel': '',
        'packermakel': '',
        'wellfinishl': '',
        'screenmakel': '',
        'screenmateriall': '',
        'icastmspecl': '',
        'dwapronmateriall': '',
        'icmateriall': '',
        'ocmaterial': '',
        'ocastmspecl': '',
        'wymethodl': '',
        'wyrate': '',
        'wyduration': '',
        'wytestdepth': '',
        'wystaticlevel': '',
        'wydepthendoftest': '',
        'wytotaldrawdown': '',
        'wyrecoveredto': '',
        'wyrecoverybyhrs': '',
        'wyrecoverybymin': '',
        'wyoverflow': '',
        'wyvolumeremoved': '',
        'wydepthtoindexlevel': '',
        'wywaterlevelrecoveredto': '',
        'wyrecoveredtoindexlevel': '',
        'wyestimatedyield': 18.93,  # Converted from 71.66 m³/d to GPM
        'depthtobedrock': '',
        'totalorfinisheddepth': 92.0,  # From DEPTH_FT
        'fractures1': '',
        'fractures2': '',
        'fractures3': '',
        'fractures4': '',
        'fractures5': '',
        'fractures6': '',
        'ocfrom': '',
        'octo': '',
        'ocdiameter': '',
        'ocwallthickness': '',
        'icfrom': '',
        'icto': '',
        'icdiameter': '',
        'icwallthickness': '',
        'lengthcasingabovebyft': '',
        'lengthcasingabovebyin': '',
        'screenlength1': '',
        'screenfrom1': '',
        'screento1': '',
        'screenslotsize1': '',
        'screenlength2': '',
        'screenfrom2': '',
        'screento2': '',
        'screenslotsize2': '',
        'screenpacksize': '',
        'screenpackfrom': '',
        'screenpackto': '',
        'cdpropertyline': '',
        'cdbuilding': '',
        'cdroadway': '',
        'cdroadname': '',
        'cdonsiteseptic': '',
        'cdoffsiteseptic': '',
        'cdwatercourse': '',
        'cdwell': '',
        'cdoiltank': '',
        'cdcesspoolorothercontaminantsource': '',
        'dwdepthofliner': '',
        'dwstonereservoirvolume': '',
        'dwstonereservoirmaterialsize': '',
        'dwaprondepth': '',
        'dwapronthickness': '',
        'dwapronwidth': '',
        'dwapronvolume': '',
        'dwbackfillvolume': '',
        'chemdata': '',
        'comments': 'no',
        'easting': 481497.0,
        'northing': 5022656.0,
        'wlatlasormapl': '',
        'wlmappage': '',
        'wlreferenceletter': '',
        'wlreferencenumber': '',
        'wlroamerletter': '',
        'wlroamernumber': '',
        'wlmapsheet': '',
        'wlreferencemap': '',
        'wltract': '',
        'wlclaim': '',
        'welllocationsketch': '',
        'wellsketch': '',
        'wydepthtowaterbeforepump': '',
        'wydepthtowaterafterpump': '',
        'dwdepthtosealedliner': '',
        'comments2': '',
        'utmaccuracy': ''
    }
    
    try:
        # Read the existing CSV
        df = pd.read_csv(main_csv_path)
        
        # Convert the new entry to a DataFrame
        new_row = pd.DataFrame([new_entry])
        
        # Append the new row
        df_updated = pd.concat([df, new_row], ignore_index=True)
        
        # Save the updated DataFrame
        if output_csv_path is None:
            output_csv_path = main_csv_path
        
        df_updated.to_csv(output_csv_path, index=False)
        print(f"Successfully added manual entry to {output_csv_path}")
        print(f"New well added for: Jeremy Smith at 43 susan ct, Colchester")
        
    except Exception as e:
        print(f"Error: {e}")

def add_manual_entry_alternative(main_csv_path, output_csv_path=None):
    """
    Alternative method using csv module if pandas has issues
    """
    if output_csv_path is None:
        output_csv_path = main_csv_path
    
    # The new entry as a list of values (in the same order as the CSV headers)
    # This matches the order from your sample data
    new_entry = [
        '',  # wellnumber
        '',  # drillerregnumberl
        'Glen Harpley',  # drillersname
        '',  # drillercompany
        1,  # dugordrilled
        '1977-09-29 00:00:00',  # datewellcompleted
        'Jeremy',  # welldrilledforfirst
        'Smith',  # welldrilledforlast
        '',  # contractor
        'Colchester',  # nearestcommunity
        '',  # nearestcommunityatlasormapl
        '43 susan ct',  # civicaddress
        '',  # lotnumber
        '',  # subdivision
        '',  # propertpid
        'Colchester',  # countyl
        '',  # postalcode
        '',  # finalstatusofwelll
        '',  # waterusel
        '',  # methodofdrillingl
        '',  # drillingfluidsl
        'none',  # wqcolourl
        '',  # wqtastel
        'none',  # wqodourl
        '',  # wqotherl
        '',  # dwbottommateriall
        '',  # dwstonereservoirmateriall
        '',  # dwbackfillmateriall
        '',  # driveshoemakel
        '',  # grouttypel
        '',  # packermakel
        '',  # wellfinishl
        '',  # screenmakel
        '',  # screenmateriall
        '',  # icastmspecl
        '',  # dwapronmateriall
        '',  # icmateriall
        '',  # ocmaterial
        '',  # ocastmspecl
        '',  # wymethodl
        '',  # wyrate
        '',  # wyduration
        '',  # wytestdepth
        '',  # wystaticlevel
        '',  # wydepthendoftest
        '',  # wytotaldrawdown
        '',  # wyrecoveredto
        '',  # wyrecoverybyhrs
        '',  # wyrecoverybymin
        '',  # wyoverflow
        '',  # wyvolumeremoved
        '',  # wydepthtoindexlevel
        '',  # wywaterlevelrecoveredto
        '',  # wyrecoveredtoindexlevel
        18.93,  # wyestimatedyield
        '',  # depthtobedrock
        92.0,  # totalorfinisheddepth
        '',  # fractures1
        '',  # fractures2
        '',  # fractures3
        '',  # fractures4
        '',  # fractures5
        '',  # fractures6
        '',  # ocfrom
        '',  # octo
        '',  # ocdiameter
        '',  # ocwallthickness
        '',  # icfrom
        '',  # icto
        '',  # icdiameter
        '',  # icwallthickness
        '',  # lengthcasingabovebyft
        '',  # lengthcasingabovebyin
        '',  # screenlength1
        '',  # screenfrom1
        '',  # screento1
        '',  # screenslotsize1
        '',  # screenlength2
        '',  # screenfrom2
        '',  # screento2
        '',  # screenslotsize2
        '',  # screenpacksize
        '',  # screenpackfrom
        '',  # screenpackto
        '',  # cdpropertyline
        '',  # cdbuilding
        '',  # cdroadway
        '',  # cdroadname
        '',  # cdonsiteseptic
        '',  # cdoffsiteseptic
        '',  # cdwatercourse
        '',  # cdwell
        '',  # cdoiltank
        '',  # cdcesspoolorothercontaminantsource
        '',  # dwdepthofliner
        '',  # dwstonereservoirvolume
        '',  # dwstonereservoirmaterialsize
        '',  # dwaprondepth
        '',  # dwapronthickness
        '',  # dwapronwidth
        '',  # dwapronvolume
        '',  # dwbackfillvolume
        '',  # chemdata
        'no',  # comments
        481497.0,  # easting
        5022656.0,  # northing
        '',  # wlatlasormapl
        '',  # wlmappage
        '',  # wlreferenceletter
        '',  # wlreferencenumber
        '',  # wlroamerletter
        '',  # wlroamernumber
        '',  # wlmapsheet
        '',  # wlreferencemap
        '',  # wltract
        '',  # wlclaim
        '',  # welllocationsketch
        '',  # wellsketch
        '',  # wydepthtowaterbeforepump
        '',  # wydepthtowaterafterpump
        '',  # dwdepthtosealedliner
        '',  # comments2
        ''   # utmaccuracy
    ]
    
    try:
        # Read existing data
        with open(main_csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Add new row
        rows.append(new_entry)
        
        # Write back to file
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        print(f"Successfully added manual entry to {output_csv_path}")
        print(f"New well added for: Jeremy Smith at 43 susan ct, Colchester")
        
    except Exception as e:
        print(f"Error: {e}")

# Usage
if __name__ == "__main__":
    # Replace with your actual CSV file path
    main_csv_file = "well_logs_with_coords.csv"
    
    # Option 1: Use pandas method (recommended)
    add_manual_entry_to_csv(main_csv_file)
    
    # Option 2: If pandas doesn't work, use the alternative method
    # add_manual_entry_alternative(main_csv_file)
    
    # To create a backup and save to new file:
    # add_manual_entry_to_csv(main_csv_file, "wells_database_updated.csv")