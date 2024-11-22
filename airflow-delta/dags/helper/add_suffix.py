import os
import pandas as pd
# Specify the main folder path
folder_path = "Data 2018-2023\Project"

# Walk through each folder and subfolder
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        # Check if the file has no extension
        if not os.path.splitext(filename)[1]:
            # Construct the old and new file paths
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, filename + ".json")
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed '{old_path}' to '{new_path}'")