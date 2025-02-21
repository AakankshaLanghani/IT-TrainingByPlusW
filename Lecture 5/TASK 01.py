import os
import glob
import shutil
import pandas as pd

# Ensure backup folder exists
if not os.path.exists("backup_folder"):
    os.makedirs("backup_folder")

# Move all CSV files from "csv_files" to "backup_folder"
csv_files = glob.glob("csv_files/*.csv")
for file in csv_files:
    shutil.move(file, "backup_folder/")
    print(f"Moved file: {file}")

# Function to export DataFrame to CSV or JSON
def export_data(df, filename, file_format):
    if file_format == "csv":
        df.to_csv(filename, index=False)
        print(f"Data exported to {filename} in CSV format.")
    elif file_format == "json":
        df.to_json(filename, orient="records")
        print(f"Data exported to {filename} in JSON format.")
    else:
        print("Unsupported format.")

# Example DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)

# Exporting DataFrame
export_data(df, "output.csv", "csv")
export_data(df, "output.json", "json")
