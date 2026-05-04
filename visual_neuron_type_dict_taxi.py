import os
import glob
import pandas as pd
import pickle


# Define file paths
input_file = "nyc-taxi-trip"
output_file = "locationid_borough_dict.pkl"


def create_locationid_type_dict(input_file, output_file):
    print("Loading CSV file(s)...")

    # Read one CSV or all CSVs in a folder
    if os.path.isdir(input_file):
        csv_files = sorted(glob.glob(os.path.join(input_file, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in folder: {input_file}")

        print(f"Found {len(csv_files)} CSV files in folder: {input_file}")
        df_list = []
        for file in csv_files:
            print(f"Loading {file} ...")
            df_list.append(pd.read_csv(file))
        df = pd.concat(df_list, ignore_index=True)
    else:
        print(f"Loading {input_file} ...")
        df = pd.read_csv(input_file)

    print("Creating dictionary...")

    # Build from both pickup and dropoff sides
    pu_df = df[['pulocationid', 'pu_borough']].dropna().copy()
    pu_df.columns = ['locationid', 'type']

    do_df = df[['dolocationid', 'do_borough']].dropna().copy()
    do_df.columns = ['locationid', 'type']

    merged_df = pd.concat([pu_df, do_df], ignore_index=True)

    # Ensure consistent types
    merged_df['locationid'] = merged_df['locationid'].astype(int)

    # Drop duplicates; keep the first if repeated
    merged_df = merged_df.drop_duplicates(subset=['locationid'])

    locationid_type_dict = dict(zip(merged_df['locationid'], merged_df['type']))

    print("Saving dictionary as pickle file...")
    with open(output_file, 'wb') as f:
        pickle.dump(locationid_type_dict, f)

    print(f"Dictionary saved to {output_file}")
    print(f"Number of unique location IDs: {len(locationid_type_dict)}")


if __name__ == "__main__":
    create_locationid_type_dict(input_file, output_file)
