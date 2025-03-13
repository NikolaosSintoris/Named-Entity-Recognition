import os
import pandas as pd



if __name__ == "__main__":

    # The data path.
    data_path = '/home/nsintoris/Documents/Projects/Orfium/data' 
    input_csv_name = 'normalization_assesment_dataset_10k.csv'

    # Read the output csv.
    df = pd.read_csv(os.path.join(data_path, input_csv_name))

    print(df.info())

    # Checking missing data patterns.
    missing_clean_text = df['CLEAN_TEXT'].isna().sum()
    missing_raw_text = df['raw_comp_writers_text'].isna().sum()

    print(f'Missing raw_comp_writers_text Entries: {missing_raw_text}')
    print(f'Missing CLEAN_TEXT Entries: {missing_clean_text}')
    print(f"'/' seperator count': {df['raw_comp_writers_text'].str.contains('/', na=False).sum()}")
    print(f"',' seperator count': {df['raw_comp_writers_text'].str.contains(',', na=False).sum()}")
    print(f"'&' seperator count': {df['raw_comp_writers_text'].str.contains('&', na=False).sum()}")