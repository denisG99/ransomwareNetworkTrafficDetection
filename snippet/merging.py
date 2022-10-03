import pandas as pd
import os

DATA_PATH = '../dataset/data/'
CONNECTION_PATH = '../dataset/connection/'

def CSV_merging(dest_df, files_path):
    for csv in os.listdir(files_path):
        df = pd.read_csv(f'{files_path + csv}')

        dest_df = pd.concat([dest_df, df])

    return dest_df

CSV_merging(pd.DataFrame(), DATA_PATH).to_csv("../dataset/datas.csv", index=False)
CSV_merging(pd.DataFrame(), CONNECTION_PATH).to_csv("../dataset/connections.csv", index=False)