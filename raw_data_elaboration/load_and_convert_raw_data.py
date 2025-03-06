import pyreadr
import pandas as pd
import tools.server_tools as server
import pickle
import argparse
import yaml

def server_data(data_path, meta_path, credentials_path):
    """Connects to the TreeNet server and downloads the entire metadata table and all the non-empty time series data.
        The metadata is a pandas dataframe and the timeseries data is stored as a list of pandas dataframes."""
    
    with open(credentials_path, 'r') as f:
        credentials = yaml.full_load(f)

    data = []
    meta = server.get_metadata(data.get('user'), data.get('password'), data.get('host'), data.get('port'), data.get('dbname'))
    for row in meta.iterrows():
        if row[1].tree_species == '-999':
            print('series id: ' + str(row[1].series_id) + ' -> empty')
        # else:
        series_id = row[1].series_id
        data.append(server.get_data(series_id))
        print('series id: ' + str(series_id) + ' -> OK')

    with open(data_path, 'wb') as f:
        pickle.dump(data, f)

    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str)
    parser.add_argument("meta_path", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("credentials_path", type=str)
    args = parser.parse_args()

    if args.source == "remote":
        server_data(args.meta_path, args.meta_path, args.credentials_path)

    # TODO: create a function that puts all the channels together into a single multi-channel time series.
