import pyreadr
import pandas as pd
import tools.server_tools as server
import pickle


def from_file(file_path, output_path, species='all'):
    # Note: loads the raw data used in the DeepT project and transforms it into the TNT standard from
    df = pyreadr.read_r(file_path)[None]
    if species != 'all':
        df = df.loc[df.species == species]
        print(species)
    df.loc[:, 'subplot'] = df.loc[:, 'subplot'].fillna('unknown')
    # Note: replace NaN by 'unknown', otherwise it will be treated as an empty data point and not an unknown site

    df_temp = []
    counter = 0
    for e in df.groupby(['station', 'subplot', 'tree']):
        e[1]['series_name'] = counter
        # Note: the first element of 'e', i.e. e[0] is a tuple of the station, subplot and tree names.
        df_temp.append(e[1])
        counter += 1

    df = pd.concat(df_temp, ignore_index=True, sort=False)
    # df['series_name'] = df.groupby(['station', 'subplot', 'tree']).grouper.group_info[0]

    data = df[['ts', 'series_name', 'GRO', 'temp', 'rh', 'vpd', 'rad', 'swp', 'total_precip', 'doy']]
    data['year'] = pd.to_datetime(data['ts']).dt.year
    data['month'] = pd.to_datetime(data['ts']).dt.month
    data['day'] = pd.to_datetime(data['ts']).dt.day
    data['hour'] = pd.to_datetime(data['ts']).dt.hour
    data = data.rename(columns={'GRO': 'LM'})
    data.to_pickle(output_path+'data_DeepT_'+species+'.pkl')

    meta = df[['series_name', 'station', 'subplot', 'tree', 'species', 'station.plot']]
    meta = meta.drop_duplicates()
    meta.to_pickle(output_path+'meta_DeepT_'+species+'.pkl')
    print(meta)
    print(meta.species.drop_duplicates())


def server_data(data_path, meta_path):
    """Connects to the TreeNet server and downloads the entire metadata table and all the non-empty time series data.
        The metadata is a pandas dataframe and the timeseries data is stored as a list of pandas dataframes."""
    data = []
    meta = server.get_metadata()
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
