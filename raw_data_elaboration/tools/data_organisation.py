import pandas as pd
import pickle


def load_data(meta_path, df_path):
    metatemp = pd.read_pickle(meta_path)
    datatemp = pd.read_pickle(df_path)

    metadata = dict()
    dataframe = dict()

    for e in datatemp:
        if len(e) > 0:
            key = e.series_id.iloc[0]
            dataframe[key] = e
            metadata[key] = metatemp[metatemp.series_id == key]

        return metadata, dataframe


def get_species(metadata):
    species = dict()
    genus_list = metadata.tree_genus.drop_duplicates().to_list()
    for e in genus_list:
        temp_df = metadata[metadata.tree_genus == e]
        species[e] = [temp_df.tree_species.drop_duplicates().to_list(), len(temp_df)]
        # Todo: Include the number of species trees

    return species


def get_sites(metadata):
    sites = dict()
    site_list = metadata.site_name.drop_duplicates().to_list()
    for e in site_list:
        temp_df = metadata[metadata.site_name == e]
        species = temp_df.tree_species.drop_duplicates().to_list()
        species_count = []
        for ee in species:
            species_count.append(temp_df[temp_df.tree_species == ee].shape[0])
        siteXcor = pd.to_numeric(temp_df.iloc[0].site_xcor)
        siteYcor = pd.to_numeric(temp_df.iloc[0].site_ycor)
        sites[e] = [species, temp_df.shape[0], species_count, [siteXcor, siteYcor]]

    return sites


def get_yearly_data_by_id(dictionary):
    # note: take care of leap years, they are also included
    data_yr = dict()
    for key, value in dictionary.items():
        df_list = _ts_by_year(value)
        temp_list = []
        for e in df_list[:]:
            if e.doy.iloc[0] == 1 or e.doy.iloc[-1] > 364:
                temp_list.append(e)
        data_yr[key] = temp_list
    return data_yr


def get_yearly_data_by_year(dictionary):
    # note: take care of leap years, they are also included
    data_by_year = dict()
    for key, value in dictionary.items():
        for e in value:
            year = e.year.iloc[0]
            if year in data_by_year:
                data_by_year[year].append(e)
            else:
                data_by_year[year] = [e]
    return data_by_year


def get_monthly_data(metadata, data):
    output = []
    return output


def save_data(metadata, data, period):
    path = '/Users/lukovic/data/FORWARDS/TNT/raw_data/modified_data/'
    with open(path + 'data_' + period + '.pkl', 'wb') as f:
        pickle.dump(data, f)

    with open(path + 'metadata_' + period + '.pkl', 'wb') as f:
        pickle.dump(metadata, f)


def _ts_by_year(data):
    data['year'] = data.ts.dt.year  # note: add a column with the year
    data['doy'] = data.ts.dayofyear  # note: add a column with the day of year
    years = data.year.unique()
    output = []
    for year in years:
        output.append(data[data.year == year])

    return output


def average_over_day(dataframe):
    return


def average_over_hour(dataframe):
    return


def get_hourly_data(dictionary):
    new_dict = dict()
    for key, value in dictionary.items():
        new_dict[key] = []
        for e in value:
            new_dict[key].append(average_over_hour(e))
    return new_dict


def get_daily_data(dictionary):
    new_dict = dict()
    for key, value in dictionary.items():
        new_dict[key] = []
        for e in value:
            new_dict[key].append(average_over_day(e))
    return new_dict

# https://towardsdatascience.com/how-to-split-a-tensorflow-dataset-into-train-validation-and-test-sets-526c8dd29438
def split_ds( ds, ds_size, train_split=0.8, val_split=0.2, shuffle=True, shuffle_size=10000):
    assert (train_split + val_split) == 1

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)
    return train_ds, val_ds


if __name__ == "__main__":

    meta, df = load_data("/Users/lukovic/data/FORWARDS/TNT/raw_data/metadata_Server.pkl",
                             "/Users/lukovic/data/FORWARDS/TNT/raw_data/data_Server.pkl")

    data_y = get_yearly_data_by_id(df)
    data_yr = get_yearly_data_by_year(data_y)

    element = data_yr[2002][0]

    new = element.groupby(['doy', 'hour'])

    list = []
    indx = []
    for a, b in new:
        list.append(b)
        indx.append(a)

    df = {'index': [], 'series_id': [], 'ts': [], 'year': [], 'doy': [], 'hour': [], 'value': []}

    for e in list:
        df['index'].append(e.index[0])
        df['series_id'].append(e.series_id.iloc[0])
        df['ts'].append(e.ts.iloc[0])
        df['year'].append(e.year.iloc[0])
        df['doy'].append(e.doy.iloc[0])
        df['hour'].append(e.hour.iloc[0])
        df['value'].append(e.value.mean())

    df_hourly = pd.DataFrame(df)
    df_hourly.set_index('index', inplace=True)
    df_hourly.index.name = ""
