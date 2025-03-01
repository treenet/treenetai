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
    data_yr = dict()
    for key, value in dictionary.items():
        df_list = _ts_by_year(value)
        temp_list = []
        for e in df_list[:]:
            if e.doy.iloc[0] == 1 or e.doy.iloc[-1] > 364:
                temp_list.append(e)
        data_yr[key] = temp_list
    return data_yr


def get_yearly_data_by_year(dictionary, yearly_by_id):
    if not yearly_by_id:
        dictionary = get_yearly_data_by_id(dictionary)
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
    data['doy'] = data.ts.dt.dayofyear # note: add a column with the day of year
    years = data.year.unique()
    output = []
    for year in years:
        output.append(data[data.year == year])

    return output


