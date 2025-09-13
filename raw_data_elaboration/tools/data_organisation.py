import pandas as pd
import pickle
import os
import numpy as np
import h5py
from datetime import datetime
import metpy.calc as metpy
from metpy.units import units


# TITLE: METADATA STATISTICS ############################################
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



# TITLE: SEPARATION BY YEAR ############################################

def get_yearly_data_by_id(dictionary):
    """ Separates the time series of each sensor into a list of time series by year. 
    Creates a dictionary where the key is the series id and the value is a list of time series by year. 
    """
    # NOTE: take care of leap years, they are also included
    data_yr = dict()
    for id, timeseries in dictionary.items():
        df_list = _ts_by_year(timeseries)
        temp_list = []
        for e in df_list[:]:
            if e.doy.iloc[0] == 1 or e.doy.iloc[-1] > 364:
                temp_list.append(e)
        data_yr[id] = temp_list
    return data_yr


def get_yearly_data_by_year(dictionary, yearly_by_id):
    """ Sorts the yearly time series according to year. 
    Creates a dictionary where the key is the year and the value are time series of different sensors of the same year. 
    """
    # NOTE: take care of leap years, they are also included
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


def _ts_by_year(data):
    """ Separates a continuous time series into a list of yearly time series segments """
    data.loc[:, 'year'] = data.ts.dt.year  # note: add a column with the year
    data.loc[:, 'doy'] = data.ts.dt.dayofyear  # note: add a column with the day of year
    data.loc[:, 'hour'] = data.ts.dt.hour  # note: add a column with the hour of day
    years = data.year.unique()
    output = []
    for year in years:
        output.append(data[data.year == year])

    return output

# TITLE: SEPARATION BY MONTH ############################################

def get_monthly_data(metadata, data): # TODO
    output = []
    return output

# TITLE: COARSE-GRAINING BY TIME ############################################

# HOURLY scale

def get_hourly_data(input):
    """ The function averages the data over every hour.
    Input: A time series with 10 min resolution
    Output: A time series with 1 h resolution
    IMPORTANT: make sure that the dataframes have DOY and hour columns
    """
    
    df = input.copy() # NOTE: this dataframe copy process eliminates the SettingWithCopyWarning Error.

    df.loc[:, 'year'] = df.ts.dt.year  # NOTE: add a column with the year
    df.loc[:, 'month'] = df.ts.dt.month  # NOTE: add a column with the month
    df.loc[:, 'day'] = df.ts.dt.day  # NOTE: add a column with the month
    df.loc[:, 'doy'] = df.ts.dt.dayofyear  # NOTE: add a column with the day of year
    df.loc[:, 'hour'] = df.ts.dt.hour  # NOTE: add a column with the hour of day
    
    temp = df.groupby(['year', 'doy', 'hour'])
    d_out = {'ts': [], 'value': []}
    for _, e in temp:
        d_out['ts'].append(pd.to_datetime(datetime(*[e.year.iloc[0], e.month.iloc[0], e.day.iloc[0], e.hour.iloc[0]])))
        d_out['value'].append(e.value.mean())  # NOTE: This is where the averaging is done
        # NOTE: AVERAGING - If the resolution of the data is 10 min, then the average is done over 6 values. However, often it is the 
        # case that there are less than 6 values because some timestamps are missing. Therefore the average is done over fewer values. 
        # I have decided to keep these values in order to avoid too many NaNs and therefore gaps in the data. The timestamp is always 
        # taken to be the timestapm of the first row within the group. Minutes are not included when constructing the timestamp above, 
        # therefore the timestamp will always have the minutes variable as zero. The reson for this is because, as already mentioned, 
        # there are cases where data is missing and therefore the timestamp of the first row of the group might not have minutes equal 
        # to zero. This then creates problems later. 
    df_hourly = pd.DataFrame(d_out)
    df_hourly.index.name = ""
    
    return df_hourly

def get_hourly_data_by_id_index(dictionary):
    """ The function averages the data over every hour.
    Input: dictionary where the key is the signal id and value is a time series with 10 min resolution
    Output: dictionary where the key is the signal id and the value is a time series with 1 h resolution
    IMPORTANT: make sure that the dataframes have DOY and hour columns
    """
    new_dict = dict()
    for id, timeseries in dictionary.items():  # note: Iterates over the years
        print("elaborating time series: ", id)
        timeseries.loc[:, 'year'] = timeseries.ts.dt.year  # note: add a column with the year
        timeseries.loc[:, 'doy'] = timeseries.ts.dt.dayofyear  # note: add a column with the day of year
        timeseries.loc[:, 'hour'] = timeseries.ts.dt.hour  # note: add a column with the hour of day
        temp = timeseries.groupby(['year', 'doy', 'hour'])
        new_dict[id] = _average_over_hour(temp)  # note: appends 
    return new_dict


def get_hourly_data_by_year_index(dictionary):
    """ The function averages the data over every hour.
    Input: dictionary where the key is the year and value is a list of yearly data with 10 min resolution
    Output: dictionary where the key is the year and the value is a list of yearly data with 1 h resolution
    IMPORTANT: make sure that the dataframes have DOY and hour columns
    """
    new_dict = dict()
    for year, list in dictionary.items():  # note: Iterates over the years
        new_dict[year] = []
        for e in list:  # note: Iterates over different trees for the same year
            temp = e.groupby(['doy', 'hour'])
            new_dict[year].append(_average_over_hour(temp))  # note: appends 
    return new_dict


# DAILY scale
def get_daily_data_by_id_index(dictionary, hourly_data):  
    # TODO: what to do if hourly_data is set as True but the input data has no 'year', 'doy' and 'hour' values? This has to be fixex.
    """ The function averages the data over every day.
    Input: dictionary where the key is the sensor id and value is a list of yearly data with 1h resolution
    Output: dictionary where the key is the sensor id and the value is a list of yearly data with 1 day resolution
    """
    new_dict = dict()
    if not hourly_data:
        dictionary = get_hourly_data_by_id_index(dictionary)
    for id, timeseries in dictionary.items():
        temp = timeseries.groupby(['year', 'doy'])
        new_dict[id] = (_average_over_day(temp))
    return (new_dict)


def get_daily_data_by_year_index(dictionary, hourly_data):
    """ The function averages the data over every day.
    Input: dictionary where the key is the year and value is a list of yearly data with 1h resolution
    Output: dictionary where the key is the year and the value is a list of yearly data with 1 day resolution
    """
    new_dict = dict()
    if not hourly_data:
        dictionary = get_hourly_data_by_year_index(dictionary)
    for year, list in dictionary.items():
        new_dict[year] = []
        for e in list:
            temp = e.groupby(['doy'])
            new_dict[year].append(_average_over_day(temp))
    return (new_dict)


def _average_over_hour(df_in):
    d_out = {'index': [], 'series_id': [], 'ts': [], 'year': [], 'doy': [], 'hour': [], 'value': []}
    for _, e in df_in:
        d_out['index'].append(e.index[0])
        d_out['series_id'].append(e.series_id.iloc[0])
        d_out['ts'].append(e.ts.iloc[0])
        d_out['year'].append(e.year.iloc[0])
        d_out['doy'].append(e.doy.iloc[0])
        d_out['hour'].append(e.hour.iloc[0])
        d_out['value'].append(e.value.mean())  # NOTE: This is where the averaging is done
    df_hourly = pd.DataFrame(d_out)
    df_hourly.set_index('index', inplace=True)
    df_hourly.index.name = ""

    # NOTE: make sure that the first and last row are whole hours, without minutes (i.e. minutes = 0). 
    # The last row might always be a whole hour due to the construction of the function. Check anyway.
    # In positive case, remove the row.
    if df_hourly.iloc[0].ts.minute != 0:
        indx = df_hourly.iloc[0].name
        df_hourly = df_hourly.drop(index=indx)

    if df_hourly.iloc[-1].ts.minute != 0:
        indx = df_hourly.iloc[-1].name
        df_hourly = df_hourly.drop(index=indx)
        
    return df_hourly


def _average_over_day(df_in):  # TODO: check this function
    d_out = {'index': [], 'series_id': [], 'ts': [], 'year': [], 'doy': [], 'value': []}
    for _, e in df_in:
        d_out['index'].append(e.index[0])
        d_out['series_id'].append(e.series_id.iloc[0])
        d_out['ts'].append(e.ts.iloc[0])
        d_out['year'].append(e.year.iloc[0])
        d_out['doy'].append(e.doy.iloc[0])
        d_out['value'].append(e.value.mean())  # NOTE: This is where the averaging is done
    df_daily = pd.DataFrame(d_out)
    df_daily.set_index('index', inplace=True)
    df_daily.index.name = ""

    # NOTE: make sure that the first and last row are whole hours, without minutes (i.e. minutes = 0). 
    # The last row might always be a whole hour due to the construction of the function. Check anyway.
    # In positive case, remove the row.
    if df_daily.iloc[0].ts.hour != 0:
        indx = df_daily.iloc[0].name
        df_daily = df_daily.drop(index=indx)

    if df_daily.iloc[-1].ts.hour != 0:
        indx = df_daily.iloc[-1].name
        df_daily = df_daily.drop(index=indx)

    return df_daily


# TITLE: DATA-FRAME SPLITTING ############################################

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


# TITLE: WEATHER TOOLS ############################################

import metpy.calc as metpy
from metpy.units import units

def getVPD(p, T, rh):
    """ A simple function to calculate the VPD from atmospheric pressure, temperature and relative humidity. It uses the MetPy library. """
    mixing_ratio = metpy.mixing_ratio_from_relative_humidity(p * units.hPa, T * units.degC, rh).to('g/kg')
    svp = metpy.saturation_vapor_pressure(T * units.degC).to('hPa')
    vp = metpy.vapor_pressure(p * units.hPa, mixing_ratio * units('g/kg')).to('hPa')
    vpd = vp - svp
    return vpd


# TITLE: BASIC TOOLS ############################################

def merge_time_series(dfs):
    """Input: A list of dataframes with two coumns, (ts, value)
       Output: A single dataframe with multiple columns, (ts, value1, value2, ....) 
    """
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='ts', how='outer')
    return merged_df


####################################### E N D #################################################
###############################################################################################





if __name__ == "__main__":

    # NOTE: The code below is for testing the correctness of the functions above

    df = pd.read_pickle("/storage/lukovic/Data/FORWARDS/treenet/server_data/combined_dendro_climate_dictionary.pkl")
    meta = pd.read_pickle("/storage/lukovic/Data/FORWARDS/treenet/server_data/metadata.pkl")

    data_y = get_yearly_data_by_id(df)
    data_yr = get_yearly_data_by_year(data_y, yearly_by_id=True)

    element = data_yr[2002][0]

    new = element.groupby(['doy', 'hour'])

    list = []
    indx = []
    for a, b in new:
        list.append(b)
        indx.append(a)

    df = {'index': [], 'series_id': [], 'ts': [], 'year': [], 'doy': [], 'hour': [], 'stem_radius': []}

    for e in list:
        df['index'].append(e.index[0])
        df['series_id'].append(e.series_id.iloc[0])
        df['ts'].append(e.ts.iloc[0])
        df['year'].append(e.year.iloc[0])
        df['doy'].append(e.doy.iloc[0])
        df['hour'].append(e.hour.iloc[0])
        df['stem_radius'].append(e.stem_radius.mean())

    df_hourly = pd.DataFrame(df)
    df_hourly.set_index('index', inplace=True)
    df_hourly.index.name = ""

    print(df_hourly)
