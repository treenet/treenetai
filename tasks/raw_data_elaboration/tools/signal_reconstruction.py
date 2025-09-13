import numpy as np
import pyreadr
import pandas as pd
from functools import reduce
from pathlib import Path
import itertools
import tensorflow as tf
import random
import math

# Sec: ----------------------------------------
# Sec: Multichannel dendrometer data processing
# Sec: ----------------------------------------


def create_multi_dendro_channel(df, meta, species, number_of_trees=3, input_sets=3, global_input_selection=True):
    """
    INPUT: df: dictionary of pandas data frames - dendrometer signals for a single tree
           meta: pandas dataframe - metadata associated to the input dictionary
           species: list - species to consider for the model
           number_of_trees: integer - number of tree dendrometer signals for input
           input_sets: integer - number of different sets of signals to consider
           global_input_selection: boolean - whether to consider trees in the same site or sample globally 
    """

    print('creating multi-channel dataframe')
    print('species used: ', species)
    print('')
    
    series_ids = list(meta[meta.tree_species.isin(species)].series_id)
    
    
    print('Number of tree signals considered: ', len(series_ids))
    print('start')

    data = {}
    counter = 0
    for key in series_ids:
        try: # NOTE: There might be cases where the metadata contains an entry for the signal, but the corresponding data frame is empty
            data[key] = df[key][['ts', 'value']].rename(columns={"value": str(key)})
            # NOTE: makes sure that the column names containing the signal are different. The timestamp column name remains
            #  the same. This is important when using the extrac_overlapping_section function later.
        except:
            print(str(counter) + ") Series with ID: " + str(key) + " does not exist even though an entry is present in the metadata. i.e. metadata points to a data frame that does not exist.")
            counter += 1

    # TODO: the commented code below is an attempt to extract a single signal and use it as test data. Perhaps this could
    #  be done in a better way.
    # test_value, test_key = random.choice(list(data.items()))  # note: extract one complete signal randomly
    # data.pop(test_key)  # note: remove the entry from the original dictionary
    # data_test = {test_key, test_value}  # note: a dictionary with a single test signal with signal id

    permutation_list = _get_permutation_list(series_ids, number_of_trees, input_sets, meta, global_input_selection)
    # NOTE: the permutation_list is a dictionary that contains the permutations of ids of the input dendrometer signals
    #  for each key, which corresponds to the id of the label signal. If 'permutations' is false, then a more restrictive combination of
    #  input signals is considered. See the function for details. 

    combined_signals = _merge_signals(data, permutation_list)
    # NOTE: the function _merge_signals puts together the input dendrometer signals and the label signal and then
    #  extracts the timestaps for which they all overlap. It returns all the time series together in a single dataframe.
    #  combined_signals is a dictionary, where the key is the series id. Each key corresponds to a list of dataframes.
    #  Each dataframe of the list contains also the time series that corresponds to the key=series_id.

    # newmeta = meta[['series_id', 'site_id', 'tree_dbh', 'tree_height', 'tree_twd_max_gp', 'tree_twd_med_gp',
    #                'tree_twd_max_nogp', 'tree_twd_med_nogp', 'tree_gro_start_doy_med', 'tree_gro_end_doy_med',
    #                'tree_gro_max_yr', 'tree_gro_med_yr', 'tree_gro_min_yr', 'tree_gro_med_month', 'tree_gro_med_week',
    #                'tree_gro_med_day', 'tree_gro_med_hr', 'tree_timing_gro_week_max', 'tree_timing_gro_hour_max',
    #                'tree_grohours_med', 'site_xcor', 'site_ycor', 'site_altitude', 'site_annual_temp',
    #                'site_annual_precip']]
    # TODO: this selection is to avoid non-numerical arguments. They also have to be included with one-hot encoding.

    ############################## #TODO ######################################################
    # NOTE: METADATA addition
    # TODO: The addition of metadata is done for now in the Python notebook. Once it is ready, the code should be moved here in 
    # the form of functions

    newmeta = meta[['series_id']]

    combined_signals_with_metadata = _add_metadata_features(combined_signals, newmeta)
    # NOTE: attaches the metadata values to the time series, also in the form of a constant time series for each
    #  metadata feature.

    #return _dictionary_to_list(combined_signals_with_metadata), newmeta
    # note: returns a dictionary where the key is the series_id and the value is the dataframe
    ##########################################################################################

    return combined_signals_with_metadata


def _add_metadata_features(signals, newmeta):
    for key, value in signals.items():
        # note: each value contains a particular permutation of the signals
        for item in value:
            # note: transofrm time stamps into year, month, day, hour format
            # item['year'] = pd.to_datetime(item['ts']).dt.year
            # item['month'] = pd.to_datetime(item['ts']).dt.month
            # item['day'] = pd.to_datetime(item['ts']).dt.day
            item['hour'] = pd.to_datetime(item['ts']).dt.hour
            item['doy'] = item.ts.dt.dayofyear
            # note: assign the values of the metadata features to the dataframes
            for el in list(newmeta)[1:]:  # Note: excludes the first entry, the series id
                temp = newmeta[newmeta.series_id == key][el].to_list()[0]
                if not temp:
                    temp = 0.0
                if not isinstance(temp, float):
                    temp = float(temp)
                item[el] = temp
    return signals


def _merge_signals(data, permutation_list):
    all_signals = dict()
    for key, permuations in permutation_list.items():
        for sequence in permuations:  # note: 'permutations' contains a list of all permutations related to 'key'
            temp_list = []
            for indx in sequence:  # note: iterates through all the signals in a single permutation set
                temp_list.append(data[indx])  # Note: appends the input dendrometer signals
            temp_list.append(data[key])  # Note: appends the label dendrometer signal with key 'key'
            signals = _extract_overlapping_section(temp_list)
            if len(signals) > 0:
                all_signals.setdefault(key, []).append(signals)

    # TODO: below is an old for loop. Same as above but using 'for e in series_ids'
    # for e in series_ids:  # note: e is a list of series_id values
    #    for ee in permutation_list[e]:
    #        temp_list = []
    #        for eee in ee:
    #            temp_list.append(data[eee])  # Note: appends the input dendrometer signals
    #        temp_list.append(data[e])  # Note: appends the label dendrometer signal
    #        signals = _extract_overlapping_section(temp_list)
    #        if len(signals) > 0:
    #            all_signals[e] = signals

    return all_signals


def _get_permutation_list_new(series_ids, number_of_trees, meta, global_input_selection=True):
    """
    INPUT:
    -------------------------------------------------------------------------------------------------------
    series_ids:             list - list of all dendrometer time series ids.
    number_of_trees:        integer - number of individual dendrometer time series to consider as input for the model.
    meta:                   data frame - the metadata corresponding to the input time series
    global_input_selection: boolean - If true, tree signals AND their permutations for the model input are 
                            selected randomly from all plots. If false, tree signals and their permutations 
                            are sampled only from the same plot as the label signal.

    OUTPUT:
    -------------------------------------------------------------------------------------------------------
    """

    print('Calculating permutations...')

    if global_input_selection:
        print('<-> Selecting randomly signals from all sites')
    else:
        print('<-> Using signals from same sites only')
    dictionary = dict()
    for i in range(len(series_ids)):
        # NOTE: each signal with series id series_ids[i] is the LABEL or reference signal. It is NOT part of the model input. 
        # It will be used as the ground truth. To this signal other dendrometer signals are associated, 
        # which will be part of the model input.
        permutation_list = []
        ids = series_ids[:i] + series_ids[i + 1:]  # NOTE: Removes and excludes the id under consideration, i.e. id with index 'i'.
        if global_input_selection:
            # NOTE: If the list of available signal ids is [1, 2, 3, 4, 5, 6] and nuber_of_trees = 3, then this method returns [[1,2,3],[4,5,6]] 
            # as the permutations. In reality, we should have [1,2,3], [1,2,4], [1,2,5], [1,2,6], [2,3,4], [2,3,5] .... 
            if len(ids) >= number_of_trees: 
                input_sets = int(len(ids)/number_of_trees)
                if input_sets >= 1:
                    selection = np.random.choice(ids, input_sets * number_of_trees, replace=False)
                    # NOTE: selects n='input_sets*number_of_trees' random ids without replacement
                    selection = selection.reshape((input_sets, number_of_trees))
                    # NOTE: reshapes the ids into n='permutation_sampels' sets of m='number_of_trees'.
                else:
                    print("ERROR! There are not enough dendrometer signals to create a tuple for the requested " + str(number_of_trees) + "-tree input combination!")
            
                # NOTE: the permutation_list collects all the permutation sequences corresponding to a single key/index
                for sequence in selection:  # NOTE: slection contains n randomly selected sequences, n=see previous note
                    for element in itertools.permutations(sequence):  # NOTE: returns all the permutations.
                        permutation_list.append(element)
        else:
            ids_site = []
            # NOTE: find site that corresponds to signal
            site_id = meta[meta['series_id'] == series_ids[i]]['site_id'].values[0]
            for e in ids:
                # NOTE: select only signals from the same site
                if meta[meta['series_id'] == e]['site_id'].values[0] == site_id:
                    ids_site.append(e)
            # NOTE: the total number of signals, number_of_trees, on a particular site should be more than 'number_of_trees'.
            if len(ids_site) >= number_of_trees:
                # NOTE: The list ids_site does not contain the id of the label/reference signal because the list is a sub-list of ids, which excludes the label signal.
                #       The length of the ids_site list should be at least as long as nuber_of_trees. This gives a single combination. 
                for element in itertools.permutations(ids_site, number_of_trees): # NOTE: iterates over all permutations of 'number_of_trees' elements
                    permutation_list.append(element)

        dictionary[series_ids[i]] = permutation_list

    return dictionary



def _get_permutation_list(series_ids, number_of_trees, input_sets, meta, global_input_selection=True):
    """
    INPUT:
    -------------------------------------------------------------------------------------------------------
    series_ids:             list - list of all dendrometer time series ids.
    number_of_trees:        integer - number of individual dendrometer time series to consider as input for the model.
    input_sets:             integer - number of input dendrometer signal sets of 'number_of_trees' to create.
    meta:                   data frame - the metadata corresponding to the input time series
    global_input_selection: boolean - If true, tree signals AND their permutations for the model input are 
                            selected randomly from all plots. If false, tree signals and their permutations 
                            are sampled only from the same plot as the label signal.

    OUTPUT:
    -------------------------------------------------------------------------------------------------------
    """

    print('Calculating permutations...')

    if global_input_selection:
        print('<-> Selecting randomly signals from all sites')
    else:
        print('<-> Using signals from same sites only')
    dictionary = dict()
    for i in range(len(series_ids)):
        # NOTE: each signal with series id series_ids[i] is the LABEL or reference signal. It is NOT part of the model input. 
        # It will be used as the ground truth. To this signal other dendrometer signals are associated, 
        # which will be part of the model input.
        permutation_list = []
        ids = series_ids[:i] + series_ids[i + 1:]  # NOTE: Removes and excludes the id under consideration, i.e. id with index 'i'.
        if global_input_selection:
            # NOTE: If the list of available signal ids is [1, 2, 3, 4, 5, 6] and nuber_of_trees = 3, then this method returns [[1,2,3],[4,5,6]] 
            # as the permutations. In reality, we should have [1,2,3], [1,2,4], [1,2,5], [1,2,6], [2,3,4], [2,3,5] .... 
            if len(ids) >= number_of_trees: 
                samples = int(len(ids)/number_of_trees)
                if samples >= input_sets:
                    selection = np.random.choice(ids, input_sets * number_of_trees, replace=False)
                    # NOTE: selects n='input_sets*number_of_trees' random ids without replacement
                    selection = selection.reshape((input_sets, number_of_trees))
                    # NOTE: reshapes the ids into n='permutation_sampels' sets of m='number_of_trees'.
                else:
                    selection = np.array(ids_site[0:samples*number_of_trees])
                    selection = selection.reshape((samples, number_of_trees))
            
                # NOTE: the permutation_list collects all the permutation sequences corresponding to a single key/index
                for sequence in selection:  # NOTE: slection contains n randomly selected sequences, n=see previous note
                    for element in itertools.permutations(sequence):  # NOTE: returns all the permutations.
                        permutation_list.append(element)
        else:
            ids_site = []
            # NOTE: find site that corresponds to signal
            site_id = meta[meta['series_id'] == series_ids[i]]['site_id'].values[0]
            for e in ids:
                # NOTE: select only signals from the same site
                if meta[meta['series_id'] == e]['site_id'].values[0] == site_id:
                    ids_site.append(e)
            # NOTE: the total number of signals, number_of_trees, on a particular site should be more than 'number_of_trees'.
            if len(ids_site) >= number_of_trees:
                # NOTE: The list ids_site does not contain the id of the label/reference signal because the list is a sub-list of ids, which excludes the label signal.
                #       The length of the ids_site list should be at least as long as nuber_of_trees. This gives a single combination. 
                for element in itertools.permutations(ids_site, number_of_trees): # NOTE: iterates over all permutations of 'number_of_trees' elements
                    permutation_list.append(element)

        dictionary[series_ids[i]] = permutation_list

    return dictionary


def _list_to_dictionary(list_of_df, series_ids_temp, features):
    data_dictionary = dict()
    ids = []
    for e in list_of_df:
        if len(e) > 10:  # note: makes sure that the dataframe/signal is not empty
            s_id = e.series_id.loc[0]
            if s_id in series_ids_temp:
                data_dictionary[s_id] = e[features]
                ids.append(s_id)
    return data_dictionary, ids


def _dictionary_to_list(dictionary):
    temp_list = []
    for item in dictionary.values():
        for element in item:
            temp_list.append(element)

    return temp_list


def _extract_overlapping_section(list_of_df):
    data_temp = reduce(lambda left, right: pd.merge(left, right, on=['ts']), list_of_df)
    return data_temp


def _combine_dataframes(df):
    return pd.concat(df, ignore_index=True, sort=False)
