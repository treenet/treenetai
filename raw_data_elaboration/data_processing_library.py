import numpy as np
import pyreadr
import pandas as pd
from functools import reduce
from pathlib import Path
import itertools
import tensorflow as tf
import random


# Sec: ----------------------------------------
# Sec: Segmentation
# Sec: ----------------------------------------

def make_segments(df, length_in_days, time_resolution, normalization):

    if df.shape[0] < 2 or df.shape[1] < 2:  # NOTE: makes sure that the dataset is not empty
        raise Exception('The data used for the experiment is not of the correct format. The data is either '
                        'empty or there is only one time series channel.')

    df = df.dropna()  # Note: remove all missing data from the dataframe
    segment_length = length_in_days * 24 * time_resolution  # Note: converts the segment length from days to time-samps
    start_time = 0

    idx = 0
    max_length = len(df)

    while (idx + segment_length) < max_length:
        if ((df.hour[df.index[idx]] == start_time) and
                (df.index[segment_length + idx] - df.index[idx] == segment_length)):
            segment = df.iloc[idx:idx + segment_length]
            # NOTE each segment has the shape [time steps, number of features/channels].
            #  This should be considered when creating an input for a machine learning model.

            segment = segment.drop(['ts'], axis=1)  # Note: removes the time-stamp

            if normalization:
                segment_normalized = _rescale(segment)
            else:
                segment_normalized = segment.apply(pd.to_numeric).to_numpy()
                # note: the apply(pd.to_numeric) function makes sure that all the entries are of the same numeric type.
                #  When using data from the server, it is of the decimal type, which creates problems when the data is
                #  saved in the TFrecords format.

            # Note: At this point the dataframe has been converted to a numpy array
            yield segment_normalized
            idx += segment_length
        else:
            idx += 1


def add_gaps(segment, segment_length, gap_size, gap_type, channels_to_fix):
    # Todo: make sure the function works from within this file.
    # NOTE: Segment is the data input. A multi-channel time series in the form of a numpy array.
    #  It has a shape of the form: (segment length, number of channels).
    #  The output is a numpy array of the same shape.
    # TODO (gap_type): add the possibility of sampling gap sizes from a uniform and an exponential distribution
    new_segment = []

    for i in range(segment.shape[1]): # NOTE: for loop over all the channels of time series
        array = segment[:, i] # NOTE: array represents the i'th channel of the time series
        if i in channels_to_fix:
            start_index = np.random.randint(segment_length - gap_size)
            # NOTE: selects a random point on the array to start with the gap
            if gap_type == 'constant':
                end_index = start_index + gap_size
            elif gap_type == 'uniform':
                end_index = start_index + gap_size
            elif gap_type == 'exponential':
                end_index = start_index + gap_size
            else:
                raise Exception(f'The distribution of gaps is unknown.')

            array[start_index:end_index] = -1

            # NOTE we assign -1 to the region of the array where the gap is.
            # TODO choose a value to assign when the array is not normalized.
            #  It cannot be -1 since the signal might be negative also.

        new_segment.append(array)

    return np.transpose(np.asarray(new_segment))
    # NOTE: converts the list of numpy arrays into a multi-dimensional numpy array and returns it.


def _rescale(segment):
    # TODO (Important!!) Maybe it is better to normalize the entire time series and not the segments separately.
    #  If we consider channels of a segment that have a constant value in that time period but that nevertheless
    #  change over a longer time period, then giving them all a value of 0.5 does not really make sense.
    # TODO the normalization is done with respect to the entire signal, i.e. the ground truth.
    #  There should also be an option to normalize with respect to the given signal. i.e. the signal with a gap.
    #  The reason is that once the model is used for prediction, we will only have signals with gaps.
    #  In that case we don't know what the min or max values are because they might be part of the missing data.
    #  This is a delicate point that has to be considered.
    # TODO consider adding the case where the signal is not normalized but is, nevertheless,
    #  shifted so that the minimum value is zero.

    if len(segment.shape) != 2:
        raise Exception(f'The dimension of the segment array should be 2. '
                        f'In this case the shape of the array is ' + str(segment.shape))

    for e in list(segment):
        if e == 'hour':
            segment.loc[:, e] = segment.loc[:, e].div(24)
        elif e == 'month':
            segment.loc[:, e] = segment.loc[:, e].div(12)
        else:
            min_val = segment[e].min()
            max_val = segment[e].max()
            if np.abs(max_val - min_val) > 1e-4:
                segment.loc[:, e] = _normalize(segment.loc[:, e])
            # else:
            #    if max_val - min_val == 0:
            #        segment.loc[:, e] = 0.0
            #    else:
            #        segment.loc[:, e] = 0.5

    return np.asarray(segment.apply(pd.to_numeric))


def _normalize(array):
    array = (array - array.min()) / (array.max() - array.min())
    return array

# Sec: END ------------------------------------



# Sec: ----------------------------------------
# Sec: Multichannel dendrometer data processing
# Sec: ----------------------------------------

# Note: Relevant for collecting into the same dataframe dendrometer signals from different trees.
def multi_dendro_channel(df, meta, species, trees=3, permutation_samples=3, combination_samples_rand=True):

    print('creating multi-channel dataframe')
    print('species used: ', species)
    print('')
    series_ids_temp = list(meta[meta.tree_species == species[0]].series_id)[0:20]
    print('Number of tree signals considered: ', len(series_ids_temp))
    # Todo: there should be a loop here in case more than one species is used. There should also be a check for the
    #  'all' command in the script 1_make_records.sh (if all then ...).
    # note: series_ids is a list of unique tree signal ids that correspond to dendrometer signals of trees of the same
    #  species. The reason it is temporary is because some of the entries downloaded from the database might be empty,
    #  without data. This is checked in the _list_to_dictionary function, which provides the final series_id list.
    # todo: so far we are only using one tree species so there is only one list. This has to be changed when more tree
    #  species are used.

    data, series_ids = _list_to_dictionary(df, series_ids_temp, ['ts', 'value'])
    # note: creates a dictionary such that the key corresponds to the id of the signal and value of the dictionary
    #  corresponds to the signal itself.
    # note: data is the dictionary and series_ids is the new (updated) id list.

    for key, value in data.items():
        # note: makes sure that the column names containing the signal are different. The timestamp column name remains
        #  the same. This is important when using the extrac_overlapping_section function later.
        data[key] = data[key].rename(columns={"value": str(key)})

    # TODO: the commented code below is an attempt to extract a sinle signal and use it as test data. Perhaps this could
    #  be done in a better way.
    # test_value, test_key = random.choice(list(data.items()))  # note: extract one complete signal randomly
    # data.pop(test_key)  # note: remove the entry from the original dictionary
    # data_test = {test_key, test_value}  # note: a dictionary with a single test signal with signal id

    # TODO: The current function (multi_dendro_channel) should be separated into two functions. The first new function
    #  should finish here.

    permutation_list = _get_permutation_list(series_ids, trees, permutation_samples, meta, combination_samples_rand)
    # Note: the permutation_list is a dictionary that contains the permutations of ids of the input dendrometer signals
    #  for each key, which corresponds to the id of the label signal.

    combined_signals = _combine_signals(data, permutation_list)
    # Note: the function _combine_signals puts together the input dendrometer signals and the label signal and then
    #  extracts the timestaps for which they all overlap. It returns all the time series together in a single dataframe.
    #  combined_signals is a dictionary, where the key is the series id. Each key corresponds to a list of dataframes.
    #  Each dataframe of the list contains also the time series that corresponds to the key=series_id.

    # newmeta = meta[['series_id', 'site_id', 'tree_dbh', 'tree_height', 'tree_twd_max_gp', 'tree_twd_med_gp',
    #                'tree_twd_max_nogp', 'tree_twd_med_nogp', 'tree_gro_start_doy_med', 'tree_gro_end_doy_med',
    #                'tree_gro_max_yr', 'tree_gro_med_yr', 'tree_gro_min_yr', 'tree_gro_med_month', 'tree_gro_med_week',
    #                'tree_gro_med_day', 'tree_gro_med_hr', 'tree_timing_gro_week_max', 'tree_timing_gro_hour_max',
    #                'tree_grohours_med', 'site_xcor', 'site_ycor', 'site_altitude', 'site_annual_temp',
    #                'site_annual_precip']]
    # todo: this selection is to avoid non-numerical arguments. They also have to be included with one-hot encoding.

    newmeta = meta[['series_id']]

    combined_signals_with_metadata = _add_metadata_features(combined_signals, newmeta)
    # note: attaches the metadata values to the time series, also in the form of a constant time series for each
    #  metadata feature.

    return _dictionary_to_list(combined_signals_with_metadata), newmeta
    # note: returns a dictionary where the key is the series_id and the value is the dataframe


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


def _combine_signals(data, permutation_list):
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

    # Todo: below is an old for loop. Same as above but using 'for e in series_ids'
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


def _get_permutation_list(series_ids, trees, permutation_samples, meta, rand):
    print('Calculating permutations...')
    # note: i = id of the signal under consideration (the label); series_ids = ids of all the signals;
    #  permutation_samples = number of permutations of the selected signals to create;
    #  trees = number of signals to use as input; meta = the metadata table;

    #  note: rand = determines whether the selection of signals from ids should be random. If not, the input signals and
    #   corresponding label will be from the same sub-plot.
    if rand:
        print('<-> Selecting randomly the input signal combinations')
    else:
        print('<-> Using all the possible input signal combinations')
    dictionary = dict()
    for i in range(len(series_ids)):
        # note: each signal with series id series_ids[i] is the label or reference signal. To this signal other
        #  dendrometer signals are associated, which are also used to construct the input.
        ids = series_ids[:i] + series_ids[i + 1:]  # note: Removes the id under consideration, i.e. id with index 'i'.
        if rand:
            selection = np.random.choice(ids, permutation_samples * trees, replace=False)
            # note: selects n='permutation_samples*trees' random ids without replacement
            selection = selection.reshape((permutation_samples, trees))
            # note: reshapes the ids into n='permutation_sampels' sets of m='trees'.
        else:
            ids_temp = []
            # note: find site that corresponds to signal
            site_id = meta[meta['series_id'] == series_ids[i]]['site_id'].values[0]
            for e in ids:
                # note: select only signals from the same site
                if meta[meta['series_id'] == e]['site_id'].values[0] == site_id:
                    ids_temp.append(e)
            # note: the total number of trees on a particular site should be more than 'trees'.
            if len(ids_temp) >= trees:
                samples = int(len(ids_temp)/trees)
                if samples >= permutation_samples:
                    selection = np.random.choice(ids_temp, permutation_samples * trees, replace=False)
                    selection = selection.reshape((permutation_samples, trees))
                else:
                    selection = np.array(ids_temp[0:samples*trees])
                    selection = selection.reshape((samples, trees))
            else:
                selection = []

        if len(selection) > 0:
            temp_list = []
            # note: the temp_list collects all the permutation sequences corresponding to a single key/index
            for sequence in selection:  # note: slection contains n randomly selected sequences, n=see previous note
                for element in itertools.permutations(sequence):  # note: determines all the permutations.
                    temp_list.append(element)

            dictionary[series_ids[i]] = temp_list

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


# Sec: END ------------------------------------



# sec ##################################################################################################
# sec. This subsection is old and obsolete. Its codes should be merged with the others and then deleted.
# TODO: the function below should be divided into smaller parts. Most of the parts already exist as functions listed
#  above.


def nn_signal_preparation(df, meta, min_trees=2):
    # Note: This is the orignal function for dealing with nearest neighbour signals. The functions above are pieces
    #  from it. Perhaps the remeining pieces shouold be made into functions as above, if not already done, and the
    #  function shouold be abbandoned. It is too big and coumbersome.
    # Note: This function is to be used with the experiment type "nearest-neighbours"
    # Todo: reorganise the metadata file so that its contents can also be used for labelling. Since the time-series of
    #  the same subplot are merged, most of the metadata will be the same. Make sure to deal properly with the metadata
    #  that differs from tree to tree in the same subplt.
    # Todo: the function is too long. It should be divided into subfunctions.
    # Note: This function finds trees belonging to the same subplot and combines the dendrometer signals of the
    #  subplots.

    # Sec: 1 ********* Add a unique name (integer number) for each subplot
    meta['sub'] = meta.groupby(['station', 'subplot']).grouper.group_info[0]
    # Note: trees of the same subplot are placed together, aligned according to time and given a unique label.
    #  An extra column is added to the meta dataframe, with subplots ranging from 0 to meta['sub'].max()

    # Sec: 2 ********* Create a lists of each subplot with the names of the unique dendrometer measurements
    subplots = []
    # Note: the unique identification values (series_name) are grouped into subplots and stored in this list.
    for i in range(meta['sub'].max() + 1):  # Note: The '+1' is essential to go through all the elements.
        subplots.append(meta[meta['sub'] == i]['series_name'].to_list())

    # Sec: 3 ********* Select plots that have a minimum of 'min_trees' trees
    # Todo
    new_subplots = []
    for e in subplots:
        if len(e) >= min_trees:
            new_subplots.append(e)
        else:
            for ee in e:
                meta.drop(meta[meta.series_name == ee].index, inplace=True)

    subplots = new_subplots

    # Sec: 4 ********* Extract and collect only the dendrometer signals, together with time-stamps
    channels = list(df)
    dataframes = []
    # Note: the actual dataframes for each time series are grouped and stored in the list
    for e in subplots:  # Note: 'subplots' list contains lists of names (series_names) of each subplot.
        temp = []
        count = 0
        for ee in e:
            temp_dataframe = df[df['series_name'] == ee][[channels[0], channels[2]]].reset_index(drop=True)
            temp_dataframe.rename(columns={channels[2]: str(count)}, inplace=True)
            # Note: in the line above, each singal in the subplot is given a new title/name for the third column.
            #  This is the column where the dendrometer signal/channel is stored and its original name is "LM". The
            #  name is substituted by a number so that we can distignuish all the signals in a subplot when we put them
            #  together into a single dataframe later.
            temp.append(temp_dataframe)
            # TODO: The above statements depend on the ordering of the features (columns) in the dataset.
            #  This should be generalised and improved.
            # Note:
            #  Only the dendrometer time-series is extracted here (for each tree).
            #  It is important to reset the index for all the time-series extracted from the complexive dataframe.
            #  This is because of the merge function that is used below. It is important that the dataframes that are
            #  merged together are also aligned according to the index, otherwise rows with different time stamps will
            #  not be eliminated but only shifted.
            count += 1

        temp.append(df[df['series_name'] == e[0]][[channels[0]] + channels[3:]].reset_index(drop=True))
        # Note: for now we assume that the evironmental signals are the same for each tree inside the same subplot.
        #  It does not matter which subplot we consider. Therefore the value of 'series_name' can be any as long as
        #  it's of the same subplot. When appending the last dataframe, it contains only the environmental data, the
        #  time stamp and the name of the time-series. The series_name label is not included. It is added
        #  subsequently, in the code below.
        # Note: we are using e[0] always (i.e. the first tree in the subplot list) because it does not matter which tree
        #  we use in the subplot, since they all have the same environmental data associated.
        #  TODO: This should be generalised.
        dataframes.append(temp)

    # Sec: 5 ********* Merge all signals from the same plot into the same dataframe according to the time stamp 'ts'
    data = []
    # Note: 'data' is a list of dataframes that correspond to single subplots and contain all the dendrometer signals
    #  of that same subplot.
    counter = 0  # Todo: find a way of not using the counter here. Could lead to errors.
    new_subplots = []
    for e in dataframes:
        # Note: dataframes is a list of lists. Each list (iterated using e) in the list contains the dataframes of all
        #  trees in the same subplot.
        data_temp = reduce(lambda left, right: pd.merge(left, right, on=['ts']), e)
        subplot_list = subplots[counter]  # Note: subplot_list contains the identifiers for each tree of a subplot.
        if len(data_temp) > 0:
            data.append(data_temp)
            new_subplots.append(subplot_list)
        else:
            for ee in subplot_list:
                meta.drop(meta[meta.series_name == ee].index, inplace=True)
        counter += 1
    subplots = new_subplots

    # Note: It is possible that during the merging, some dataframes are lost because none of the time stamps
    #  coincide. It is also possible that the dendrometer signal of one or more trees, but not all, in a subplot is
    #  also lost. Therefore the original 'subplot' list might no longer be valid/correct. For this reason, the new
    #  list 'new_subplots' is introduced.

    # Sec: 6 ********* Add all possible combinations of the dendrometer signals of the same plot to the data frame
    extended_data = []  # Note: Includes all the possible combinations.
    counter = 0
    for e in data:
        n = 0
        if len(e) > 0:  # Note: Checks whether the entry in the dataframe is not empty.
            for el in list(e):  # Note: counts how many columns represent dendrometer signals
                if el.isdigit():
                    n += 1
            if n == 0:  # Note: If there are no dendrometer signals, there is an error.
                raise Exception(f'None of the channel labels correspond to an integer!')

            combinations = permutations(list(e), n, min_trees)
            # Note: the variable min_trees is very important! It makes sure that the number of columns of each
            #  dendrometer dataframe is always the same, i.e. min_trees, irrespective of the number of dendrometers in a
            #  single subplot. What might change is the number of dendrometer dataframes per subplot. If the number of
            #  dendrometers in a subplot is greater than min_trees, then there will be the same number of columns for
            #  each dataframe (with only min_trees dendrometer signals), but the number of dataframes will be larger
            #  than min_trees. If min_trees is 2 and there are 3 dendrometers in a subplot, then there will be 3
            #  dataframes with 2 dendrometer signal columns. Combinations = [['0','1','2'], ['1','2','0'],
            #  ['2','0','1']], therefore, df1 = ['0','1']; df2 = ['1','2']; df3 = ['2','0'].
            # TODO: The combinaitons can be extended to include more data. Since the last column is the label and the
            #  rest are data, then we can also have df1a = ['0','1']; df1b = ['2','1']; df2a = ['1','2'];
            #  df2b = ['0','2']; df3a = ['2','0']; df3b = ['1','0']

            for el in combinations:
                # Note: el is a list of labels. e.g. ['ts', '0', '1', '2', '3', '4', '5', 'series_name', 'LM','rh',
                #  'vpd', 'rad', 'swp', 'total_precip', 'doy', 'year', 'month', 'day']
                dataframe = e[el]
                subplot_name = int(el[1])  # Note: converts the string digit to an integer.
                series_name = subplots[counter][subplot_name]  # Note: this gives a string integer
                name_column = np.zeros(len(dataframe)) + series_name
                # Note: Adds unique identifier
                dataframe.insert(dataframe.shape[1], 'series_name', name_column.astype(int))
                # Note: Makes sure the new columns always have the same name
                column_names = {}
                for i in range(min_trees):
                    column_names[dataframe.columns[i+1]] = 'LM' + str(i)
                    # Todo: This is a hack. It requires that we know what the nuber of neighbours should be.
                    #  This has to be then indicated in the 1_make_records.sh script. Perhaps the mechanism of
                    #  naming the neighbours can be improved.
                dataframe = dataframe.rename(columns=column_names)
                extended_data.append(dataframe)
                # Note: last line appends all possible combinations of dendrometer signals
        counter += 1

        # Note: Regarding the size of the metadata array (number of rows), it should be noted that it remains the same
        #  except from the fact that a new column is introduced within this function with the unique numerical
        #  identification of the subplot. The number of entries in the df array/dataframe changes twice within the
        #  function. First it shrinks because we group all the dendrometer signals of a single subplot together (this
        #  is the 'data' array). However, it then grows back to its original size because we make copies of each element
        #  in the 'data' array corresponding to the number of dendrometers in the subplot. We then permute the
        #  dendrometer signals in each copy so that there is always a different signal in the second column of the
        #  dataframe. The dataframe is organised so that the first column is always the time stamp 'ts', followed by
        #  columns containing the dendrometer signals and the followed by all the signals from the environment.

    # Sec: 7 ********* Consistency check
    for e in extended_data:
        # TODO: write a description.
        indx = e['series_name'][0]  # Note: Gets the identifier for the tree/dendrometer
        temp = e.iloc[:, 1]  # Note: Extracts the time series of the dendrometer signal from dataframe
        a = [x for x in temp if str(x) != 'nan']  # Note: Gets rid of the NaNs.
        temp = df[df['series_name'] == indx]['LM']  # Note: Extracts the time series from the original daataframe
        b = [x for x in temp if str(x) != 'nan']
        total = sum(a) - sum(b)
        if len(a) == len(b) and int(total) != 0:  # Note: tatal is float so can be very small but not zero.
            raise Exception(
                f'The mergeing of timeseries did not work out. There seems to be an issue with the time alignment of '
                f'the rows according to the time stamps!')

    # Sec: 8 ********* Merge all the dataframes in the extended_data list into a single continuous dataframe
    result = pd.concat(extended_data, ignore_index=True, sort=False)

    return result, meta


# todo: the function 'permutations' is old and should be removed
def permutations(original_list, elements, min_el):
    # Todo: add the appropriate sereis_name to each dataframe
    permutation_list = []
    labels = original_list
    n = elements
    head = [labels[0]]
    middle = labels[1:n + 1]
    tail = labels[n + 1:]

    for i in range(n):
        temp_middle = []
        for j in range(n):
            temp_middle += [middle[(j + i) % n]]
        permutation_list.append(head + temp_middle[:min_el] + tail)

    return permutation_list

# sec END ##############################################################################################


# Sec: ----------------------------------------
# Sec: File processing functions
# Sec: ----------------------------------------

def clean_metadata(file):
    return 1

def load_dataframe(data_path, file_type='pkl', database=None):
    if file_type == 'rda':  # TODO: make sure that the rda file is loaded as a pandas dataframe. The resto of the codes depends on it.
        df = pyreadr.read_r(data_path)[database]
    elif file_type == 'pkl':
        df = pd.read_pickle(data_path)
    else:
        raise Exception('Input data file format not recognised. Use .rda or .pkl formats.')
    return df

def get_folders_to_process(dir_paths):
    folders = []
    for dir_path in dir_paths:
        files = dir_path.glob("*")
        n_files = sum(1 for _ in files)
        if n_files >= 500:
            folders.append(dir_path)
    return folders


def write_file_paths(sorted_files, output_file=Path("").joinpath("all_images.txt")):
    if output_file.exists():
        raise FileExistsError("Warning: file already exists.")

    with open(output_file, "w") as file:
        for line in sorted_files:
            file.write(line + "\n")


def make_dataframe_from_csv_stats(csv):
    df = pd.read_csv(csv, usecols=["0", "1", "2"])
    df = df.rename(columns={"1": "height", "2": "width"})
    return df


def make_list_from_df(df):
    all_images = df["0"].map(lambda x: Path(x).stem).values
    all_images_sorted = sorted(list(all_images), key=lambda x: int("".join(x.split("_")[:2])), reverse=False)
    return all_images_sorted


def save_list_from_df(df, out_path=Path("").joinpath("all_images.txt")):
    sorted_list_images = make_list_from_df(df)
    write_file_paths(sorted_list_images, output_file=out_path)


def load_list(input_path):
    lines = []
    with open(input_path, "r") as file:
        for line in file.readlines():
            lines.append(line.rstrip("\n"))
    return lines


# Sec: END ------------------------------------


# Sec: ----------------------------------------
# Sec: TFrecords
# Sec: ----------------------------------------


def get_schema(df, columns=None):
    schema = {}
    for col, val in df.to_dict().items():
        # It is only the second part of the dataframe (df) that has to be
        if columns and col not in columns:
            continue

        if isinstance(val, (list, np.ndarray)):
            schema[col] = (lambda f: lambda x: tf.train.FeatureList(feature=[f(i) for i in x]))(
                _get_feature_func(val[0]))
        else:
            schema[col] = (lambda f: lambda x: f(x))(_get_feature_func(val))
    return schema


def get_tfrecords(df, schema):
    for _, row in df.iterrows():
        features = {}
        feature_lists = {}

        for col, val in row.items():
            f = schema[col](val)
            # Note: schema is a function for each column of the dataframe. When you specify a colum with 'col', then
            #  the function will return a value for every 'val' in the type defined for that column.
            #  See https://www.machinelearningmindset.com/tfrecords-for-tensorflow/

            if type(f) is tf.train.FeatureList:
                feature_lists[col] = f

            if type(f) is tf.train.Feature:
                features[col] = f

        context = tf.train.Features(feature=features)
        if feature_lists:
            ex = tf.train.SequenceExample(
                context=context,
                feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
        else:
            ex = tf.train.Example(features=context)
        yield ex
        # Note: See https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def get_feature(val):
    if isinstance(val, (bytes, str, type(tf.constant(0)))):
        return _bytes_feature(val)

    if isinstance(val, (int, np.integer, bool, np.bool_)):
        return _int64_feature(val)

    if isinstance(val, (float, np.floating)):
        return _float_feature(val)

    raise Exception(f'Unsupported type {type(val)!r}')


def _get_feature_func(val):
    if isinstance(val, (bytes, str, type(tf.constant(0)))):
        return _bytes_feature

    if isinstance(val, (int, np.integer, bool, np.bool_)):
        return _int64_feature

    if isinstance(val, (float, np.floating)):
        return _float_feature

    raise Exception(f'Unsupported type {type(val)!r}')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""

    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if isinstance(value, str):
        value = value.encode('utf-8')
    # TODO: make sure the line above is correct. Perhaps it should be value = value.encode(str) ?
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""

    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Sec: END ------------------------------------
