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

def make_segments(df, length_in_days, stride, time_resolution):
    """
    Input: pandas data frame
    Output: pandas data farame

    # - Creates segments of fixed length
    # - Removes time stamp from data frame
    # - Normalizes if True
    # - Converts data frame to numpy array through the _rescale function, i.e. only if normalization is True
    """

    if df.shape[0] < 2 or df.shape[1] < 2:  # NOTE: makes sure that the dataset is not empty
        raise Exception('The data used for the experiment is not of the correct format. The data is either '
                        'empty or there is only one time series channel.')

    df = df.dropna()  # Note: remove all missing data from the dataframe
    segment_length = length_in_days * 24 * time_resolution  # Note: converts the segment length from days to time-samps
    stride_length = stride * 24 * time_resolution

    idx = 0
    max_length = len(df)

    segments = []
    while (idx + segment_length) < max_length:
        if (df.index[segment_length + idx] - df.index[idx] == segment_length):
            segments.append(df.iloc[idx:idx + segment_length])
            idx += stride_length
        else:
            idx += 1
    return segments


def normalize_dataframe(df):
    """
    Input: Pandas data frame with multiple columns (not necessarily numeric)
    Output: Pandas data frame where the values of the numeric columns are normalized according to column.
    """
    # TODO (Important!!) Maybe it is better to normalize the entire time series and not the segments separately.
    #  If we consider channels of a segment that have a constant value in that time period but that nevertheless
    #  change over a longer time period, then giving them all a value of 0.5 does not really make sense.
    # TODO consider adding the case where the signal is not normalized but is, nevertheless,
    #  shifted so that the minimum value is zero.

    if len(df.shape) != 2:
        raise Exception(f'_rescale_all() in data_processing_library.py The dimension of the segment array should be 2. '
                        f'In this case the shape of the array is ' + str(df.shape))

    numeric_columns = df.select_dtypes(include='number').columns
    
    minima = {}
    differences = {}
    df_normalized = df.copy()

    for e in numeric_columns:
        if e == 'hour':
            df_normalized[e] = df[e]/24.0
            minima[e] = 0.0
            differences[e] = 24.0
        elif e == 'doy':
            df_normalized[e] = df[e]/365.0 # TODO: leap years are not considered. should not create a significant error. Try to improve.
            minima[e] = 1.0
            differences[e] = 365.0
        elif e == 'month':
            df_normalized[e] = df[e]/12.0
            minima[e] = 0.0
            differences[e] = 12.0
        else:
            min_val = df[e].min(skipna=True)
            max_val = df[e].max(skipna=True)
            minima[e] = min_val
            if pd.isna(min_val) or pd.isna(max_val): # If True, then all the values in the column are NaNs
                df_normalized[e] = df[e] # return the original column values, i.e. leave the NaNs.
                differences[e] = pd.NA
            else:
                difference = max_val - min_val
                if np.abs(difference) > 1e-4:
                    # NOTE: The Pandas library functions are aware of the NaNs when performing arithmetic functions. Therefore, it is safe to keep 
                    #   them in the dataframe. It is not necessary to do anything about it. 
                    df_normalized[e] = (df[e] - min_val)/difference
                    differences[e] = difference
                else:
                    # NOTE: if the min/max difference is very small the signal is shifted, without division. this also avoids problems of division by zero
                    #       in the _normalize() function.
                    df_normalized[e] = df[e] - min_val
                    differences[e] = 1.0

    return df_normalized, minima, differences