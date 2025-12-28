import numpy as np
import pyreadr
import pandas as pd
from functools import reduce
from pathlib import Path
import itertools
import tensorflow as tf
import random


# Sec: ----------------------------------------
# Sec: Gap-filling
# Sec: ----------------------------------------


def add_gaps(segment, gap_size_days, gap_type, channels, time_resolution):
    """
    Adds gaps to segments of a time series
    """
    # TODO (gap_type): add the possibility of sampling gap sizes from a uniform and an exponential distribution
    # TODO: add the posibility of filling the gaps with -1 and not just NaN.

    segment_modified = segment.copy()
    gap_size = gap_size_days * 24 * time_resolution

    for channel in channels: # NOTE: 'for' loop over the requested channels
        start_index = np.random.randint(len(segment) - gap_size)
        # NOTE: selects a random point on the array to start with the gap
        if gap_type == 'constant':
            end_index = start_index + gap_size
        elif gap_type == 'uniform':
            end_index = start_index + gap_size  # TODO: not complete; gap size should be random
        elif gap_type == 'exponential':
            end_index = start_index + gap_size  # TODO: not complete; gap size should be random
        else:
            raise Exception(f'The distribution of gaps is unknown.')

        index_labels = segment_modified.index[start_index:end_index]
        segment_modified.loc[index_labels, channel] = np.nan

    return segment_modified



def weighted_random_subset(channels, main_channel):
    """
    Input: list of channels/physical properties/column names to choose from
    Output: list of randomly selected channels

    The number of channels that is returned by the function is greater than one. 
    The channel 'GRO' is always returned. The other channels are returned according
    to their probability weights. The number of channels that is restored is also 
    a random number. All these parameters are defined within the function. In order
    to change them, they have to be changed inside the function.
    """
    subset = [main_channel]
    remaining = channels[1:]

    # Generate weights for number of additional elements
    max_len = len(remaining)
    possible_lengths = list(range(0, max_len + 1))  # 0 to max_len

    # Example: weight = 5 for length=2, lower weights for others
    length_weights = [1 if i != 1 else 5 for i in possible_lengths]

    # Choose number of additional elements
    num_to_select = random.choices(possible_lengths, weights=length_weights, k=1)[0]

    # Priority weights based on position (higher priority = higher weight)
    priority_weights = list(reversed(range(1, len(remaining) + 1)))

    # Sample without replacement using priority weights
    selected = random.choices(remaining, weights=priority_weights, k=num_to_select)

    # Remove duplicates
    subset += list(set(selected))

    return subset

