#!/bin/bash

# NOTE: O P T I O N S   &   I N S T R U C T I O N S
#----------------------------------------------------------------------------------------------------------------------#
# segment_length: enter the number of days
#----------------------------------------------------------------------------------------------------------------------#
# time_resolution: depends on the resolution of the input data. Enter the factor to multiply the entire segment length.
#                  e.g. The time resolution of the raw treenet data is 10min, therefore it would require 6 cells in the 
#                  vector to accomodate 1 hour. If the resolution of the raw data is 1hr, then one cell is enough.
#----------------------------------------------------------------------------------------------------------------------#
# data_channels: the features measured over time to be considered (stem radius, temperature, etc...)
#                IMPORTANT!!! The features should be listed in such a way that the first channel(s) is the dendrometer
#                signal and only then the weather and soil signals. TODO: This should be imporved and made fail-safe.
#----------------------------------------------------------------------------------------------------------------------#
# experiment_type: gap-filling
#----------------------------------------------------------------------------------------------------------------------#
# channels_to_fix: this is necessary only if 'gap-filling' is used as the experiment type.
#                  these are the channels of the time series that have to be trained for gap-filling
#----------------------------------------------------------------------------------------------------------------------#
# random_state: this is the seed of the random number generator that is then used to split ranodmly the data into train
#               and test.
#----------------------------------------------------------------------------------------------------------------------#
# file_id: unique identifier for the file containing the tfrecord data
#----------------------------------------------------------------------------------------------------------------------#
# notes: add any notes to describe the particular experiment
#----------------------------------------------------------------------------------------------------------------------#
# NOTE: END

dataPath=/storage/lukovic/Data/FORWARDS/treenet/raw_data/data_Server.pkl
metadataPath=/storage/lukovic/Data/FORWARDS/treenet/raw_data/metadata_server.pkl
tfrecordsDirPath=/home/lukovic/data/treenet/tfrecords

# TODO: START
#  1) channels_to_fix option has to be changed so that it requires a description such as dendrometer,
#  temperature, etc.. instead of numbers. Alternatively, the numbers could be kept so that each number, starting with
#  zero corresponds to the channels described under "--data_channels".
#  2) for now only one label can be used, specifically, the time series. The code has to be changed in order for the
#  other features to be included as labels.
# TODO: END

python raw_data_elaboration/tfrecord_make.py \
        --file_id 1 \
        --data_file_path $dataPath \
        --metadata_file_path $metadataPath \
        --tfrecords_dir_path $tfrecordsDirPath\
        --segment_length 10 \
        --time_resolution 6 \
        --data_channels 'series_id','ts','value' \
        --data_split 0.2 \
        --random_state 1 \
        --gap_size 10 \
        --gap_type 'constant' \
        --experiment_type 'gap-filling' \
        --channels_to_fix 0 \
        --file_type 'pkl' \
        --normalization \
        --species 'abies' \
        --notes 'Reduced data. The input data contains no metadata.' \
