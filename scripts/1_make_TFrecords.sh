#!/bin/bash

# NOTE: O P T I O N S   &   I N S T R U C T I O N S
#-----------------------------------------------------------------------------------------------------------------------------#
# file_id:              unique identifier for the file containing the tfrecord data
#-----------------------------------------------------------------------------------------------------------------------------#
# file_type:            file format of the input files containing the data and metadata
#-----------------------------------------------------------------------------------------------------------------------------#
# data_file_path:       path to file with the input data
#-----------------------------------------------------------------------------------------------------------------------------#
# metadata_file_path:   path to file with the metadata    
#-----------------------------------------------------------------------------------------------------------------------------#
# tfrecords_dir_path:   path to directory where the tfrecords files will be stored
#-----------------------------------------------------------------------------------------------------------------------------#
# species:              tree species to consider. options are a single sepcies or "all" for all available tree species
#-----------------------------------------------------------------------------------------------------------------------------#
# segment_length:       enter the number of days
#-----------------------------------------------------------------------------------------------------------------------------#
# time_resolution:      depends on the resolution of the input data. Enter the factor to multiply the entire segment length.
#                       e.g. The time resolution of the raw treenet data is 10min, therefore it would require 6 cells in the 
#                       vector to accomodate 1 hour. If the resolution of the raw data is 1hr, then one cell is enough.
#-----------------------------------------------------------------------------------------------------------------------------#
# data_channels:        the features measured over time to be considered (stem radius, temperature, etc...)
#                       IMPORTANT!!! The features should be listed in such a way that the first channel(s) is the dendrometer
#                       signal and only then the weather and soil signals. TODO: This should be imporved and made fail-safe.
#-----------------------------------------------------------------------------------------------------------------------------#
# experiment_type:      gap-filling
#-----------------------------------------------------------------------------------------------------------------------------#
# data_split:           train:validation spllit of the data. the value indicated is the proportion used for validation
#-----------------------------------------------------------------------------------------------------------------------------#
# random_state:         this is the seed of the random number generator that is then used to split ranodmly the data into 
#                       train and test.
#-----------------------------------------------------------------------------------------------------------------------------#
# gap_size:             size in days of the gaps in the time series
#-----------------------------------------------------------------------------------------------------------------------------#
# gap_type:             constant: all the gaps used during training hve the same size, gap_size
#                       gaussian: random gap sizes used with average gap_size and variance ??? (TODO)
#-----------------------------------------------------------------------------------------------------------------------------#
# channels_to_fix:      this is necessary only if 'gap-filling' is used as the experiment type.
#                       these are the channels of the time series that have to be trained for gap-filling
#-----------------------------------------------------------------------------------------------------------------------------#
# normalization:        true or false for normalization of the data (partial TODO)
#-----------------------------------------------------------------------------------------------------------------------------#
# notes:                add any notes to describe the particular experiment
#-----------------------------------------------------------------------------------------------------------------------------#
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
#  3) The parameters should be stored in a file and loaded directly from it.
# TODO: END

python raw_data_elaboration/tfrecord_make.py \
        --file_id 1 \
        --file_type 'pkl' \
        --data_file_path $dataPath \
        --metadata_file_path $metadataPath \
        --tfrecords_dir_path $tfrecordsDirPath\
        --species 'abies' \
        --segment_length 30 \
        --time_resolution 6 \
        --data_channels 'series_id','ts','value' \
        --experiment_type 'gap-filling' \
        --data_split 0.2 \
        --random_state 1 \
        --gap_size 10 \
        --gap_type 'constant' \
        --channels_to_fix 0 \
        --normalization \
        --notes 'Reduced data. The input data contains no metadata.' \
