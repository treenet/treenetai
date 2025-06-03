#!/bin/bash

# NOTE:                 O P T I O N S   &   I N S T R U C T I O N S
#-----------------------------------------------------------------------------------------------------------------------------#
# file_id:              unique identifier for the file containing the tfrecord data
#-----------------------------------------------------------------------------------------------------------------------------#
# file_type:            file format of the input files containing the data and metadata; .rda and .pkl for now.
#-----------------------------------------------------------------------------------------------------------------------------#
# data_file_path:       path to file with the input data
#-----------------------------------------------------------------------------------------------------------------------------#
# metadata_file_path:   path to file with the metadata    
#-----------------------------------------------------------------------------------------------------------------------------#
# tfrecords_dir_path:   path to directory where the tfrecords files will be stored
#-----------------------------------------------------------------------------------------------------------------------------#
# species:              tree species to consider. options are a single sepcies or "all" for all available tree species
#-----------------------------------------------------------------------------------------------------------------------------#
# segment_length:       enter the number of days for the segment lengths
#-----------------------------------------------------------------------------------------------------------------------------#
# time_resolution:      this is the time resolution of the input data. Enter the factor to multiply the entire segment length.
#                       e.g. The time resolution of the raw treenet data is 10min, therefore it would require 6 cells in the 
#                       vector to accomodate 1 hour. If the resolution of the raw data is 1hr, then one cell is enough.
#-----------------------------------------------------------------------------------------------------------------------------#
# data_channels:        the features measured over time to be considered (stem radius, temperature, etc...)
#                       There are more features than might be required. Fore example the series and site identifications (IDs) 
#                       are not necessary for training the model and should therefore not be included. For options, see the 
#                       section channels_to_fix below. IMPORTANT: always include the time stamp. It is necessary for other 
#                       functions. The time stamp will be automatically removed when the segments are created (make_segments() 
#                       function).                       
#-----------------------------------------------------------------------------------------------------------------------------#
# experiment_type:      'gap-filling' - gap-filling dendrometer data
#                       'climate-processing' - processing climate data from L1 to Lm
#                       (others in preparation)
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
#                       these are the channels of the time series that have to be trained for gap-filling.
#                       each channel is represented by the following strings:
#
#                       'stem_radius'   - dendrometer
#                       'temp'          - temperature
#                       'rh'            - relative humidity 
#                       'vpd'           - vapour pressure deficit 
#                       'rad'           - solar radiation
#                       'swp'           - soil water potential
#                       'total_precip'  - total precipitation
#-----------------------------------------------------------------------------------------------------------------------------#
# normalization:        true or false for normalization of the data (partial TODO)
#-----------------------------------------------------------------------------------------------------------------------------#
# notes:                add any notes to describe the particular experiment
#-----------------------------------------------------------------------------------------------------------------------------#
# NOTE: END

#########################################
# E N T E R the correct paths below
#########################################
dataPath=/storage/lukovic/Data/FORWARDS/treenet/server_data/processed/weather_data.pkl
metadataPath=/storage/lukovic/Data/FORWARDS/treenet/server_data/metadata_all.pkl
tfrecordsDirPath=~/data/treenet/tfrecords
#########################################

# TODO: START
#  1) channels_to_fix option has to be changed so that it requires a description such as dendrometer,
#  temperature, etc.. instead of numbers. Alternatively, the numbers could be kept so that each number, starting with
#  zero corresponds to the channels described under "--data_channels".
#  2) for now only one label can be used, specifically, the time series. The code has to be changed in order for the
#  other features to be included as labels.
#  3) The parameters should be stored in a file and loaded directly from it.
# TODO: END

python3 ~/codes/treenetai/raw_data_elaboration/tfrecord_make.py \
        --file_id 1 \
        --file_type 'pkl' \
        --data_file_path $dataPath \
        --metadata_file_path $metadataPath \
        --tfrecords_dir_path $tfrecordsDirPath \
        --species 'all' \
        --segment_length 30 \
        --time_resolution 1 \
        --data_channels '['ts','stem_radius','temp','rh','vpd','rad','swp','total_precip']' \
        --experiment_type 'climate-processing' \
        --data_split 0.2 \
        --random_state 1 \
        --gap_size 10 \
        --gap_type 'constant' \
        --channels_to_fix 0 \
        --species_to_ignore '['aria','robur']' \
        --normalization \
        --notes 'climate data processing trials' \

mv ~/data/treenet/tfrecords/* /storage/lukovic/Data/FORWARDS/treenet/tfrecords


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# DATA STRUCTURE of the output
# NOTE: The output looks like the following:
#
#  [['LM with gaps', 'weather data with gaps', 'soil data with gaps', 'extra features with gaps'],
#   ['LM', 'weather data', 'soil data', 'extra features'],
#   [metadata],
#   [rescaling constants for each channel] 
#  ]
#
# NOTE: The format of the list is [[sample1_data_np.array, sample1_label_np.array, sample1_metadata_pd.df.record, sample1_scale_constants_np.array],
#                                  [sample2_data_np.array, sample2_label_np.array, sample2_metadata_pd.df.record, sample1_scale_constants_np.array],
#                                  [sample3_data_np.array, sample3_label_np.array, sample3_metadata_pd.df.record, sample1_scale_constants_np.array],...