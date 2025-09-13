#!/bin/bash

########################################## D E S C R I P T I O N ###################################################
# 
# Evaluation process:
# -------------------
# The input consists of a single temperature signal and relative humidity signal from the same site. 
# Therefore, the input must come in pairs. Besides the signals, the coordinates of the sensors or site is required. 
# It is assumed that both sensors have the same coordinates. These coordinates are then used to 
# determine the temperature and relative humidity from outside products.
#
# Training process:
# -----------------
# During the training, the raw temperature and relative humidity signals are combinatorily paired together and then
# the outside climate data is added for those coordinates. For example, if a certain site has 3 temperature sensors
# [t1, t2, t3] and 2 relative humidity signals [h1, h2], then the data from this site will look like the following 
# (ot = outside temperature, oh = outside relative humidity):
# s1 -> [t1, h1, ot, oh]
# s2 -> [t1, h2, ot, oh]
# s3 -> [t2, h1, ot, oh]
# s4 -> [t2, h2, ot, oh]
# s5 -> [t3, h1, ot, oh]
# s6 -> [t3, h2, ot, oh]
# Therefore, there will be 6 input points in this case. The important thing is that the input structure is always
# constant and equal to 4 in this case. 
# The output is a 2 channel time series, that contains the corrected and improved temperature and relative humidity
# signal.
####################################################################################################################

meta_path="/storage/lukovic/Data/FORWARDS/treenet/server_data/metadata_all.pkl"
meta_temperature_l1="/storage/lukovic/Data/FORWARDS/treenet/server_data/metadata_data_all_l1_temperature.pkl"
meta_humidity_l1="/storage/lukovic/Data/FORWARDS/treenet/server_data/metadata_data_all_l1_humidity.pkl"

temperature_l1="/storage/lukovic/Data/FORWARDS/treenet/server_data/data_all_l1_temperature_dictionary.pkl"
humidity_l1="/storage/lukovic/Data/FORWARDS/treenet/server_data/data_all_l1_humidity_dictionary.pkl"

clima_l2="/storage/lukovic/Data/FORWARDS/treenet/server_data/data_meteo_l2_dictionary.pkl"
clima_lm="/storage/lukovic/Data/FORWARDS/treenet/server_data/data_meteo_lm_dictionary.pkl"
cosmo_data_folder="/storage/lukovic/Data/FORWARDS/treenet/COSMO_FromCirrus/"

year_start=2022
year_end=2023

output_folder="/home/lukovic/data/treenet/server_data/processed/"

python3 ~/codes/treenetai/raw_data_elaboration/prepare_data_for_data-cleaning.py $meta_path $meta_temperature_l1 $meta_humidity_l1 $temperature_l1 $humidity_l1 $clima_l2 $clima_lm $cosmo_data_folder $year_start $year_end $output_folder

mv /home/lukovic/data/treenet/server_data/processed/* /storage/lukovic/Data/FORWARDS/treenet/server_data/processed/