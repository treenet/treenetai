#!/bin/bash

##############################################################################################################################################
# database:         This is the name of the database to be used. Options are:
#                   metadata, data_all_l0, data_all_l1, data_dendro_l2, data_dendro_lm, data_meteo_l2, data_meteo_lm
# variable_name:    Name of the variable inside the database. Options are:
#                   "tree stem radius change", "air temperature", "relative humidity"
##############################################################################################################################################

credentials_path=~/codes/treenetai/raw_data_elaboration/tools/config_server.yml
path=~/data/treenet/server_data

for database in metadata data_dendro_lm; do
    python3 ~/codes/treenetai/raw_data_elaboration/get_raw_data.py $database $path $credentials_path
    mv /home/lukovic/data/treenet/server_data/* /storage/lukovic/Data/FORWARDS/treenet/server_data
done

