#!/bin/bash

#######################################################################
# database: This is the name of the database to be used. Options are:
#           metadata, dendrometer, climate
#######################################################################

credentials_path=~/codes/treenetai/raw_data_elaboration/tools/config_server.yml

for database in metadata dendrometer climate; do
    path=~/data/treenet/server_data
    python3 ~/codes/treenetai/raw_data_elaboration/get_raw_data.py $database $path $credentials_path
done