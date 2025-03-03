#!/bin/bash

source="local"
OUT_META_PATH="/storage/lukovic/Data/FORWARDS/treenet/raw_data/metadata_server.pkl"
OUT_DATA_PATH="/storage/lukovic/Data/FORWARDS/treenet/raw_data/data_server.pkl"
CREDENTIALS_PATH="raw_data_elaborations/tools/config_server.yml"

python raw_data_elaboration/load_and_convert_raw_data.py $source $OUT_META_PATH $OUT_DATA_PATH $CREDENTIALS_PATH