#!/bin/bash

##############################################################################################################################################
# database:         This is the name of the database to be used. Options are:
#                   metadata, data_all_l0, data_all_l1, data_dendro_l2, data_dendro_lm, data_meteo_l2, data_meteo_lm
# variable_name:    Name of the variable inside the database. Options are:
#                   "tree stem radius change", "air temperature", "relative humidity"
##############################################################################################################################################

THRESHOLD=$((400 * 1024 * 1024))
credentials_path=/home/lukovic/codes/treenetai/raw_data_elaboration/tools/config_server.yml
path=/home/lukovic/data/treenet/server_data

task() {
    python3 /home/lukovic/codes/treenetai/raw_data_elaboration/get_raw_data.py $1 $2 $3 &
    PID=$!
    echo "${PID}"
    while true; do
        TOT_RAM_USAGE=$(free | awk '/Mem:/ {print $3}')
        if [ "$TOT_RAM_USAGE" -gt "$THRESHOLD" ]; then
            echo "RAM usage ($TOT_RAM_USAGE KB) exceeds the threshold ($THRESHOLD KB). Killing the process: ${PID}"
            kill -9 "$PID"
            exit 0
        else
            echo "Process with PID $PID, Total RAM usage is $TOT_RAM_USAGE KB."
        fi
        sleep 5
        if ! ps -p "$PID" > /dev/null 2>&1; then
            echo "Process with PID $PID is no longer running."
            break
        fi
    done
}


for database in metadata data_dendro_l2; do
    echo $database
    task $database $path $credentials_path
    mv /home/lukovic/data/treenet/server_data/* /storage/lukovic/Data/FORWARDS/treenet/server_data
done

