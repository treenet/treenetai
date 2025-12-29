#!/bin/bash

python build_normalized_dataset_treenet.py \
  --out_root /home/lukovic/data/treenet/outputs_yearly \
  --metadata_pickle /storage/lukovic/Data/FORWARDS/treenet/server_data/metadata_all.pkl \
  --meteo_dir /storage/lukovic/Data/FORWARDS/treenet/server_data/climate \
  --thermo_dir /storage/lukovic/Data/FORWARDS/treenet/server_data/thermometer_l1 \
  --hygro_dir /storage/lukovic/Data/FORWARDS/treenet/server_data/hygrometer_l1 \
  --dendro_l2_dir /storage/lukovic/Data/FORWARDS/treenet/server_data/dendrometer_l2 \
  --dendro_lm_dir /storage/lukovic/Data/FORWARDS/treenet/server_data/dendrometer_lm \
  --train_site_ids_csv /home/lukovic/data/treenet/train_sites.csv \
  --test_site_ids_csv  /home/lukovic/data/treenet/test_sites.csv \
  --years 2014 2015 \
  --per_year true \
  --tz Europe/Zurich \
  --require_complete_locals false \
  --stem_mode delta \
  --input_mode combinations \
  --target_mode lm_site_median \
  --max_combos_per_site 64 \
  --min_local_coverage 0.70 \
  --min_lm_series 1 \
  --overlap_days 10
