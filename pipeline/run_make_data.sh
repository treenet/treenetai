
python3 simple_builder.py \
  --out_root /home/lukovic/data/treenet/outputs_yearly \
  --site_id 3 \
  --year 2019 \
  --tz Europe/Zurich \
  --thermo_dir /storage/lukovic/Data/FORWARDS/treenet/server_data/thermometer_l1 \
  --hygro_dir  /storage/lukovic/Data/FORWARDS/treenet/server_data/hygrometer_l1 \
  --dendro_l2_dir /storage/lukovic/Data/FORWARDS/treenet/server_data/dendrometer_l2 \
  --dendro_lm_dir /storage/lukovic/Data/FORWARDS/treenet/server_data/dendrometer_lm \
  --meteo_dir /storage/lukovic/Data/FORWARDS/treenet/meteo_data \
  --t_id 10 --h_id 8 --d_id 18 \
  --stem_mode absolute \
  --allow_missing_locals true \
  --min_local_coverage 0.50
