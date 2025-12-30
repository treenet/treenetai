
python3 build_coverage_report_treenet_utc.py \
  --out_root /home/lukovic/data/treenet/coverage_run \
  --metadata_pickle /storage/lukovic/Data/FORWARDS/treenet/server_data/metadata_all.pkl \
  --thermo_dir /storage/lukovic/Data/FORWARDS/treenet/server_data/thermometer_l1 \
  --hygro_dir  /storage/lukovic/Data/FORWARDS/treenet/server_data/hygrometer_l1 \
  --dendro_l2_dir /storage/lukovic/Data/FORWARDS/treenet/server_data/dendrometer_l2 \
  --dendro_lm_dir /storage/lukovic/Data/FORWARDS/treenet/server_data/dendrometer_lm \
  --years 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 \
  --tz Europe/Zurich \
  --stem_mode delta \
  --sites_csv /home/lukovic/data/treenet/train_sites.csv \
  --plots true \
  --debug true

