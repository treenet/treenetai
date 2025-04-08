import pandas as pd
import argparse
import pickle

import data_processing_library as dpl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument("dendro_path", type=str)
    parser.add_argument("clima_path", type=str)
    parser.add_argument("output_folder", type=str)
    args = parser.parse_args()

    meta = dpl.load_dataframe(args.meta_path)
    dendro = dpl.load_dataframe(args.dendro_path)
    clima = dpl.load_dataframe(args.clima_path)

    meta_id = {}
    # NOTE: iterate over the metadata table and convert it to a dictionary in which the series_id is the key.
    for index, row in meta.iterrows():    
        meta_id[row.series_id] = row

    df = {} # Initialize dictionary
    
    for key, data in dendro.items():
        print(key)
        climate_data = clima[meta_id[key].site_id]
        if len(data) > 0 and len(climate_data) > 0:
            a = pd.merge(data, climate_data, on = "ts", how = "outer")  # NOTE: merge the dendrometer data with the climate data
            b = a [a.value.first_valid_index():a.value.last_valid_index()]  # NOTE: remove the head and tail where "value" (in this case, the dendrometer value) is not defined
            c = b[["series_id", "site_id", "ts", "value", "temp", "rh", "swp", "total_precip", "rad", "vpd"]].rename(columns={"value": "stem_radius"})  # NOTE: select the features to use
            c['site_id'] = c['site_id'].ffill().bfill()  # NOTE: the merging process changes the site and seris ids to double because of the presence of NaNs. This command fills the rows with the corresponding index.
            c['series_id'] = c['series_id'].ffill().bfill()
            
            c['site_id'] = c['series_id'].astype(int) # NOTE: makes sure that the values are integers
            c['series_id'] = c['series_id'].astype(int)

            start = c.ts.iloc[0]
            end = c.ts.iloc[-1]
            complete_times = {'ts': pd.date_range(start=start, end=end, freq='10Min')} 
            d = pd.merge(c, pd.DataFrame(complete_times), on = "ts", how = "outer")  # NOTE: make sure that all the time stamps are present
            df[key] = d
    
    with open(args.output_folder+"/combined_dendro_climate_dictionary.pkl", 'wb') as f:
        pickle.dump(df, f)

    with open(args.output_folder+"/metadata_dictionary.pkl", 'wb') as f:
        pickle.dump(meta_id, f)