import pandas as pd
import argparse

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

    data = []
    for el in dendro:
        if len(el) > 0:
            data.append(el)
    

    df = {}
    
    for i in range(len(data)):
        if data[i].series_id == clima[i].series_id:
            a = pd.merge(dendro[i], clima[i], on = "ts", how = "outer")
            b = a [a.value.first_valid_index():a.value.last_valid_index()]
            c = b[["series_id", "ts", "value", "temp", "rh", "swp", "total_precip", "rad", "vpd"]].rename(columns={"value": "stem_radius"})
            start = c.ts.iloc[0]
            end = c.ts.iloc[-1]
            complete_times = {'ts': pd.date_range(start=start, end=end, freq='10Min')}

            d = pd.merge(c, pd.DataFrame(complete_times), on = "ts", how = "outer")
            df[int(d.iloc[0].series_id)] = d