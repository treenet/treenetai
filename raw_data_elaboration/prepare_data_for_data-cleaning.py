import pandas as pd
import argparse
import pickle

import tools.data_processing_library as dpl
import tools.data_organisation as do


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("meta_path", type=str)
    parser.add_argument("meta_temperature_l1_path", type=str)
    parser.add_argument("meta_humidity_l1_path", type=str)

    parser.add_argument("temperature_l1_path", type=str)
    parser.add_argument("humidity_l1_path", type=str)

    parser.add_argument("year_start", type=str)
    parser.add_argument("year_end", type=str)

    parser.add_argument("output_folder", type=str)
    
    args = parser.parse_args()

    meta = dpl.load_dataframe(args.meta_path)
    meta_temperature_l1 = dpl.load_dataframe(args.meta_temperature_l1_path)
    meta_humidity_l1 = dpl.load_dataframe(args.meta_humidity_l1_path)
    temperature_l1 = dpl.load_dataframe(args.temperature_l1_path)
    humidity_l1 = dpl.load_dataframe(args.humidity_l1_path)
    dendro = dpl.load_dataframe(args.dendro_path)
    clima = dpl.load_dataframe(args.clima_path)
    
    year_start = args.year_start
    year_end = args.year_end

    ################################################################# O L D - S T A R T ######################################################################
    # 1-OLD. Put together the raw weather data from the TreeNet database
    # NOTE: IMPORTANT!! This function is based on site_id and not series_id and is therefore deprecated. I leave it here only in case some parts 
    # of it might be necessary for other use. Perhaps, when required to make averages of different sensors on the same site, in another function.
    clima_l1 = {}
    # NOTE: consider only sites that have both sensors, for temperature and humidity. 
    #  The map(int, ...) function is used so as to avoid the np.int64 type prefix for each resulting element in the list.
    sites = list(set(list(map(int, meta_temperature_l1.site_id.unique()))) & set(list(map(int, meta_humidity_l1.site_id.unique())))) 
    for site in sites:
        
        temperature = []
        counter = 0
        for series in meta_temperature_l1[meta_temperature_l1.site_id == site].series_id:  # NOTE: there could be more than mone sensor for the same property on a site
            df_temp = temperature_l1[series][['ts', 'value']]
            df_temp.columns = ['ts', str(counter)]
            temperature.append(df_temp)
            counter = counter + 1
        
        humidity = []
        counter = 0
        for series in meta_humidity_l1[meta_humidity_l1.site_id == site].series_id:
            df_temp = humidity_l1[series][["ts", "value"]]
            df_temp.columns = ['ts', str(counter)]
            humidity.append(df_temp)
            counter = counter + 1

        merged_temperature = do.merge_time_series(temperature)
        merged_temperature['value'] = merged_temperature.drop('ts', axis=1).mean(axis=1)

        merged_humidity = do.merge_time_series(humidity)
        merged_humidity['value'] = merged_humidity.drop('ts', axis=1).mean(axis=1)

        hourly_temperature = do.get_hourly_data(merged_temperature[['ts','value']])[['ts','value']].rename(columns={"value": "av_temperature"})
        hourly_humidity = do.get_hourly_data(merged_humidity[['ts', 'value']])[['ts','value']].rename(columns={"value": "av_humidity"})

        average_df = pd.merge(hourly_temperature, hourly_humidity, on='ts', how='outer')

        # NOTE: make sure that all the time stamps are present
        start = average_df.ts.iloc[0]
        end = average_df.ts.iloc[-1]
        complete_times = {'ts': pd.date_range(start=start, end=end, freq='1h')}   # NOTE: alternative: "10Min". For more information see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        average_df = pd.merge(average_df, pd.DataFrame(complete_times), on = "ts", how = "outer")

        clima_l1[site] = average_df
    ############################################################### E N D ########################################################################

    # 1. Put together the raw weather data from the TreeNet database
    
    # 2. Put together the data product from COSMO
    data_path = '/storage/lukovic/Data/FORWARDS/treenet/COSMO_FromCirrus/'
    site_coordinates = do.get_site_coordinates(sites, meta)
    clima_cosmo = do.get_site_cosmo_clima(site_coordinates, year_start, year_end, data_path)  # NOTE: dictionary (key = site_id, value = datarame [ts, temperature, relative humidity])

    # 3. Merge together the TreeNet data and the COSMO data

    input_df = {}
    for site in sites:
        clima_input = pd.merge(clima_cosmo[site], clima_l1[site], on = "ts", how = "outer")
        input_df[site] = clima_input[clima_input.cosmo_temp.first_valid_index():clima_input.cosmo_temp.last_valid_index()]  # NOTE: remove the head and tail where "value" is not defined

    # 4. Prepare the ground truth data from TreeNet (the Lm weather data)

    


    