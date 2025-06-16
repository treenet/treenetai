import pandas as pd
import argparse
import pickle
import os

import tools.data_processing_library as dpl
import tools.data_organisation as do


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("meta_path", type=str)
    parser.add_argument("meta_temperature_l1_path", type=str)
    parser.add_argument("meta_humidity_l1_path", type=str)

    parser.add_argument("temperature_l1_path", type=str)
    parser.add_argument("humidity_l1_path", type=str)
    parser.add_argument("clima_l2_path", type=str)
    parser.add_argument("clima_lm_path", type=str)
    parser.add_argument("cosmo_data_path", type=str)

    parser.add_argument("year_start", type=int)
    parser.add_argument("year_end", type=int)

    parser.add_argument("output_folder", type=str)
    
    args = parser.parse_args()

    metadata = dpl.load_dataframe(args.meta_path)
    meta_temperature_l1 = dpl.load_dataframe(args.meta_temperature_l1_path)
    meta_humidity_l1 = dpl.load_dataframe(args.meta_humidity_l1_path)
    
    temperature_l1 = dpl.load_dataframe(args.temperature_l1_path)
    humidity_l1 = dpl.load_dataframe(args.humidity_l1_path)
    clima_l2 = dpl.load_dataframe(args.clima_l2_path)
    clima_lm = dpl.load_dataframe(args.clima_lm_path)

    year_start = args.year_start
    year_end = args.year_end

    ################################################################# O L D - S T A R T ######################################################################
    # 1-OLD. Put together the raw weather data from the TreeNet database
    # NOTE: IMPORTANT!! This function is based on site_id and not series_id and is therefore deprecated. The idea was to collect signals from different sensors 
    # that measure the same quantity on the same site and take the average. We have decided not to do this, but rather to focus on single sensors alone. 
    # I leave it here only in case some parts of it might be necessary for other use. Perhaps, when required to make averages of different sensors on the same 
    # site, in another function.
    #def treenet_weather_data_combine_old():
    #    clima_l1 = {}
    #    # NOTE: consider only sites that have both sensors, for temperature and humidity. 
    #    #  The map(int, ...) function is used so as to avoid the np.int64 type prefix for each resulting element in the list.
    #    sites = list(set(list(map(int, meta_temperature_l1.site_id.unique()))) & set(list(map(int, meta_humidity_l1.site_id.unique())))) 
    #    for site in sites:
    #        
    #        temperature = []
    #        counter = 0
    #        for series in meta_temperature_l1[meta_temperature_l1.site_id == site].series_id:  # NOTE: there could be more than mone sensor for the same property on a site
    #            df_temp = temperature_l1[series][['ts', 'value']]
    #            df_temp.columns = ['ts', str(counter)]
    #            temperature.append(df_temp)
    #            counter = counter + 1
    #        
    #        humidity = []
    #        counter = 0
    #        for series in meta_humidity_l1[meta_humidity_l1.site_id == site].series_id:
    #            df_temp = humidity_l1[series][["ts", "value"]]
    #            df_temp.columns = ['ts', str(counter)]
    #            humidity.append(df_temp)
    #            counter = counter + 1
    #
    #        merged_temperature = do.merge_time_series(temperature)
    #        merged_temperature['value'] = merged_temperature.drop('ts', axis=1).mean(axis=1)
    #
    #        merged_humidity = do.merge_time_series(humidity)
    #        merged_humidity['value'] = merged_humidity.drop('ts', axis=1).mean(axis=1)
    #
    #        hourly_temperature = do.get_hourly_data(merged_temperature[['ts','value']])[['ts','value']].rename(columns={"value": "av_temperature"})
    #        hourly_humidity = do.get_hourly_data(merged_humidity[['ts', 'value']])[['ts','value']].rename(columns={"value": "av_humidity"})
    #
    #        average_df = pd.merge(hourly_temperature, hourly_humidity, on='ts', how='outer')
    #
    #        # NOTE: make sure that all the time stamps are present
    #        start = average_df.ts.iloc[0]
    #        end = average_df.ts.iloc[-1]
    #        complete_times = {'ts': pd.date_range(start=start, end=end, freq='1h')}   
    #        # NOTE: alternative: "10Min". For more information see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    #        average_df = pd.merge(average_df, pd.DataFrame(complete_times), on = "ts", how = "outer")
    #
    #        clima_l1[site] = average_df
    #
    #    return clima_l1
    ############################################################### O L D - E N D ########################################################################

    
    # NOTE: get sites with temperature sensors
    site_groups = meta_temperature_l1.groupby('site_id')
    temperature_sensors = {}
    for i, row in site_groups:
        temperature_sensors[i] = row.series_id

    # NOTE: get sites with humidity sensors
    site_groups = meta_humidity_l1.groupby('site_id')
    humidity_sensors = {}
    for i, row in site_groups:
        humidity_sensors[i] = row.series_id

    # NOTE: get sites that have all the necessary data
    sites = list(set(temperature_sensors.keys()) & set(humidity_sensors.keys()) & set(clima_lm.keys()) & set(clima_l2.keys()))

    site_coordinates = do.get_site_coordinates(sites, metadata)
    clima_cosmo = do.get_site_cosmo_clima(site_coordinates, year_start, year_end, args.cosmo_data_path)

    weather_data = []
    weather_data_identifiers = []
    for site in sites:
        for temp_id in temperature_sensors[site]:
            for humidity_id in humidity_sensors[site]:
                df_temp_raw = temperature_l1[temp_id][['ts', 'value']]  # NOTE: raw temperature
                df_temp_raw.columns = ['ts', 'temp_raw']
                df_rh_raw = humidity_l1[humidity_id][['ts', 'value']]  # NOTE: raw relative humidity
                df_rh_raw.columns = ['ts', 'rh_raw']
                df_temp_processed = clima_l2[site][['ts', 'temp']]  # NOTE: curated temperature
                df_temp_processed.columns = ['ts', 'temp_processed']
                df_rh_processed = clima_lm[site][['ts', 'rh']]  # NOTE: curated relative humidity
                df_rh_processed.columns = ['ts', 'rh_processed']

                # NOTE: the ground truth data might have NaNs at the beginning and end.
                df_temp_processed = df_temp_processed.dropna()  # NOTE: remove NaNs.
                # NOTE: make sure that all the time stamps are present
                start = df_temp_processed.ts.iloc[0]
                end = df_temp_processed.ts.iloc[-1]
                complete_times = {'ts': pd.date_range(start=start, end=end, freq='1h')}   
                # NOTE: alternative: "10Min". For more information see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
                df_temp_processed = pd.merge(df_temp_processed, pd.DataFrame(complete_times), on = "ts", how = "outer")

                # NOTE: First merge the input data
                input_df = pd.merge(df_temp_raw, df_rh_raw, how='outer', on='ts')
                # NOTE: Then merge the ground truth data
                ground_truth_df = pd.merge(df_temp_processed, df_rh_processed, how='left', on='ts')
                # NOTE: Then merge all the data into a single data frame
                treenet_df = pd.merge(input_df, ground_truth_df, how='right', on='ts')
                total_df = pd.merge(clima_cosmo[site], treenet_df, how='inner', on='ts')
                
                weather_data.append(total_df)
                weather_data_identifiers.append([site, temp_id, humidity_id])  # NOTE: this is the corresponding list that contains the site and series IDs

    with open(args.output_folder + "weather_data.pkl", 'wb') as f:
        pickle.dump(weather_data, f)
    with open(args.output_folder + "weather_data_site_ids.pkl", 'wb') as f:
        pickle.dump(weather_data_identifiers, f)
