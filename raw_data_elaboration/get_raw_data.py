import pyreadr
import pandas as pd
import tools.server_tools as server
import pickle
import argparse
import yaml

def write_text(file_path, text_to_append):
    try:
        with open(file_path, 'a') as file:
            file.write(text_to_append + '\n')
    except Exception as e:
        print("File write ERROR...")

def server_metadata(meta_path, credentials_path):
    write_text(meta_path+"/screen.log", "Getting the metadata...")
    with open(credentials_path, 'r') as f:
        credentials = yaml.full_load(f)

    server.get_database_info( "view_metadata",
                              credentials.get('user'), 
                              credentials.get('password'), 
                              credentials.get('host'), 
                              credentials.get('port'), 
                              credentials.get('dbname') )

    meta = server.get_metadata( credentials.get('user'), 
                                credentials.get('password'), 
                                credentials.get('host'), 
                                credentials.get('port'), 
                                credentials.get('dbname') )
    
    with open(meta_path+"/metadata_all.pkl", 'wb') as f:
        pickle.dump(meta, f)

    write_text(meta_path+"/screen.log", "Done...")

def server_data_dendrometer(database, data_path, credentials_path):
    """Connects to the TreeNet server and downloads the entire metadata table and all the non-empty time series data.
        The metadata is a pandas dataframe and the timeseries data is stored as a list of pandas dataframes."""
    
    write_text(data_path+"/screen_" + database + ".log", "Getting " + database + " ...")
    with open(credentials_path, 'r') as f:
        credentials = yaml.full_load(f)

    data = {}
    new_meta = []
    meta = server.get_metadata( credentials.get('user'), 
                                credentials.get('password'), 
                                credentials.get('host'), 
                                credentials.get('port'), 
                                credentials.get('dbname') )
    
    server.get_database_info( database,
                              credentials.get('user'), 
                              credentials.get('password'), 
                              credentials.get('host'), 
                              credentials.get('port'), 
                              credentials.get('dbname') )
    
    query = "SELECT series_id,ts,value FROM " + database + " WHERE series_id=%s ORDER BY ts" 
    
    for row in meta.iterrows():
        if row[1].variable_name == "tree stem radius change":  
            # NOTE: makes sure that only dendrometer data is selected. there is also other sensor metadata stored in the same file.
            print('series id: ' + str(row[1].series_id))
            series_id = row[1].series_id
            value = (series_id,)

            temp = server.get_data_element( value, 
                                            query,
                                            credentials.get('user'), 
                                            credentials.get('password'), 
                                            credentials.get('host'), 
                                            credentials.get('port'), 
                                            credentials.get('dbname') )
            if len(temp) > 10:      
                write_text(data_path+"/screen_" + database + ".log", 'series id: ' + str(series_id) + ' -> OK')
                data[series_id] = temp
                new_meta.append(row[1])
            else:
                write_text(data_path+"/screen_" + database + ".log", 'series id: ' + str(series_id) + ' -> empty or less than 10 rows')

    with open(data_path+"/"+database+"_dictionary.pkl", 'wb') as f:
        pickle.dump(data, f)

    new_meta = pd.DataFrame(new_meta)
    with open(data_path+"/metadata_" + database + ".pkl", 'wb') as f:
        pickle.dump(new_meta, f)

    write_text(data_path+"/screen_" + database + ".log", "Done...")


def server_data_climate(database, data_path, credentials_path):
    """Connects to the TreeNet server and downloads the entire metadata table and all the non-empty time series data.
        The metadata is a pandas dataframe and the timeseries data is stored as a list of pandas dataframes."""
    
    write_text(data_path+"/screen_" + database + ".log", "Getting " + database + " ...")
    with open(credentials_path, 'r') as f:
        credentials = yaml.full_load(f)

    data = {}
    meta = server.get_metadata( credentials.get('user'), 
                                credentials.get('password'), 
                                credentials.get('host'), 
                                credentials.get('port'), 
                                credentials.get('dbname') )
    
    server.get_database_info( database,
                              credentials.get('user'), 
                              credentials.get('password'), 
                              credentials.get('host'), 
                              credentials.get('port'), 
                              credentials.get('dbname') )
    
    query = "SELECT * FROM " + database + " WHERE site_id=%s ORDER BY ts" 
    
    for row in meta.iterrows():
        print('site id: ' + str(row[1].site_id))
        site_id = row[1].site_id
        value = (site_id,)

        temp = server.get_data_element( value, 
                                        query,
                                        credentials.get('user'), 
                                        credentials.get('password'), 
                                        credentials.get('host'), 
                                        credentials.get('port'), 
                                        credentials.get('dbname') )
        if len(temp) > 10:      
            write_text(data_path+"/screen_" + database + ".log", 'site id: ' + str(site_id) + ' -> OK')

            data[site_id] = temp
        else:
            write_text(data_path+"/screen_" + database + ".log", 'site id: ' + str(site_id) + ' -> empty or less than 10 rows')

    with open(data_path + "/" + database + "_dictionary.pkl", 'wb') as f:
        pickle.dump(data, f)

    write_text(data_path+"/screen_" + database + ".log", "Done...")


def server_data(database, data_path, credentials_path):
    """ This download function is for the datasets with "all" in the name.
        Connects to the TreeNet server and downloads the entire metadata table and all the non-empty time series data.
        The metadata is a pandas dataframe and the timeseries data is stored as a list of pandas dataframes."""
    
    write_text(data_path+"/screen_" + database + ".log", "Getting " + database + " ...")
    with open(credentials_path, 'r') as f:
        credentials = yaml.full_load(f)

    temperature = {}
    humidity = {}
    swp = {}
    temperature_meta = []
    humidity_meta = []
    swp_meta = []

    meta = server.get_metadata( credentials.get('user'), 
                                credentials.get('password'), 
                                credentials.get('host'), 
                                credentials.get('port'), 
                                credentials.get('dbname') )
    
    server.get_database_info( database,
                              credentials.get('user'), 
                              credentials.get('password'), 
                              credentials.get('host'), 
                              credentials.get('port'), 
                              credentials.get('dbname') )
    
    query = "SELECT * FROM " + database + " WHERE series_id=%s ORDER BY ts"  
    
    for row in meta.iterrows():
        variable = row[1].variable_name
        if variable == "air temperature" or variable == "relative humidity" or variable == "soil water potential":  
            print('series id: ' + str(row[1].series_id))
            series_id = row[1].series_id
            value = (series_id,)

            temp = server.get_data_element( value, 
                                            query,
                                            credentials.get('user'), 
                                            credentials.get('password'), 
                                            credentials.get('host'), 
                                            credentials.get('port'), 
                                            credentials.get('dbname') )
            if len(temp) > 10:      
                write_text(data_path+"/screen_" + database + ".log", 'series id: ' + str(series_id) + ' -> OK')
                if variable == "air temperature":
                    temperature[series_id] = temp
                    temperature_meta.append(row[1])
                elif variable == "relative humidity":
                    humidity[series_id] = temp
                    humidity_meta.append(row[1])
                else:
                    swp[series_id] = temp
                    swp_meta.append(row[1])
            else:
                write_text(data_path+"/screen_" + database + ".log", 'series id: ' + str(series_id) + ' -> empty or less than 10 rows')

    with open(data_path + "/" + database + "_temperature_dictionary.pkl", 'wb') as f:
        pickle.dump(temperature, f)
    with open(data_path + "/" + database + "_humidity_dictionary.pkl", 'wb') as f:
        pickle.dump(humidity, f)
    with open(data_path + "/" + database + "_swp_dictionary.pkl", 'wb') as f:
        pickle.dump(swp, f)

    temperature_meta = pd.DataFrame(temperature_meta)
    with open(data_path+"/metadata_" + database + "_temperature.pkl", 'wb') as f:
        pickle.dump(temperature_meta, f)

    humidity_meta = pd.DataFrame(humidity_meta)
    with open(data_path+"/metadata_" + database + "_humidity.pkl", 'wb') as f:
        pickle.dump(humidity_meta, f)

    swp_meta = pd.DataFrame(swp_meta)
    with open(data_path+"/metadata_" + database + "_swp.pkl", 'wb') as f:
        pickle.dump(swp_meta, f)

    write_text(data_path+"/screen_" + database + ".log", "Done...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=str)
    parser.add_argument("path", type=str)
    parser.add_argument("credentials_path", type=str)
    args = parser.parse_args()

    if args.database == "metadata":
        server_metadata(args.path, args.credentials_path)
    elif args.database == "data_dendro_lm" or args.database == "data_dendro_l2":
        server_data_dendrometer(args.database, args.path, args.credentials_path)
    elif args.database == "data_meteo_lm" or args.database == "data_meteo_l2":
        server_data_climate(args.database, args.path, args.credentials_path)
    else:
        server_data(args.database, args.path, args.credentials_path)

    # TODO: create a function that puts all the channels together into a single multi-channel time series.
