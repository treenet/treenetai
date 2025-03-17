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
    
    with open(meta_path+"/metadata.pkl", 'wb') as f:
        pickle.dump(meta, f)

    write_text(meta_path+"/screen.log", "Done...")

def server_data_dendrometer(database, data_path, credentials_path):
    """Connects to the TreeNet server and downloads the entire metadata table and all the non-empty time series data.
        The metadata is a pandas dataframe and the timeseries data is stored as a list of pandas dataframes."""
    
    write_text(data_path+"/screen.log", "Getting " + database + " ...")
    with open(credentials_path, 'r') as f:
        credentials = yaml.full_load(f)

    data = []
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
        if row[1].tree_species == '-999':
            print('series id: ' + str(row[1].series_id) + ' -> empty')
        # else:
        series_id = row[1].series_id
        value = (series_id,)

        data.append(server.get_data_element(value, 
                                            query,
                                            credentials.get('user'), 
                                            credentials.get('password'), 
                                            credentials.get('host'), 
                                            credentials.get('port'), 
                                            credentials.get('dbname')
                                            ))
        write_text(data_path+"/screen.log", 'series id: ' + str(series_id) + ' -> OK')

    with open(data_path+"/"+database+".pkl", 'wb') as f:
        pickle.dump(data, f)

    write_text(data_path+"/screen.log", "Done...")

def server_data_climate(database, data_path, credentials_path):
    """Connects to the TreeNet server and downloads the entire metadata table and all the non-empty time series data.
        The metadata is a pandas dataframe and the timeseries data is stored as a list of pandas dataframes."""
    
    write_text(data_path+"/screen.log", "Getting " + database + " ...")
    with open(credentials_path, 'r') as f:
        credentials = yaml.full_load(f)

    data = []
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
    
    query = "SELECT site_id,ts,temp,rh,swp,total_precip,rad,vpd, vpd_bo FROM " + database + " WHERE site_id=%s ORDER BY ts" 
    
    for row in meta.iterrows():
        if row[1].tree_species == '-999':
            print('series id: ' + str(row[1].series_id) + ' -> empty')
        # else:

        series_id = row[1].series_id
        site_id = row[1].site_id
        value = (site_id,)

        data.append(server.get_data_element(value, 
                                            query,
                                            credentials.get('user'), 
                                            credentials.get('password'), 
                                            credentials.get('host'), 
                                            credentials.get('port'), 
                                            credentials.get('dbname')
                                            ))
        write_text(data_path+"/screen.log", 'series id: ' + str(series_id) + ' -> OK')

    with open(data_path+"/"+database+".pkl", 'wb') as f:
        pickle.dump(data, f)

    write_text(data_path+"/screen.log", "Done...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=str)
    parser.add_argument("path", type=str)
    parser.add_argument("credentials_path", type=str)
    args = parser.parse_args()

    if args.database == "metadata":
        server_metadata(args.path, args.credentials_path)
    elif args.database == "denrometer":
        server_data_dendrometer("data_dendro_lm", args.path, args.credentials_path)
    elif args.database == "climate":
        server_data_climate("data_meteo_l2", args.path, args.credentials_path)
    else:
        print("Error! Incorrect choice of database.")

    # TODO: create a function that puts all the channels together into a single multi-channel time series.
