import pandas as pd
import numpy as np
import argparse
import pickle
import h5py

import tools.data_processing_library as dpl
import tools.data_organisation as do


import h5py


def load_cosmo_grid(data_path : str, year : int, month : int, day : int) -> np.array:
    """"Function to load the forecasts array for 1 day (24 hours) from file
        Already takes care of handling whether the COSMO forecast was a single prediction or an ensemble
        In case of an ensemble, only mean is returned
        The array is additionally reshaped so that hours are at dimension 0
        This means that this function always returns an array of shape (24, 224, 320, 8)
    """
    
    # Load data from file
    hf = h5py.File(data_path + '%04i/%04i%02i%02i.h5' % (year, year, month, day), 'r')
    hf.keys()
    cosmo_grid = np.array(hf.get('cosmo_grid'))

    # Take only mean prediction
    if len(cosmo_grid.shape) == 5:
        cosmo_grid = cosmo_grid[..., 0].transpose(2, 0, 1, 3)
    else:
        cosmo_grid = cosmo_grid.transpose(2, 0, 1, 3)
    cosmo_grid[:, :, :, [3, 4]] -= 273.15  # Convert temperatures from Kelvins into Degrees celsius
    
    return cosmo_grid 


if __name__ == "__main__":
    
    """ # TITLE: Prepare utility functions to load pre-processed COSMO data
        COSMO has been a single model until 17.9.2020, and an ensemble of 11 forecasts since 18.9.2020.
        Therefore, the <u>files until 17.9.2020</u> have the following shape:
        (224, 320, 24, 8) -> (grid_height, grid_width, hours, num_variables)
        and <u>files from 18.9.2020</u> have the following shape:
        (224, 320, 24, 8, 2) -> (grid_height, grid_width, hours, num_variables, ensemble_mean_and_stddev), where ensemble_mean is at location [0] and emsemble_stddev at dimension [1]
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument("clima_path", type=str)
    parser.add_argument("cosmo_folder_path", type=str)
    parser.add_argument("coordinate_file", type=str)
    parser.add_argument("year_start", type=int)
    parser.add_argument("year_end", type=int)
    parser.add_argument("output_folder", type=str)
    args = parser.parse_args()


    # The dictionary of features available in each file
    feature_name_to_index = {
        'tp': 0,            # Total Precipitaton        [mm]
        'ws': 1,            # Wind Speed                [m/s]
        'wdir': 2,          # Wind Direction            [°]
        '2t': 3,            # 2m Temperature            [°C]
        '2d': 4,            # 2m Dewpoint Temperature   [°C]
        'nswrs': 5,         # Net Shorware Radiation    [W/m^2]
        'nlwrs': 6,         # Net Longwave Radiation    [W/m^2]
        'vis': 7            # Visibility                (not sure what exactly this is for, I am not using it)
    }


    meta = dpl.load_dataframe(args.meta_path)
    clima = dpl.load_dataframe(args.clima_path)
    coordinates = dpl.load_dataframe(args.coordinate_file)
    height_map = dpl.load_dataframe(args.cosmo_folder_path + 'height_map.pkl')
    year_start = args.year_start
    year_end = args.year_end

    # TITLE: Locating closest point on the grid w.r.t. given GPS location
    # NOTE: first compute the distance map

    #location = np.array([47.362950, 8.454470]) # Coordinates from metadata of a tree in Birmensdorf

    for location in coordinates:
        for year in range(year_start, year_end):

            height, width, chans = height_map.shape
            hmap = height_map.reshape(-1, chans)[:, :2]
            all_dists = np.sqrt(((hmap[None, :, :] - location[None, None, :])**2).sum(axis=2)) # (1, num_pts)

            # TITLE: Get location of the target point (smallest distance) in the height_map array and use it to extract corresponding COSMO forecasts.

            # Extract the index of location with minimum distance
            min_idx = all_dists.argmin()
            # Get data from COSMO
            cosmo_grid = load_cosmo_grid(args.cosmo_folder_path, year, 9, 18)  
            # Reshape COSMO forecasts to flatten across spatial dimensions
            cosmo_flat = cosmo_grid.reshape(cosmo_grid.shape[0], -1, cosmo_grid.shape[3])
            # Use the min-distance index to extract COSMO variables for the desired location
            cosmo_location = cosmo_flat[:, min_idx, :]

    