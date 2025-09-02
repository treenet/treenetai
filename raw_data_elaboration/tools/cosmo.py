# TITLE: TOOLS FOR MANAGING THE COSMO WEATHER DATA #######################

def get_site_coordinates(site_list, metadata):
    sites = {}
    for e in site_list:
        temp_df = metadata[metadata.site_id == e]
        siteXcor = pd.to_numeric(temp_df.iloc[0].site_xcor)
        siteYcor = pd.to_numeric(temp_df.iloc[0].site_ycor)
        sites[e] = [siteXcor, siteYcor]
    return sites

def load_cosmo_grid(data_path : str, year : int, month : int, day : int) -> np.array:
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
    
    return cosmo_grid[:, :, :, [3, 4]]  # NOTE: returns only the temperature and dew temperature

def treenet_to_cosmo_grid_sites(data_path, site_coordinates):
    """Converts the TreeNet site ids to cosmo grid ids by matching the nearest coordinates.
        Input: TreeNet site ids and corresponding coordinates in the form of a dictionary.
        Output: COSMO grid ids.
    """
    with open(data_path + 'height_map.pkl', 'rb') as f:
        elevation_map = pickle.load(f)

    conversion = {}

    _, _, channels = elevation_map.shape
    # NOTE: elevation_map has the shape (# horizontal cells, # vertical cells, 3). For each grid point (horizontal cell, vertical cell), 
    # height map gives an array with 3 elements of the form (latitude, longitude, elevation above sea level)
    cosmo_coordinates_temp = elevation_map.reshape(-1, channels)[:, :2]
    cosmo_coordinates = cosmo_coordinates_temp[:,[1,0]]  # NOTE: the coordinates in the COSMO file are inverted. This line corrects it.

    for site_id, coord in site_coordinates.items():
        
        treenet_coordinates = np.array(coord) # Coordinates from metadata of a tree in Birmensdorf
       
        # NOTE: cosmo_coordinates is a list of coordinates of all cells in the grid. 
        all_dists = np.sqrt(((cosmo_coordinates[None, :, :] - treenet_coordinates[None, None, :])**2).sum(axis=2)) # NOTE: (1, num_pts) contains the distance of all pixels with respect to a chosen point.
        # Extract the index of location with minimum distance
        min_idx = all_dists.argmin()
        conversion[site_id] = min_idx
        
    return conversion

def get_site_cosmo_clima(site_coordinates, year_start, year_end, data_path):
    """Iterates through all the sites and extracts the hourly temperature and dew 
    temperature for all the data available in the data_path directory"""
    
    clima_cosmo = {new_list: [] for new_list in site_coordinates.keys()}  
    # NOTE: initialize the dictionary where the key is the site_id and the value is a list of multivariate climate data

    cosmo_ids = treenet_to_cosmo_grid_sites(data_path, site_coordinates)  # NOTE: for every TreeNet site, returns the closest COSMO grid id.

    # Get data from COSMO
    for year in range(year_start, year_end+1):
        dir_list = os.listdir(data_path + str(year))
        sorted_list = sorted(dir_list, key=lambda x: datetime.strptime(x.split('.')[0], "%Y%m%d")) # NOTE: This step is very important. It makes sure that the files are loaded in the correct order.
        for file in sorted_list: # 
            
            year = int(file[0:4])
            month = int(file[4:6])
            day = int(file[6:8])

            # NOTE: load data for the particular day and for all the pixels
            cosmo_grid = load_cosmo_grid(data_path, year, month, day)  # NOTE: map containing the 8 quantities for each pixel for each hour
            # NOTE: Reshape COSMO forecasts to flatten across spatial dimensions
            cosmo_flat = cosmo_grid.reshape(cosmo_grid.shape[0], -1, cosmo_grid.shape[3])  # NOTE: converts the 2D grid into a 1D array
            # NOTE: Use the min-distance index to extract COSMO variables for the desired location

            for site_id in site_coordinates.keys():
                min_index = cosmo_ids[site_id]
                cosmo_cell_climate = cosmo_flat[:, min_index, :]  # NOTE: numpy array with 24 rows and columns corresponding to the number of quantities chosen
                hour = 0
                for row in cosmo_cell_climate:
                    # NOTE: iterate the list and add the timestamp and convert dew temperature to relative humidity
                    value = row.tolist()
                    clima_cosmo[site_id].append([pd.to_datetime(datetime(year,month,day,hour)), value[0], metpy.relative_humidity_from_dewpoint(value[0] * units.degC, value[1] * units.degC).to('percent').magnitude])
                    # NOTE: make sure to use pd.to_datetime() funciton so that the timestamp is in the correct type, i.e. dtype=datetime64[ns]
                    hour += 1
    
    df_dictionary = {}
    for site, list in clima_cosmo.items():
        # NOTE: convert the dictionary of lists into a dictionary of dataframes. 
        df_dictionary[site] = pd.DataFrame(list, columns = ['ts', 'cosmo_temp', 'cosmo_rh'])
        
    return df_dictionary