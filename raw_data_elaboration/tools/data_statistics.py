""" """
import pandas as pd


class dataset(object):
    def __init__(self, data_filepath, metadata_filepath):
        self.df = pd.read_pickle(data_filepath)
        self.meta = pd.read_pickle(metadata_filepath)
        self.species = dict()
        self.sites = dict()
        self.time_series_properties = dict()
        self.series_length_in_years = []

    def get_species(self):
        genus_list = self.meta.tree_genus.drop_duplicates().to_list()
        for e in genus_list:
            temp_df = self.meta[self.meta.tree_genus == e]
            self.species[e] = [temp_df.tree_species.drop_duplicates().to_list(), len(temp_df)]
            # Todo: Include the number of species trees

        return self.species

    def get_sites(self):
        site_list = self.meta.site_name.drop_duplicates().to_list()
        for e in site_list:
            temp_df = self.meta[self.meta.site_name == e]
            species = temp_df.tree_species.drop_duplicates().to_list()
            species_count = []
            for ee in species:
                species_count.append(temp_df[temp_df.tree_species == ee].shape[0])
            siteXcor = pd.to_numeric(temp_df.iloc[0].site_xcor)
            siteYcor = pd.to_numeric(temp_df.iloc[0].site_ycor)
            self.sites[e] = [species, temp_df.shape[0], species_count, [siteXcor, siteYcor]]

        return self.sites

    def get_metadata_parameters(self):
        parameters = list(self.meta)
        return parameters

    def get_ts_stats(self):
        for e in self.df:
            if e.shape[0] > 0:
                series_id = e.series_id[0]
                length = e.shape[0]
                len_days = length / 144.0
                len_years = length / 52560.
                series_start = e.ts.iloc[0]
                series_end = e.ts.iloc[-1]
                meta_row = self.meta[self.meta.series_id == series_id]
                species = meta_row.tree_species
                site = meta_row.site_name
                self.time_series_properties[series_id] = [length, len_days, len_years, series_start, series_end, species, site]
        return self.time_series_properties

    def get_ts_length_distro(self):
        for e in self.time_series_properties.values():
            self.series_length_in_years.append(e[2])
        return self.series_length_in_years


if __name__ == "__main__":

    metadata_path = "/Users/lukovic/data/FORWARDS/TNT/raw_data/metadata_Server.pkl"
    data_path = "/Users/lukovic/data/FORWARDS/TNT/raw_data/data_Server.pkl"

    df = dataset(data_path, metadata_path)

