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

    
