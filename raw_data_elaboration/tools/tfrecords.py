import numpy as np
import pyreadr
import pandas as pd
from functools import reduce
from pathlib import Path
import itertools
import tensorflow as tf
import random



# Sec: ----------------------------------------
# Sec: File processing functions
# Sec: ----------------------------------------


def load_dataframe(data_path, file_type='pkl', database=None):
    if file_type == 'rda':  # TODO: make sure that the rda file is loaded as a pandas dataframe. The resto of the codes depends on it.
        df = pyreadr.read_r(data_path)[database]
    elif file_type == 'pkl':
        df = pd.read_pickle(data_path)
    else:
        raise Exception('Input data file format not recognised. Use .rda or .pkl formats.')
    return df

def get_folders_to_process(dir_paths):
    folders = []
    for dir_path in dir_paths:
        files = dir_path.glob("*")
        n_files = sum(1 for _ in files)
        if n_files >= 500:
            folders.append(dir_path)
    return folders


def write_file_paths(sorted_files, output_file=Path("").joinpath("all_images.txt")):
    if output_file.exists():
        raise FileExistsError("Warning: file already exists.")

    with open(output_file, "w") as file:
        for line in sorted_files:
            file.write(line + "\n")


def make_dataframe_from_csv_stats(csv):
    df = pd.read_csv(csv, usecols=["0", "1", "2"])
    df = df.rename(columns={"1": "height", "2": "width"})
    return df


def make_list_from_df(df):
    all_images = df["0"].map(lambda x: Path(x).stem).values
    all_images_sorted = sorted(list(all_images), key=lambda x: int("".join(x.split("_")[:2])), reverse=False)
    return all_images_sorted


def save_list_from_df(df, out_path=Path("").joinpath("all_images.txt")):
    sorted_list_images = make_list_from_df(df)
    write_file_paths(sorted_list_images, output_file=out_path)


def load_list(input_path):
    lines = []
    with open(input_path, "r") as file:
        for line in file.readlines():
            lines.append(line.rstrip("\n"))
    return lines


# Sec: END ------------------------------------


# Sec: ----------------------------------------
# Sec: TFrecords
# Sec: ----------------------------------------


def get_schema(df, columns=None):
    schema = {}
    for col, val in df.to_dict().items():
        # It is only the second part of the dataframe (df) that has to be
        if columns and col not in columns:
            continue

        if isinstance(val, (list, np.ndarray)):
            schema[col] = (lambda f: lambda x: tf.train.FeatureList(feature=[f(i) for i in x]))(
                _get_feature_func(val[0]))
        else:
            schema[col] = (lambda f: lambda x: f(x))(_get_feature_func(val))
    return schema


def get_tfrecords(df, schema):
    for _, row in df.iterrows():
        features = {}
        feature_lists = {}

        for col, val in row.items():
            f = schema[col](val)
            # Note: schema is a function for each column of the dataframe. When you specify a colum with 'col', then
            #  the function will return a value for every 'val' in the type defined for that column.
            #  See https://www.machinelearningmindset.com/tfrecords-for-tensorflow/

            if type(f) is tf.train.FeatureList:
                feature_lists[col] = f

            if type(f) is tf.train.Feature:
                features[col] = f

        context = tf.train.Features(feature=features)
        if feature_lists:
            ex = tf.train.SequenceExample(
                context=context,
                feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
        else:
            ex = tf.train.Example(features=context)
        yield ex
        # Note: See https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def get_feature(val):
    if isinstance(val, (bytes, str, type(tf.constant(0)))):
        return _bytes_feature(val)

    if isinstance(val, (int, np.integer, bool, np.bool_)):
        return _int64_feature(val)

    if isinstance(val, (float, np.floating)):
        return _float_feature(val)

    raise Exception(f'Unsupported type {type(val)!r}')


def _get_feature_func(val):
    if isinstance(val, (bytes, str, type(tf.constant(0)))):
        return _bytes_feature

    if isinstance(val, (int, np.integer, bool, np.bool_)):
        return _int64_feature

    if isinstance(val, (float, np.floating)):
        return _float_feature

    raise Exception(f'Unsupported type {type(val)!r}')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""

    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if isinstance(value, str):
        value = value.encode('utf-8')
    # TODO: make sure the line above is correct. Perhaps it should be value = value.encode(str) ?
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""

    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Sec: END ------------------------------------
