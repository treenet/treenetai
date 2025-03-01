import argparse
import tensorflow as tf
import pprint
import json
import os


class Constants(object):
    """
    This is a singleton.
    """

    class __Constants:
        def __init__(self):
            self.DATA_DIR = os.environ["DATA"]
            self.EXPERIMENT_DIR = os.environ["EXPERIMENTS"]

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)


CONSTANTS = Constants()


class Configuration:
    print('start_parsing')

    def __init__(self, adict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    @staticmethod
    def parse_args_function():
        parser = argparse.ArgumentParser()

        # sec: Data loading

        parser.add_argument(
            "--file_id", type=int, default=''
        )

        # sec: Model configurations

        parser.add_argument(
            "--model_name", type=str, default="simpleCNN"
        )
        parser.add_argument(
            "--experiment_type", type=str, default=None
        )
        parser.add_argument(
            "--model_init_ckpt",
            type=str,
            default="",
            help="Path to the checkpoint for the initialization of the model weights.",
        )
        parser.add_argument(
            "--timeseries_label",
            action="store_true",
            help="default is: use as a target variable",
        )
        parser.add_argument(
            "--site_xcor",
            action="store_true",
            help="default is: use as a target variable",
        )
        parser.add_argument(
            "--site_ycor",
            action="store_true",
            help="default is: use as a target variable",
        )
        parser.add_argument(
            "--series_name",
            action="store_true",
            help="default is: use as a target variable",
        )
        parser.add_argument(
            "--series_ancestor",
            action="store_true",
            help="default is: use as a target variable",
        )
        parser.add_argument(
            "--region",
            action="store_true",
            help="default is: use as a target variable",
        )
        parser.add_argument(
            "--site",
            action="store_true",
            help="default is: use as a target variable",
        )
        parser.add_argument(
            "--genus",
            action="store_true",
            help="default is: use as a target variable",
        )
        parser.add_argument(
            "--species",
            action="store_true",
            help="default is: use as a target variable",
        )
        parser.add_argument(
            "--site_temp_ref",
            action="store_true",
            help="default is: use as a target variable",
        )
        parser.add_argument(
            "--site_temp_location",
            action="store_true",
            help="default is: use as a target variable",
        )
        parser.add_argument(
            "--tree_proc_tol_out",
            action="store_true",
            help="default is: use as a target variable",
        )
        parser.add_argument(
            "--tree_proc_tol_jump",
            action="store_true",
            help="default is: use as a target variable",
        )
        parser.add_argument(
            "--tree_proc_frost_thr",
            action="store_true",
            help="default is: use as a target variable",
        )
        parser.add_argument(
            "--start_of_segment_ts",
            action="store_true",
            help="Time stamp at which the segment starts",
        )
        parser.add_argument(
            "--doy",
            action="store_true",
            help="Day of year at which the segment starts",
        )

        # sec: Data preprocessing configurations

        # sec: Transfer learning configurations
        parser.add_argument("--encoder", type=str, default="ResNet50")
        parser.add_argument("--feature_extractor_finetuning", action="store_true")
        parser.add_argument("--reg_model_name", type=str, default=None)

        # sec: Augmentation configurations
        parser.add_argument("--aug_model_name", type=str, default=None)

        # sec: Training configurations
        parser.add_argument("--train_batch", type=int, default=16)
        parser.add_argument("--test_batch", type=int, default=16)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument(
            "--patience",
            type=int,
            default=100,
            help="Epochs we wait for an improvement on the val ds before we stop training",
        )
        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
        parser.add_argument(
            "--optimizer",
            type=str,
            default="adam",
            choices=["adam", "nadam"],
        )

        # sec: Performance and other configurations
        parser.add_argument("--shuffle_buf_size", type=int, default=2048)
        parser.add_argument("--prefetch_buf_size", type=int, default=tf.data.AUTOTUNE)
        parser.add_argument("--num_parallel_calls", type=int, default=tf.data.AUTOTUNE)
        parser.add_argument("--verbose", type=int, default=2)

        # sec: Logging configurations
        parser.add_argument(
            "--api_key",
            type=str,
            default="tEQD7y9m4xMdhublSnGIG6AxY",
            help="Comet ML API key for experiment logging",
        )
        parser.add_argument(
            "--proj_name", type=str, default="TNT", help="Comet ML project name"
        )
        parser.add_argument(
            "--exp_key",
            type=str,
            default="",
            help="Comet ML experiment key to continue logging from an existing experiment",
        )
        parser.add_argument(
            "--exp_description",
            type=str,
            default="",
            help="Short description of the experiment for easier identification",
        )
        print('labelling_pattern')
        config = parser.parse_args()

        # sec: Labelling scheme

        config.outputs = (
                int(config.timeseries_label)
                + int(config.species)
                + int(config.site_xcor)
                + int(config.site_ycor)
                + int(config.series_name)
                + int(config.series_ancestor)
                + int(config.region)
                + int(config.site)
                + int(config.genus)
                + int(config.species)
                + int(config.site_temp_ref)
                + int(config.site_temp_location)
        )

        # sec: Labelling scheme

        config.labelling_pattern = {
            "timeseries_label": config.timeseries_label,
            "measure_point": config.species,
            "site_xcor": config.site_xcor,
            "site_ycor": config.site_ycor,
            "series_name": config.series_name,
            "series_ancestor": config.series_ancestor,
            "region": config.region,
            "site": config.site,
            "genus": config.genus,
            "species": config.species,
            "site_temp_ref": config.site_temp_ref,
            "site_temp_location": config.site_temp_location,
        }
        return Configuration(vars(config))

    @staticmethod
    def from_json(json_path):
        """Load configurations from a JSON file."""
        with open(json_path, "r") as f:
            config = json.load(f)

            return Configuration(config)

    def to_json(self, json_path):
        """Dump configurations to a JSON file."""
        with open(json_path, "w") as f:
            s = json.dumps(vars(self), indent=2, sort_keys=True)
            f.write(s)

    def update(self, adict):
        self.__dict__.update(adict)
