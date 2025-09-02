"""
Some utility functions.

Copyright ETH Zurich, Manuel Kaufmann
"""
import glob
import os
import sys
import zipfile
import tensorflow as tf

from config import CONSTANTS as C
from raw_data_elaboration.tfrecords_load import (
    get_dataset_dim,
    get_dataset,
    preprocess_data,
)
from utils.get_model import get_model

sys.path.append("..")


def setup_model_and_datasets(config):
    ########################################################################
    # SECTION: Create datasets
    ########################################################################

    # NOTE: Load data
    # TODO: Make sure that the file name is stored in the logs (also in CometAI)
    # TODO: Make sure that the data_file_id in 2_train.sh corresponds to the one in 1_make_records.sh
    print('setting up model and datasets...')
    print('')
    print('path....  ', os.path.join(C.DATA_DIR, 'train_'+str(config.data_file_id)+'.tfrecords'))
    train_fn = os.path.join(C.DATA_DIR, 'train_'+str(config.data_file_id)+'.tfrecords')
    val_fn = os.path.join(C.DATA_DIR, 'validation_'+str(config.data_file_id)+'.tfrecords')

    train_ds = get_dataset(
        train_fn,
        batch_size=config.train_batch,
        shuffle_buffer_size=config.shuffle_buf_size,
        prefetch_buffer_size=config.prefetch_buf_size,
        num_parallel_calls=config.num_parallel_calls,
        shuffle=True,
        labelling_pattern=config.labelling_pattern,
    )
    print(train_ds.element_spec)

    val_ds = get_dataset(
        val_fn,
        batch_size=config.test_batch,
        shuffle_buffer_size=config.shuffle_buf_size,
        prefetch_buffer_size=config.prefetch_buf_size,
        num_parallel_calls=config.num_parallel_calls,
        shuffle=False,
        labelling_pattern=config.labelling_pattern,
    )

    (
        batch_size,
        segment_length,
        channels,
    ) = get_dataset_dim(val_ds, bool(config.labelling_pattern)) 
    # TODO: 1) make sure the labelling_pattern option works. 2) add the shape of the output also.
    
    config.input_segment_length = segment_length
    config.channels = channels

    ########################################################################
    # SECTION: Get model and define callbacks
    ########################################################################

    inputs = tf.keras.Input(shape=(segment_length, channels))
    
    # TODO: finish implementing the shape of the output signal in the same way as the shape of the input signal
    # outputs = tf.keras.Input(shape=(segment_length, channels))

    if config.aug_model_name:
        augmentation_model = get_model(model=config.aug_model_name, **vars(config))
        augmentation_model = augmentation_model.make_model()
    else:
        augmentation_model = None

    if config.reg_model_name:
        regression_model = get_model(model=config.reg_model_name, **vars(config))
    else:
        regression_model = tf.keras.layers.Dense(config.outputs)

    # NOTE: a model instance may expose a preprocessing function which we pass on to get_dataset
    model_instance = get_model(
        model=config.model_name,
        inputs=inputs,
        aug_model=augmentation_model,
        reg_model=regression_model,
        **vars(config)
    )

    ########################################################################
    # SECTION: Preprocess model
    ########################################################################

    train_ds = preprocess_data(
        train_ds,
        model_instance.preprocessing_function,
        num_parallel_calls=config.num_parallel_calls,
    )
    val_ds = preprocess_data(
        val_ds,
        model_instance.preprocessing_function,
        num_parallel_calls=config.num_parallel_calls,
    )

    return model_instance, train_ds, val_ds


def get_optimizer(optimizer, lr=0.001, train_batch=16):
    if optimizer == "big_transfer":
        # NOTE: define optimizer according to
        #  https://colab.research.google.com/github/google-research/big_transfer/blob/master/colabs/big_transfer_tf2.ipynb#scrollTo=3DiIrQFBhe9R
        SCHEDULE_LENGTH = 1500
        SCHEDULE_LENGTH = SCHEDULE_LENGTH * 512 / train_batch
        SCHEDULE_BOUNDARIES = [
            SCHEDULE_LENGTH * 0.3,
            SCHEDULE_LENGTH * 0.6,
            SCHEDULE_LENGTH * 0.9,
        ]
        lr = 0.003 * train_batch / 512
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=SCHEDULE_BOUNDARIES,
            values=[lr, lr * 0.1, lr * 0.001, lr * 0.0001],
        )
        opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    elif optimizer == "nadam":
        opt = tf.keras.optimizers.Nadam(learning_rate=lr)
    elif optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer {optimizer}")

    return opt


def create_model_dir(experiment_main_dir, experiment_id, model_summary):
    """
    Create a new model directory.
    :param experiment_main_dir: Where all experiments are stored.
    :param experiment_id: The ID of this experiment.
    :param model_summary: A summary string of the model.
    :return: A directory where we can store model logs. Raises an exception if the model directory already exists.
    Copyright ETH Zurich, Manuel Kaufmann
    """
    model_name = "{}-{}".format(experiment_id, model_summary)
    model_dir = os.path.join(experiment_main_dir, model_name)
    if os.path.exists(model_dir):
        raise ValueError("Model directory already exists {}".format(model_dir))
    os.makedirs(model_dir)
    return model_dir


def get_model_dir(experiment_dir, model_id):
    """
    Return the directory in `experiment_dir` that contains the given `model_id` string.
    Copyright ETH Zurich, Manuel Kaufmann
    """
    model_dirs = glob.glob(
        os.path.join(experiment_dir, str(model_id) + "-*"), recursive=False
    )
    return None if len(model_dirs) == 0 else model_dirs[0]


def export_code(file_list, output_file):
    """
    Stores files in a zip.
    Copyright ETH Zurich, Manuel Kaufmann
    """
    if not output_file.endswith(".zip"):
        output_file += ".zip"
    ofile = output_file
    counter = 0
    while os.path.exists(ofile):
        counter += 1
        ofile = output_file.replace(".zip", "_{}.zip".format(counter))
    zipf = zipfile.ZipFile(ofile, mode="w", compression=zipfile.ZIP_DEFLATED)
    for f in file_list:
        zipf.write(f)
    zipf.close()