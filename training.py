import comet_ml
# NOTE: import comet_ml at the top of your file
import datetime
import glob
import json
import os
import sys
from pprint import pprint
import numpy as np
import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # TODO: check why there is an error

from config import CONSTANTS as C
from config import Configuration
from utils.logger import ExperimentLogger
from utils.utils import get_optimizer, setup_model_and_datasets
import utils.utils as utils

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#print('')

#gpus = tf.config.list_physical_devices('GPU')
gpus = 0
if gpus:
    try:
        # NOTE: Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # NOTE: Memory growth must be set before GPUs have been initialized
        print(e)

# NOTE: START added for future reference - in case it is useful
# tf.config.set_soft_device_placement(True)
# tf.debugging.set_log_device_placement(True)
# NOTE: END


def train(config):
    # NOTE: Create a tf keras MirroredStrategy for MultiGpuSupport, https://keras.io/guides/distributed_training/
    #multigpu_strategy = tf.distribute.MirroredStrategy()
    #with multigpu_strategy.scope():
    model_instance, train_ds, val_ds = setup_model_and_datasets(config)
    model = model_instance.make_model()

    # TODO: Add a check here for the experiment type. Make sure that the data loaded is the one for the particular
    #  experiment.

    # NOTE: if they exist, load specific weights for model
    if config.model_init_ckpt:
        print(f"Load weights from {config.model_init_ckpt}")
        model.load_weights(config.model_init_ckpt)

    opt = get_optimizer(
        optimizer=config.optimizer, lr=config.lr, train_batch=config.train_batch
    )

    # NOTE: define loss function and evaluation metric
    # r2_metric = tf.keras.metrics.R2Score() # TODO: define a proper metric function
    r2_metric = None
    # model.compile(loss="mse", optimizer=opt, metrics=[r2_metric])
    model.compile(loss="mse", optimizer='adam')
    model.summary()

    ########################################################################
    # NOTE: Checkpoint setup
    ########################################################################

    experiment_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = model_instance.model_name()
    model_dir = utils.create_model_dir(C.EXPERIMENT_DIR, experiment_id, experiment_name)
    print('model directory: ', model_dir)

    # NOTE: Save code as zip into the model directory.
    code_files = glob.glob("./**/*.py", recursive=True)
    utils.export_code(code_files, os.path.join(model_dir, "code.zip"))

    # NOTE: Save the command line that was used.
    cmd = " ".join(sys.argv)
    config.cmd = cmd

    # NOTE: create checkpoint path in experiment directory and in config
    checkpoint_path = os.path.join(model_dir, "checkpoint.weights.h5")
    print(f"Store checkpoint at {checkpoint_path}")
    print(f"Set checkpoint path in config")
    config.ckpt = checkpoint_path

    # NOTE: Save config as json into the model directory.
    config.to_json(os.path.join(model_dir, "config.json"))

    if config.verbose > 0:
        print("Configuration:")
        pprint(vars(config))

    # NOTE: Setup Comet ML logger
    exp_logger = ExperimentLogger(config)

    # NOTE: Callbacks
    #  Add early stopping: stop training if the validation loss does not improve within PATIENCE epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.patience)

    # NOTE: Add model checkpoint callback: store model everytime validation loss reaches new minimum
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        # monitor="val_r_square", # TODO: restore the coefficient of variation as soon as it is available
        monitor="val_loss",
        # mode="max", # TODO: part of todo right above
        mode="min",
        verbose=config.verbose,
        save_best_only=True,
    )

    # NOTE: Save a data sample for inspection. Only the first batch is saved.
    print('Saving sample of input data...')
    print('Batch size: ', config.train_batch)
    print('Total number of batches: ', len(list(train_ds.as_numpy_iterator())))
    with open(model_dir + '/train_sample_input_' + str(config.file_id) + '.dat', 'a') as f:
        print('First batch shape (Labels):', list(train_ds.as_numpy_iterator())[0][0].shape)
        for element in list(train_ds.as_numpy_iterator())[0][0]:  # Note iterates through the first batch
            np.savetxt(f, element)
            f.write('\n')

    with open(model_dir + '/train_sample_label_' + str(config.file_id) + '.dat', 'a') as g:
        print('First batch shape (Labels):', len(list(train_ds.as_numpy_iterator())))
        for element in list(train_ds.as_numpy_iterator())[0][1]:
            np.savetxt(g, element)
            g.write('\n')
    f.close()
    g.close()

    ########################################################################
    # NOTE: Train model
    ########################################################################

    history = model.fit(
        train_ds,
        epochs=config.epochs,
        verbose=config.verbose,
        validation_data=val_ds,
        callbacks=[early_stopping, model_checkpoint_callback]
    )

    history_path = os.path.join(model_dir, "history.json")
    with open(history_path, "w") as f:
        # NOTE: history.history is dict containing val_acc amongst others
        s = json.dumps(history.history, indent=2, sort_keys=True)
        f.write(s)


if __name__ == "__main__":
    configuration = Configuration.parse_args_function()
    train(configuration)
