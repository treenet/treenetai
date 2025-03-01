import argparse
import json
import os
from pprint import pprint

import tensorflow as tf

import utils.utils as utils
from config import CONSTANTS as C
from config import Configuration
from utils.utils import setup_model_and_datasets


def evaluate(config, store_eval=True):

    model_instance, _, val_ds = setup_model_and_datasets(config)

    model = model_instance.make_model()
    model.load_weights(config.ckpt).expect_partial()

    # TODO: Try and use the coefficient of variation to evaluate the accuracy of predicition
    #  -----------------------------------------------------------------------------------------
    #  R2_metric = r2_score(multioutput="raw_values")
    #  R2 score on validation set
    #  y_true = tf.concat([y for x, y in val_ds], axis=0)
    #  y_true = tf.reshape(y_true, shape=(-1, config.outputs))
    #  y_pred = model.predict(val_ds)
    #  R2_metric.update_state(y_true, y_pred)
    #  result = R2_metric.result()
    #  val_r2 = result.numpy()
    #  val_r2_mean = val_r2.mean().item()
    #  val_r2 = val_r2.tolist()
    #  print("Validation R2 score:", val_r2, "Avg: ", val_r2_mean)
    #  -----------------------------------------------------------------------------------------

    y_true = tf.concat([y for x, y in val_ds], axis=0)
    y_true = tf.reshape(y_true, shape=(-1, config.outputs))
    y_pred = model.predict(val_ds)
    print('groudn truth: ', y_true)
    print('prediction: ', y_pred)

    if store_eval:
        path_store_eval = os.path.join(os.path.dirname(config.ckpt), "evaluation.json")
        print(f"Storing the validation scores in {path_store_eval}")
        with open(path_store_eval, "w") as file:
            data = {
                "validation_r2": val_r2,
                "validation_r2_mean": val_r2_mean,
                "validation_r2_tta": val_r2_tta,
                "validation_r2_tta_mean": val_r2_tta_mean,
            }
            s = json.dumps(data, indent=4, sort_keys=True)
            file.write(s)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True, help="Which models to evaluate.")
    parser.add_argument("--experiment_type", help="Type of experiment that produced the data.")
    parser.add_argument(
        "--store_eval",
        action="store_true",
        help="Store metric values from evaluation in experiment folder",
    )

    args = parser.parse_args()

    print(f"Evaluating model with id {args.model_id}")
    model_dir = utils.get_model_dir(C.EXPERIMENT_DIR, args.model_id)
    assert model_dir is not None

    print("Model configuration:")
    model_config = Configuration.from_json(os.path.join(model_dir, "config.json"))
    pprint(vars(model_config))

    evaluate(model_config, args.store_eval)
