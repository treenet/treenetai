import argparse
import json
import os
from pprint import pprint

import tensorflow as tf

import utils.utils as utils
from config import CONSTANTS as C
from config import Configuration
from utils.utils import setup_model_and_datasets


from sklearn.metrics import r2_score

def evaluate(config, store_eval, use_partial=False):

    model_instance, _, val_ds = setup_model_and_datasets(config)

    model = model_instance.make_model()
    
    if use_partial:
        model.load_weights(config.ckpt).expect_partial() # NOTE: use when fine-tuning from a larger model.
    else:
        model.load_weights(config.ckpt ) # NOTE: use when architecture and weights are a perfect match, i.e. exact model restoration.

    # SECTION: load_weights() from checkpoint with and without expect_partial()
    # ✅ No errors = full match: TensorFlow checks that all weights in the model are matched by weights in the checkpoint, and vice versa.
    # ⚠️ If you accidentally saved a partial checkpoint (e.g., only some layers), then load_weights() without .expect_partial() would raise an error.
    # 🧪 If you're ever unsure, you can inspect the return value of load_weights() — it returns a tf.train.CheckpointLoadStatus object, 
    # which has methods like .assert_consumed() and .assert_existing_objects_matched() for more detailed checks.



    # TODO: Try and use the coefficient of variation to evaluate the accuracy of predicition
    #  -----------------------------------------------------------------------------------------
    R2_metric = r2_score(multioutput="raw_values")
    # R2 score on validation set
    y_true = tf.concat([y for x, y in val_ds], axis=0)
    y_true = tf.reshape(y_true, shape=(-1, config.outputs))
    y_pred = model.predict(val_ds)
    R2_metric.update_state(y_true, y_pred)
    result = R2_metric.result()
    val_r2 = result.numpy()
    val_r2_mean = val_r2.mean().item()
    val_r2 = val_r2.tolist()
    print("Validation R2 score:", val_r2, "Avg: ", val_r2_mean)
    #  -----------------------------------------------------------------------------------------

    y_true = tf.concat([y for x, y in val_ds], axis=0)
    y_true = tf.reshape(y_true, shape=(-1, config.outputs))
    y_pred = model.predict(val_ds)

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
    parser.add_argument("model_id", help="Which models to evaluate.", type=str)
    parser.add_argument("experiment_type", help="Type of experiment (task) that produced the data.", type=str)
    parser.add_argument(
        "--store_eval",
        action="store_true",
        help="Store metric values from evaluation in experiment folder"
    )

    args = parser.parse_args()

    print(f"Evaluating model with id {args.model_id}")
    model_dir = utils.get_model_dir(C.EXPERIMENT_DIR, args.model_id)
    #assert model_dir is not None
    print('dir', model_dir)

    print("Model configuration:")
    model_config = Configuration.from_json(os.path.join(model_dir, "config.json"))
    pprint(vars(model_config))

    print('eval: ', args.store_eval)

    evaluate(model_config, args.store_eval)
