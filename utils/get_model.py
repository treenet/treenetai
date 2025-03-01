from models.basic_regression_model import BasicRegressionModel
from models.simpleCNN import simpleCNN
from models.CNN5 import CNN5
from models.transfer_learner import TransferLearnerRegr
from models.transfer_learner_double import TransferLearnerDoubleRegr
from models.autoencoder2D import autoencoder2D
from models.CNN_LSTM_gapfill import CNN_LSTM_gapfill
from models.CNN_LSTM_reconstruction import CNN_LSTM_reconstruction


def get_model(model, **kwargs):
    if model == "simpleCNN":
        return simpleCNN(**kwargs)
    elif model == "CNN5":
        return CNN5(**kwargs)
    elif model == "TransferLearnerRegr":
        return TransferLearnerRegr(**kwargs)
    elif model == "TransferLearnerDoubleRegr":
        return TransferLearnerDoubleRegr(**kwargs)
    elif model == "BasicRegressionModel":
        return BasicRegressionModel(**kwargs)
    elif model == "autoencoder2D":
        return autoencoder2D(**kwargs)
    elif model == "CNN_LSTM_gapfill":
        return CNN_LSTM_gapfill(**kwargs)
    elif model == "CNN_LSTM_reconstruction":
        return CNN_LSTM_reconstruction(**kwargs)
    elif model == "dendrometer":
        return CNN_LSTM_gapfill(**kwargs)
    else:
        raise ValueError(f"Unknown model_name {model}")
