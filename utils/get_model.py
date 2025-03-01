from models.autoencoder2D import autoencoder2D
from models.LSTM_CNN import LSTM_CNN
from models.LSTM_encoder import LSTM_encoder


def get_model(model, **kwargs):
    if model == "autoencoder2D":
        return autoencoder2D(**kwargs)
    elif model == "LSTM_CNN":
        return LSTM_CNN(**kwargs)
    elif model == "LSTM_encoder":
        return LSTM_encoder(**kwargs)
    else:
        raise ValueError(f"Unknown model_name {model}")
